"""What this machine and its network are doing, sampled once a second.

Everything that answers "how are things right now" lives here, and nothing
here knows about MCP or about drawing.  Two very different things need the
same readings -- the tools FreeClaw calls, and the live widgets on the
overlay -- and if each fetched its own they would disagree, because a CPU
percentage is a measurement over an interval and two callers asking at the
same moment get two different intervals.

So there is one sampler, on one thread, at one cadence, and both sides read
its last answer.  That also makes history free: the sampler keeps a couple of
minutes of every series it takes, which is what lets a widget draw a graph
instead of a number, and what lets Jarvis answer "has it been like this for
long?" rather than only "what is it now".

**Rates versus levels.**  psutil reports network and disk traffic as counters
that only ever go up, so "3.4 MB/s" is not something you can read -- it is a
difference between two readings divided by the time between them.  Doing that
subtraction in the sampler, once, is why the tools and the widgets agree about
the number and why neither of them has to remember the last one.

**Cost.**  Everything sampled per second is a cheap counter read; together
they are well under a millisecond.  The two expensive readings are deliberately
not on that path: Wi-Fi details come from `netsh`, which is a process launch,
and the process table is a walk over a few hundred processes.  Both are cached
and refreshed on their own slower clocks, so asking for them a hundred times a
second costs what asking once costs.
"""

from __future__ import annotations

import collections
import re

import socket
import subprocess
import threading
import time

# How often the cheap counters are read, and how much history is kept. Two
# minutes at 1Hz is enough to see a spike arrive and pass, which is the
# question a glance at a widget is actually asking.
PERIOD = 1.0
HISTORY = 120

# The expensive readings, and how stale each is allowed to get. Wi-Fi details
# change on the timescale of walking across a room; the process table is
# worth having fresher than that but not every frame.
WIFI_TTL = 6.0
PROCESS_TTL = 3.0

# Windows hides console windows from subprocesses only if asked.
_NO_WINDOW = 0x08000000

_lock = threading.Lock()
_sampler = None


def _psutil():
    try:
        import psutil
    except ImportError:                                            # noqa: BLE001
        return None
    return psutil


# -- the things that need two readings to mean anything --------------------


class Sampler:
    """One thread, reading the cheap counters and remembering the answers."""

    def __init__(self, period: float = PERIOD, depth: int = HISTORY) -> None:
        self.period = period
        self.available = _psutil() is not None
        self.unavailable_reason = "" if self.available else "psutil is not installed."
        self.series: dict[str, collections.deque] = {
            key: collections.deque(maxlen=depth)
            for key in ("cpu", "memory", "net_down", "net_up", "disk_read",
                        "disk_write", "battery")}
        self.latest: dict[str, float] = {}
        self.started_at = time.time()
        self._stop = threading.Event()
        self._last_net = None
        self._last_disk = None
        self._last_at = None
        self._thread = None

    def start(self) -> bool:
        if not self.available or self._thread is not None:
            return self.available
        psutil = _psutil()
        psutil.cpu_percent(interval=None)        # prime: the first is garbage
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="jarvis-sensors",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.period):
            try:
                self._sample()
            except Exception:                                      # noqa: BLE001
                pass              # a sensor that throws must not end the run

    def _sample(self) -> None:
        psutil = _psutil()
        now = time.monotonic()
        gap = None if self._last_at is None else max(now - self._last_at, 1e-6)
        self._last_at = now

        reading = {"cpu": float(psutil.cpu_percent(interval=None)),
                   "memory": float(psutil.virtual_memory().percent)}

        net = psutil.net_io_counters()
        if self._last_net is not None and gap:
            reading["net_down"] = max(0.0, (net.bytes_recv - self._last_net.bytes_recv) / gap)
            reading["net_up"] = max(0.0, (net.bytes_sent - self._last_net.bytes_sent) / gap)
        self._last_net = net

        disk = psutil.disk_io_counters()
        if disk is not None and self._last_disk is not None and gap:
            reading["disk_read"] = max(0.0, (disk.read_bytes - self._last_disk.read_bytes) / gap)
            reading["disk_write"] = max(0.0, (disk.write_bytes - self._last_disk.write_bytes) / gap)
        self._last_disk = disk

        power = psutil.sensors_battery()
        if power is not None:
            reading["battery"] = float(power.percent)

        with _lock:
            for key, value in reading.items():
                self.series[key].append(value)
            self.latest.update(reading)

    # -- reading it back ---------------------------------------------------

    def now(self, key: str, default: float = 0.0) -> float:
        with _lock:
            return float(self.latest.get(key, default))

    def history(self, key: str) -> list[float]:
        with _lock:
            return list(self.series.get(key, ()))

    def mean(self, key: str, seconds: float = 60.0) -> float | None:
        """The average of the last `seconds` of a series, or None if empty."""
        points = self.history(key)[-max(1, int(seconds / self.period)):]
        return sum(points) / len(points) if points else None

    def peak(self, key: str, seconds: float = 60.0) -> float | None:
        points = self.history(key)[-max(1, int(seconds / self.period)):]
        return max(points) if points else None


def sampler() -> Sampler:
    """The one sampler, started on first use."""
    global _sampler
    if _sampler is None:
        _sampler = Sampler()
        _sampler.start()
    return _sampler


# -- formatting ------------------------------------------------------------


def human_bytes(count: float) -> str:
    """A byte count someone can read out loud."""
    step = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if step < 1024 or unit == "TB":
            return f"{step:.0f}{unit}" if unit == "B" else f"{step:.1f}{unit}"
        step /= 1024
    return f"{step:.1f}TB"


def human_rate(per_second: float) -> str:
    return f"{human_bytes(per_second)}/s"


def human_span(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h {seconds % 3600 // 60}m"
    return f"{seconds // 86400}d {seconds % 86400 // 3600}h"


# -- point-in-time readings ------------------------------------------------


def cpu() -> dict:
    psutil = _psutil()
    if psutil is None:
        return {}
    out = {"percent": sampler().now("cpu"),
           "cores": psutil.cpu_count(logical=False),
           "threads": psutil.cpu_count(logical=True),
           "average_1m": sampler().mean("cpu", 60.0),
           "peak_1m": sampler().peak("cpu", 60.0)}
    try:
        freq = psutil.cpu_freq()
        if freq is not None:
            out["mhz"], out["max_mhz"] = freq.current, freq.max
    except Exception:                                              # noqa: BLE001
        pass
    return out


def memory() -> dict:
    psutil = _psutil()
    if psutil is None:
        return {}
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {"percent": ram.percent, "used": ram.total - ram.available,
            "total": ram.total, "available": ram.available,
            "swap_percent": swap.percent, "swap_total": swap.total}


def battery() -> dict:
    psutil = _psutil()
    if psutil is None:
        return {}
    power = psutil.sensors_battery()
    if power is None:
        return {"present": False}
    out = {"present": True, "percent": power.percent,
           "plugged": bool(power.power_plugged)}
    # psutil reports "unknown" as a huge sentinel rather than None, and a
    # laptop on the charger reports it too -- so it is only meaningful while
    # actually discharging.
    left = power.secsleft
    if not power.power_plugged and left is not None and 0 <= left < 60 * 60 * 48:
        out["seconds_left"] = int(left)
    # How fast it is going, measured rather than guessed: the charge history
    # is already being kept, so the slope over the last few minutes is a
    # better answer than whatever the firmware's instantaneous estimate says.
    charge = sampler().history("battery")
    if len(charge) >= 30:
        span = (len(charge) - 1) * sampler().period
        drift = (charge[-1] - charge[0]) / max(span / 3600.0, 1e-6)
        if abs(drift) >= 0.5:
            out["percent_per_hour"] = round(drift, 1)
    return out


def disks() -> list[dict]:
    psutil = _psutil()
    if psutil is None:
        return []
    out = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue                          # an empty card reader, usually
        out.append({"device": part.device.rstrip("\\"), "mount": part.mountpoint,
                    "filesystem": part.fstype, "total": usage.total,
                    "free": usage.free, "percent": usage.percent})
    return out


def uptime() -> float:
    psutil = _psutil()
    return 0.0 if psutil is None else time.time() - psutil.boot_time()


# -- the network -----------------------------------------------------------


def _run(command: list[str], timeout: float = 6.0) -> str:
    try:
        done = subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout, creationflags=_NO_WINDOW)
        return done.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


_wifi_cache: tuple[float, dict] = (0.0, {})


def wifi() -> dict:
    """The wireless link, from `netsh`. Cached: it is a process launch."""
    global _wifi_cache
    when, cached = _wifi_cache
    if time.monotonic() - when < WIFI_TTL:
        return cached

    out = {}
    text = _run(["netsh", "wlan", "show", "interfaces"])
    # netsh pads with spaces and localises nothing on an English install, so
    # the labels are stable; the values are everything after the first colon.
    fields = dict(re.findall(r"^\s{2,}([A-Za-z][^:]*?)\s*:\s*(.+?)\s*$",
                             text, re.MULTILINE))
    if fields.get("State", "").lower() == "connected":
        out = {"connected": True,
               "ssid": fields.get("SSID", ""),
               "band": fields.get("Band", ""),
               "channel": fields.get("Channel", ""),
               "radio": fields.get("Radio type", ""),
               "security": fields.get("Authentication", ""),
               "adapter": fields.get("Description", "")}
        signal = re.match(r"(\d+)", fields.get("Signal", ""))
        if signal:
            out["signal_percent"] = int(signal.group(1))
        for label, key in (("Receive rate (Mbps)", "rx_mbps"),
                           ("Transmit rate (Mbps)", "tx_mbps")):
            try:
                out[key] = float(fields[label])
            except (KeyError, ValueError):
                pass
    elif text.strip():
        out = {"connected": False,
               "detail": fields.get("State", "not connected")}
    _wifi_cache = (time.monotonic(), out)
    return out


def gateway() -> str:
    """The default route's next hop, or "" if there is not one."""
    text = _run(["route", "print", "-4", "0.0.0.0"])
    found = re.search(r"^\s*0\.0\.0\.0\s+0\.0\.0\.0\s+(\S+)", text, re.MULTILINE)
    return found.group(1) if found else ""


def adapters() -> list[dict]:
    """Every network interface that is up and has an address."""
    psutil = _psutil()
    if psutil is None:
        return []
    stats = psutil.net_if_stats()
    out = []
    for name, addresses in psutil.net_if_addrs().items():
        state = stats.get(name)
        if state is None or not state.isup:
            continue
        v4 = [a for a in addresses if a.family == socket.AF_INET]
        if not v4:
            continue
        out.append({"name": name, "address": v4[0].address,
                    "netmask": v4[0].netmask, "speed_mbps": state.speed,
                    "mtu": state.mtu,
                    "loopback": v4[0].address.startswith("127.")})
    return out


def reachable(host: str = "1.1.1.1", port: int = 53, timeout: float = 1.5):
    """Whether the internet answers, and how long it took, in milliseconds.

    A TCP connect to a DNS resolver rather than a ping: ICMP is blocked often
    enough on hotel and campus networks that a failed ping means "maybe" where
    a failed connect to port 53 means something much closer to "no". It also
    needs no privileges and no subprocess.
    """
    start = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, (time.monotonic() - start) * 1000.0
    except OSError:
        return False, None


_online_cache: tuple[float, tuple] = (0.0, (False, None))
ONLINE_TTL = 5.0


def online(ttl: float = ONLINE_TTL):
    """Whether the internet answers, remembering the last answer briefly.

    `reachable` opens a socket, which is a few milliseconds and an actual
    packet on the wire. The network widget asks every time it redraws, which
    was ten times a second -- a connection to a DNS resolver ten times a
    second, for a number that cannot meaningfully change that fast. Whether
    the internet is up is a question worth asking every few seconds, not every
    frame.
    """
    global _online_cache
    when, cached = _online_cache
    if time.monotonic() - when < ttl:
        return cached
    answer = reachable()
    _online_cache = (time.monotonic(), answer)
    return answer


def network() -> dict:
    """Everything about the connection, without asking anyone on the internet."""
    down, up = sampler().now("net_down"), sampler().now("net_up")
    up_now, latency = online()
    return {"adapters": [a for a in adapters() if not a["loopback"]],
            "gateway": gateway(), "wifi": wifi(),
            "down_bps": down, "up_bps": up,
            "online": up_now, "latency_ms": latency,
            "hostname": socket.gethostname()}


# -- what is running -------------------------------------------------------


_process_cache: tuple[float, list] = (0.0, [])
# pid -> (total cpu seconds, when it was read). A process's CPU share is a
# rate, so one walk of the table cannot produce it -- psutil's own
# `cpu_percent` returns 0.0 the first time it is asked about any process, for
# exactly this reason, and a cached single walk therefore reports every
# process as idle forever. Keeping the previous reading turns the next walk
# into a real measurement instead of a second helping of zeros.
_cpu_marks: dict[int, tuple[float, float]] = {}


def processes(limit: int = 12, by: str = "cpu") -> list[dict]:
    """The heaviest processes. Cached, because it walks the whole table."""
    global _process_cache
    psutil = _psutil()
    if psutil is None:
        return []
    # The very first walk can only record marks, never read them, so a caller
    # that asks once would be told the machine is entirely idle. Walking twice
    # a third of a second apart costs that third of a second once per run and
    # makes the first answer a real one.
    if not _cpu_marks:
        _walk()
        time.sleep(0.35)
        cached = _walk()          # the pair, not the cache the first one left
    else:
        when, cached = _process_cache
        if time.monotonic() - when >= PROCESS_TTL:
            cached = _walk()
    key = "memory" if by.startswith("mem") else "cpu"
    ordered = sorted(cached, key=lambda r: r[key], reverse=True)
    return ordered[:max(1, int(limit))]


def _walk() -> list[dict]:
    """One pass over the process table, turning CPU counters into shares."""
    global _process_cache
    psutil = _psutil()
    if psutil is None:
        return []
    threads = psutil.cpu_count(logical=True) or 1
    now = time.monotonic()
    rows, seen = [], set()
    for proc in psutil.process_iter(["pid", "name", "memory_info",
                                     "cpu_times", "username"]):
        try:
            info = proc.info
            pid = info["pid"]
            seen.add(pid)
            times = info["cpu_times"]
            used = (times.user + times.system) if times else 0.0
            before = _cpu_marks.get(pid)
            _cpu_marks[pid] = (used, now)
            share = 0.0
            if before is not None:
                gap = now - before[1]
                if gap > 0.05:
                    # Divided by the thread count, so 100% means the whole
                    # machine rather than one core -- which is what anyone
                    # reading "Chrome: 300%" has to translate in their head.
                    share = max(0.0, (used - before[0]) / gap / threads * 100.0)
            rows.append({"pid": pid, "name": info["name"] or "?",
                         "cpu": share,
                         "memory": int(getattr(info["memory_info"], "rss", 0)),
                         "user": (info["username"] or "").split("\\")[-1]})
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue
    # Processes that have exited would otherwise accumulate marks forever.
    for dead in set(_cpu_marks) - seen:
        _cpu_marks.pop(dead, None)
    _process_cache = (now, rows)
    return rows
