"""The tools that answer questions rather than change things.

`tools.py` owns files, `actions.py` owns doing, and this owns knowing.  The
split is not tidiness: these are the tools that are safe to call speculatively,
so the persona can tell Jarvis to look before it answers without that advice
also licensing it to go and *do* something first.

Everything here reads `sensors.py` rather than measuring for itself, so the
number Jarvis says out loud is the same number the widget on screen is showing.
The one exception is anything that has to leave the machine, which is kept in
its own clearly-named tools -- `fetch_url` and `public_network` -- because
"what is my IP" and "what does the internet think my IP is" are different
questions with very different consequences, and a tool list is the wrong place
to be vague about which one is being asked.
"""

from __future__ import annotations

import fnmatch
import pathlib
import socket
import time

from . import sensors


class StatusError(Exception):
    """Something could not be read, said in a sentence."""


# How much of a fetched page is worth handing to a model by default, and the
# ceiling whatever it asks for. A page is context, and context is the scarce
# thing in a turn that also has 30 tools in it.
FETCH_DEFAULT, FETCH_CEILING = 6000, 40000
FETCH_TIMEOUT = 15.0

# Processes that nothing should be ending on an agent's say-so, whatever it
# was asked. Killing any of these either bluescreens Windows or logs the user
# out, and no phrasing of "close the thing using all the CPU" means that.
PROTECTED = {
    "system", "system idle process", "registry", "smss.exe", "csrss.exe",
    "wininit.exe", "winlogon.exe", "services.exe", "lsass.exe", "svchost.exe",
    "dwm.exe", "explorer.exe", "fontdrvhost.exe", "sihost.exe", "ctfmon.exe",
    "memory compression", "audiodg.exe", "python.exe", "pythonw.exe",
}


# -- the machine -----------------------------------------------------------


def system_info() -> str:
    """CPU, memory, battery, disks and uptime, with a minute of context."""
    cpu = sensors.cpu()
    if not cpu:
        raise StatusError("psutil is not installed, so I cannot read this machine.")
    ram = sensors.memory()
    lines = [f"CPU: {cpu['percent']:.0f}% now across {cpu['threads']} threads "
             f"({cpu.get('cores')} cores)"]
    if cpu.get("average_1m") is not None:
        lines.append(f"     {cpu['average_1m']:.0f}% average and "
                     f"{cpu['peak_1m']:.0f}% peak over the last minute")
    if cpu.get("mhz"):
        lines.append(f"     running at {cpu['mhz']:.0f} MHz of {cpu['max_mhz']:.0f}")
    lines.append(f"Memory: {ram['percent']:.0f}% used -- "
                 f"{sensors.human_bytes(ram['used'])} of "
                 f"{sensors.human_bytes(ram['total'])}, "
                 f"{sensors.human_bytes(ram['available'])} free")
    if ram.get("swap_total"):
        lines.append(f"Swap: {ram['swap_percent']:.0f}% of "
                     f"{sensors.human_bytes(ram['swap_total'])}")
    lines.append(battery_status())
    for disk in sensors.disks():
        lines.append(f"Disk {disk['device']}: "
                     f"{sensors.human_bytes(disk['free'])} free of "
                     f"{sensors.human_bytes(disk['total'])} "
                     f"({disk['percent']:.0f}% used)")
    lines.append(f"Hostname: {socket.gethostname()}")
    lines.append(f"Up for {sensors.human_span(sensors.uptime())}")
    return "\n".join(lines)


def battery_status() -> str:
    """Charge, whether it is plugged in, and how long it has left."""
    power = sensors.battery()
    if not power:
        raise StatusError("psutil is not installed, so I cannot read the battery.")
    if not power.get("present"):
        return "Battery: none -- this machine runs on mains power."
    parts = [f"Battery: {power['percent']:.0f}%",
             "charging" if power["plugged"] else "on battery"]
    if power.get("seconds_left"):
        parts.append(f"about {sensors.human_span(power['seconds_left'])} left")
    if power.get("percent_per_hour"):
        drift = power["percent_per_hour"]
        parts.append(f"{'gaining' if drift > 0 else 'losing'} "
                     f"{abs(drift):.1f}% an hour")
    return ", ".join(parts)


def disk_usage(path: str = "") -> str:
    """Free space, for one drive or all of them."""
    found = sensors.disks()
    if not found:
        raise StatusError("I could not read any drives on this machine.")
    if path:
        want = str(path).rstrip("\\/").lower()
        found = [d for d in found
                 if want in (d["device"].lower(), d["mount"].rstrip("\\/").lower())]
        if not found:
            raise StatusError(f"There is no drive matching '{path}'.")
    return "\n".join(
        f"{d['device']} ({d['filesystem'] or 'unknown'}): "
        f"{sensors.human_bytes(d['free'])} free of "
        f"{sensors.human_bytes(d['total'])}, {d['percent']:.0f}% used"
        for d in found)


def list_processes(by: str = "cpu", limit: int = 12) -> str:
    """What is running, heaviest first."""
    order = "memory" if str(by).lower().startswith("mem") else "cpu"
    rows = [r for r in sensors.processes(int(limit) + 2, order)
            if r["name"].lower() not in ("system idle process",)]
    if not rows:
        raise StatusError("I could not read the process list.")
    rows = rows[:max(1, int(limit))]
    out = [f"The {len(rows)} heaviest processes by {order}:"]
    for row in rows:
        out.append(f"  {row['name']} (pid {row['pid']}): "
                   f"{row['cpu']:.1f}% CPU, "
                   f"{sensors.human_bytes(row['memory'])} memory")
    return "\n".join(out)


def end_process(name: str = "", pid: int = 0) -> str:
    """Close a program, by name or by process id.

    Asks first and kills second: `terminate()` is a close request a program
    can act on, which lets an editor save what it was holding. Only if it is
    still there after a moment does this take the other route.
    """
    try:
        import psutil
    except ImportError:
        raise StatusError("psutil is not installed.") from None
    if not name and not pid:
        raise StatusError("Tell me which process -- a name or a pid.")

    targets = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if pid and proc.info["pid"] == int(pid):
                targets = [proc]
                break
            if name and name.lower() in (proc.info["name"] or "").lower():
                targets.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not targets:
        raise StatusError(f"Nothing is running called "
                          f"'{name or pid}'. Use list_processes to see what is.")

    # The guard is on what was *found*, not on what was asked for: "end the
    # thing on port 80" could resolve to lsass by accident, and the check has
    # to catch that too.
    for proc in targets:
        try:
            if (proc.info["name"] or "").lower() in PROTECTED or proc.info["pid"] < 10:
                raise StatusError(
                    f"{proc.info['name']} is part of Windows itself -- ending it "
                    f"would take the session down with it. I will not do that.")
        except psutil.NoSuchProcess:
            continue

    closed = []
    for proc in targets[:8]:
        try:
            label = f"{proc.info['name']} (pid {proc.info['pid']})"
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
                closed.append(f"{label} closed")
                continue
            except psutil.TimeoutExpired:
                proc.kill()
                closed.append(f"{label} would not close and was killed")
        except psutil.NoSuchProcess:
            closed.append(f"{label} had already gone")
        except psutil.AccessDenied:
            closed.append(f"{label} refused -- it is running as another user")
    return "\n".join(closed)


def sensor_history(series: str = "cpu", minutes: float = 2.0) -> str:
    """How a reading has behaved recently, rather than only right now."""
    key = str(series).lower().replace(" ", "_")
    aliases = {"net": "net_down", "download": "net_down", "upload": "net_up",
               "ram": "memory", "disk": "disk_read"}
    key = aliases.get(key, key)
    tap = sensors.sampler()
    if key not in tap.series:
        raise StatusError(f"There is no reading called '{series}'. "
                          f"I keep: {', '.join(sorted(tap.series))}.")
    points = tap.history(key)[-max(2, int(float(minutes) * 60 / tap.period)):]
    if len(points) < 2:
        raise StatusError(f"I have not been watching {key} long enough to say.")
    rate = key.startswith(("net_", "disk_"))
    show = sensors.human_rate if rate else (lambda v: f"{v:.0f}%")
    span = sensors.human_span(len(points) * tap.period)
    return (f"{key} over the last {span}: now {show(points[-1])}, "
            f"average {show(sum(points) / len(points))}, "
            f"lowest {show(min(points))}, highest {show(max(points))}")


# -- the network -----------------------------------------------------------


def network_status() -> str:
    """The connection, entirely from this machine -- nothing is asked online."""
    net = sensors.network()
    lines = []
    link = net["wifi"]
    if link.get("connected"):
        lines.append(f"Wi-Fi: \"{link['ssid']}\" on {link.get('band', '?')}, "
                     f"{link.get('radio', '')} {link.get('security', '')}".rstrip())
        if link.get("signal_percent") is not None:
            quality = ("excellent" if link["signal_percent"] >= 75 else
                       "good" if link["signal_percent"] >= 50 else
                       "weak" if link["signal_percent"] >= 30 else "very weak")
            lines.append(f"  Signal: {link['signal_percent']}% ({quality})")
        if link.get("rx_mbps"):
            lines.append(f"  Link rate: {link['rx_mbps']:.0f} Mbps down, "
                         f"{link.get('tx_mbps', 0):.0f} Mbps up")
    elif link:
        lines.append(f"Wi-Fi: {link.get('detail', 'not connected')}")

    for adapter in net["adapters"]:
        lines.append(f"{adapter['name']}: {adapter['address']} "
                     f"(netmask {adapter['netmask']}, "
                     f"{adapter['speed_mbps']} Mbps)")
    if net["gateway"]:
        lines.append(f"Gateway: {net['gateway']}")
    if net["online"]:
        lines.append(f"Internet: reachable, {net['latency_ms']:.0f} ms to 1.1.1.1")
    else:
        lines.append("Internet: NOT reachable -- nothing answered on the way out.")
    lines.append(f"Traffic right now: {sensors.human_rate(net['down_bps'])} down, "
                 f"{sensors.human_rate(net['up_bps'])} up")
    lines.append(f"Hostname: {net['hostname']}")
    return "\n".join(lines)


def ping(host: str, port: int = 0) -> str:
    """Whether something answers, and how quickly. Works on LAN names too."""
    if not str(host).strip():
        raise StatusError("Give me something to reach.")
    host = str(host).strip()
    # A bare hostname resolving at all is half the answer, and the half that
    # tells "the network is down" apart from "that name is wrong".
    try:
        address = socket.gethostbyname(host)
    except OSError as exc:
        why = exc.strerror or exc
        raise StatusError(f"{host} does not resolve to an address ({why}).") from None
    ports = [int(port)] if port else [443, 80, 22]
    for candidate in ports:
        ok, took = sensors.reachable(address, candidate, timeout=2.0)
        if ok:
            return (f"{host} ({address}) answered on port {candidate} "
                    f"in {took:.0f} ms")
    return (f"{host} resolves to {address}, but nothing answered on "
            f"{', '.join(str(p) for p in ports)}.")


def public_network() -> str:
    """What the internet sees: the public address, and who it belongs to.

    Kept apart from `network_status` because this one talks to a third party.
    The request carries nothing but the connection itself -- no name, no
    account, no identifier of the user -- but it is still a different thing
    from reading an adapter, and a tool list should say so.
    """
    try:
        import requests
    except ImportError:
        raise StatusError("requests is not installed.") from None
    try:
        answer = requests.get("https://ipinfo.io/json", timeout=8.0,
                              headers={"User-Agent": "jarvis-overlay"})
        answer.raise_for_status()
        data = answer.json()
    except Exception as exc:                                       # noqa: BLE001
        raise StatusError(f"I could not ask what the public address is ({exc}).") from None
    where = ", ".join(x for x in (data.get("city"), data.get("region"),
                                  data.get("country")) if x)
    return (f"Public address: {data.get('ip', 'unknown')}\n"
            f"Provider: {data.get('org', 'unknown')}\n"
            f"Appears to be in: {where or 'unknown'}")


def fetch_url(url: str, max_chars: int = FETCH_DEFAULT) -> str:
    """Fetch a web address from *this* machine and return it as text.

    The point of having this alongside whatever FreeClaw can already reach is
    that this request comes from here: a router's status page, a printer, a
    NAS, something on localhost. None of those are reachable from wherever
    FreeClaw happens to be running.
    """
    try:
        import requests
    except ImportError:
        raise StatusError("requests is not installed.") from None
    address = str(url).strip()
    if not address:
        raise StatusError("Give me an address to fetch.")
    if "://" not in address:
        address = "http://" + address
    if not address.lower().startswith(("http://", "https://")):
        raise StatusError("I can only fetch http and https addresses.")
    try:
        answer = requests.get(address, timeout=FETCH_TIMEOUT, allow_redirects=True,
                              headers={"User-Agent": "jarvis-overlay"})
    except Exception as exc:                                       # noqa: BLE001
        raise StatusError(f"{address} could not be fetched ({exc}).") from None

    kind = answer.headers.get("Content-Type", "")
    body = answer.text
    if "html" in kind.lower():
        body = _text_from_html(body)
    limit = min(max(int(max_chars), 200), FETCH_CEILING)
    clipped = body[:limit]
    note = "" if len(body) <= limit else f"\n... [{len(body) - limit:,} more characters]"
    return (f"{answer.status_code} {answer.reason} from {answer.url} "
            f"({kind or 'unknown type'})\n\n{clipped.strip()}{note}")


def _text_from_html(html: str) -> str:
    """Readable text out of a page, without pulling in a parser.

    Deliberately crude. This exists so that a router's status page is legible,
    not so that Jarvis can read the news -- FreeClaw's own web tools are
    better at that and are not restricted to this machine's network.
    """
    import re

    text = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?i)<(br|/p|/div|/tr|/li|/h[1-6])[^>]*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    import html as _html

    text = _html.unescape(text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


# -- finding things --------------------------------------------------------


def search_files(pattern: str, folder: str = "", limit: int = 40) -> str:
    """Find files by name under the folders Jarvis is allowed to see."""
    from .tools import ROOTS, _resolve

    if not str(pattern).strip():
        raise StatusError("Give me something to search for.")
    glob = str(pattern).strip()
    if not any(ch in glob for ch in "*?["):
        glob = f"*{glob}*"                    # a bare word means "contains"

    if folder:
        bases = [_resolve(folder)]
    else:
        bases = list(ROOTS)
    limit = min(max(int(limit), 1), 200)

    found, scanned, started = [], 0, time.monotonic()
    for base in bases:
        for path in base.rglob("*"):
            scanned += 1
            # A search is a convenience, not a job: a runaway walk of a
            # OneDrive tree would hold the whole turn open.
            if scanned > 200_000 or time.monotonic() - started > 10.0:
                break
            if fnmatch.fnmatch(path.name.lower(), glob.lower()):
                try:
                    size = path.stat().st_size if path.is_file() else 0
                except OSError:
                    continue
                found.append((path, size))
                if len(found) >= limit:
                    break
        if len(found) >= limit:
            break
    if not found:
        return f"Nothing matching '{pattern}' under " \
               f"{', '.join(str(b) for b in bases)}."
    out = [f"{len(found)} match{'' if len(found) == 1 else 'es'} for '{pattern}':"]
    for path, size in found:
        out.append(f"  {path}" + (f"  ({sensors.human_bytes(size)})" if size else ""))
    return "\n".join(out)


# name -> (function, description, {argument: (type, description, required)})
STATUS = {
    "system_info": (
        system_info,
        "CPU, memory, battery, disks and uptime for this machine, with how "
        "the last minute compared.", {}),
    "battery_status": (
        battery_status,
        "Battery charge, whether it is plugged in, and how long it has left.",
        {}),
    "disk_usage": (
        disk_usage, "Free space on the drives.",
        {"path": ("string", "One drive, e.g. C:. Omit for all of them", False)}),
    "list_processes": (
        list_processes, "What is running on this machine, heaviest first.",
        {"by": ("string", "cpu or memory", False),
         "limit": ("integer", "How many to list", False)}),
    "end_process": (
        end_process,
        "Close a running program by name or process id. Asks it to close "
        "first and forces it only if it will not. Refuses parts of Windows.",
        {"name": ("string", "Part of the program's name", False),
         "pid": ("integer", "Its process id, if known", False)}),
    "sensor_history": (
        sensor_history,
        "How a reading has behaved over the last few minutes rather than "
        "only right now: cpu, memory, net_down, net_up, disk_read, "
        "disk_write, battery.",
        {"series": ("string", "Which reading", False),
         "minutes": ("number", "How far back to look", False)}),
    "network_status": (
        network_status,
        "The network this machine is on: Wi-Fi name and signal, addresses, "
        "gateway, whether the internet is reachable and how fast traffic is "
        "flowing. Reads only this machine.", {}),
    "ping": (
        ping, "Check whether a host answers, and how quickly.",
        {"host": ("string", "A hostname or address", True),
         "port": ("integer", "A port to try; omit to try 443, 80 and 22", False)}),
    "public_network": (
        public_network,
        "The public IP address, provider and rough location, as the internet "
        "sees this connection. Asks a third-party service.", {}),
    "fetch_url": (
        fetch_url,
        "Fetch a web address from this machine and return it as text. Use it "
        "for things only this machine can reach -- a router, a printer, a NAS, "
        "something on localhost.",
        {"url": ("string", "The address to fetch", True),
         "max_chars": ("integer", "How much of it to return", False)}),
    "search_files": (
        search_files,
        "Find files by name in the user's documents, desktop, downloads and "
        "pictures.",
        {"pattern": ("string", "A name or a glob like *.pdf", True),
         "folder": ("string", "Search only here", False),
         "limit": ("integer", "How many matches to return", False)}),
}
