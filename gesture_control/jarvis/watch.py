"""Standing orders: conditions Jarvis watches for and speaks up about.

A monitor on the screen is something *you* have to look at.  This is the other
half -- "tell me if the network drops" -- and it is the thing that makes an
assistant worth leaving running in the corner of a desk rather than something
you go and consult.

Three decisions make the difference between this being useful and being a
nuisance:

**It has to hold.**  A condition that has been true for one sample is noise.
CPU touches 100% every time anything launches, and an assistant that announces
each of those is one you turn off within the hour.  So a watch has to be
satisfied continuously for `FOR_SECONDS` before it says anything.

**It only fires once.**  Having said "the disk is nearly full", saying it again
thirty seconds later adds nothing and is actively worse -- the second one
carries no information and interrupts whatever you did about the first.  A
tripped watch stays quiet until the reading has recovered past a margin, which
is hysteresis rather than a cooldown: a value sitting exactly on the threshold
cannot chatter across it.

**It waits its turn.**  Announcements are held while a turn is running or
while Jarvis is already speaking.  Cutting across your own conversation to
mention the CPU is the behaviour of an alarm clock, not an assistant.
"""

from __future__ import annotations

import threading
import time

from . import sensors

# How long a condition has to hold before it is worth mentioning, how often
# the conditions are checked, and how far a reading has to come back before
# the same watch can fire again.
FOR_SECONDS = 8.0
PERIOD = 2.0
RECOVERY = 0.12                # of the threshold, as a fraction

# What can be watched, and how to read it. Each is (how to read it now, unit,
# whether *higher* is the worrying direction by default).
READINGS = {
    "cpu": (lambda: sensors.sampler().now("cpu"), "%", True),
    "memory": (lambda: sensors.sampler().now("memory"), "%", True),
    "battery": (lambda: sensors.battery().get("percent"), "%", False),
    "disk_free": (lambda: _free_gb(), "GB", False),
    "download": (lambda: sensors.sampler().now("net_down") / 1e6, "MB/s", True),
    "upload": (lambda: sensors.sampler().now("net_up") / 1e6, "MB/s", True),
    "latency": (lambda: _latency(), "ms", True),
}

ALIASES = {"ram": "memory", "net": "download", "network": "latency",
           "internet": "latency", "disk": "disk_free", "free": "disk_free",
           "cpu_percent": "cpu", "ping": "latency"}


def _free_gb() -> float | None:
    found = sensors.disks()
    return min(d["free"] for d in found) / 1e9 if found else None


def _latency() -> float | None:
    """Round trip to the internet, or a large number when there is none.

    Unreachable has to be a *number* rather than None, because the whole point
    of watching latency is to catch the connection going away, and a watch
    whose reading vanishes at the moment of interest can never fire.
    """
    online, took = sensors.reachable()
    return took if online else 9999.0


class WatchError(Exception):
    """A standing order that could not be set, said in a sentence."""


class Watch:
    """One condition, and what to say when it has held long enough."""

    def __init__(self, name: str, reading: str, above: bool, threshold: float,
                 message: str) -> None:
        self.name = name
        self.reading = reading
        self.above = above
        self.threshold = float(threshold)
        self.message = message
        self.created = time.time()
        self.fired = 0
        self.last_fired = 0.0
        self._since = None               # when the condition started holding
        self._armed = True               # False until the reading recovers

    def satisfied(self, value: float) -> bool:
        return value > self.threshold if self.above else value < self.threshold

    def recovered(self, value: float) -> bool:
        """Far enough back the other way to be worth arming again."""
        margin = abs(self.threshold) * RECOVERY
        return (value < self.threshold - margin if self.above
                else value > self.threshold + margin)

    def describe(self) -> str:
        unit = READINGS[self.reading][1]
        return (f"{self.name}: {self.reading} "
                f"{'above' if self.above else 'below'} "
                f"{self.threshold:g}{unit}"
                + (f", fired {self.fired}x" if self.fired else ", not yet fired"))


class Watcher:
    """Every standing order, and the thread that checks them."""

    def __init__(self, announce=None, period: float = PERIOD) -> None:
        self.watches: dict[str, Watch] = {}
        self.announce = announce or (lambda text: None)
        # Asked before speaking: the owner says whether now is a good moment.
        self.may_speak = lambda: True
        self.period = period
        self._pending: list[str] = []
        # Reentrant, because the error paths describe the current orders --
        # and `listing()` takes this same lock. With a plain Lock, asking to
        # cancel a watch that does not exist deadlocked the caller for good,
        # which on the MCP server means the tool call never returns at all.
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="jarvis-watch",
                                        daemon=True)
        self._thread.start()

    # -- setting them ------------------------------------------------------

    def add(self, reading: str, above: bool, threshold: float,
            message: str = "", name: str = "") -> str:
        key = ALIASES.get(str(reading).strip().lower(),
                          str(reading).strip().lower())
        if key not in READINGS:
            raise WatchError(f"I cannot watch '{reading}'. I can watch: "
                             f"{', '.join(sorted(READINGS))}.")
        if READINGS[key][0]() is None:
            raise WatchError(f"There is no {key} reading on this machine to watch.")
        try:
            level = float(threshold)
        except (TypeError, ValueError):
            raise WatchError("A threshold has to be a number.") from None
        unit = READINGS[key][1]
        label = (name or key).strip().lower()[:24]
        said = message.strip() or (
            f"The {key.replace('_', ' ')} has gone "
            f"{'above' if above else 'below'} {level:g}{unit}, sir.")
        with self._lock:
            self.watches[label] = Watch(label, key, above, level, said)
        return (f"Watching {key.replace('_', ' ')} for "
                f"{'above' if above else 'below'} {level:g}{unit}. "
                f"I will say so when it holds.")

    def remove(self, name: str) -> str:
        key = str(name).strip().lower()
        with self._lock:
            if key in ("all", "*", "everything"):
                count = len(self.watches)
                self.watches.clear()
                return (f"Cleared {count} standing order"
                        f"{'' if count == 1 else 's'}."
                        if count else "There was nothing being watched.")
            if key not in self.watches:
                raise WatchError(f"I am not watching anything called '{name}'. "
                                 f"{self.listing()}")
            del self.watches[key]
        return f"Stopped watching {key}."

    def listing(self) -> str:
        with self._lock:
            if not self.watches:
                return "I have no standing orders."
            rows = [f"  {w.describe()}" for w in self.watches.values()]
        return "Watching for:\n" + "\n".join(rows)

    # -- checking them -----------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self.period):
            try:
                self._check()
                self._flush()
            except Exception:                                      # noqa: BLE001
                pass         # a sensor that throws must not end the watch

    def _check(self) -> None:
        now = time.monotonic()
        with self._lock:
            watches = list(self.watches.values())
        for watch in watches:
            try:
                value = READINGS[watch.reading][0]()
            except Exception:                                      # noqa: BLE001
                continue
            if value is None:
                continue
            if not watch._armed:
                if watch.recovered(value):
                    watch._armed = True
                    watch._since = None
                continue
            if not watch.satisfied(value):
                watch._since = None
                continue
            if watch._since is None:
                watch._since = now
                continue
            if now - watch._since < FOR_SECONDS:
                continue
            unit = READINGS[watch.reading][1]
            watch.fired += 1
            watch.last_fired = time.time()
            watch._armed = False
            watch._since = None
            with self._lock:
                self._pending.append(f"{watch.message} It is at {value:.0f}{unit}.")

    def _flush(self) -> None:
        """Say anything waiting, if now is a reasonable moment to speak."""
        with self._lock:
            if not self._pending or not self.may_speak():
                return
            # One at a time: two conditions tripping together should not
            # produce a monologue.
            text = self._pending.pop(0)
        self.announce(text)

    def stop(self) -> None:
        self._stop.set()


_watcher = None
_watcher_lock = threading.Lock()


def watcher() -> Watcher:
    global _watcher
    with _watcher_lock:
        if _watcher is None:
            _watcher = Watcher()
        return _watcher


def bind(announce, may_speak=None) -> None:
    """Tell the watcher how to speak, and when it is allowed to."""
    live = watcher()
    live.announce = announce
    if may_speak is not None:
        live.may_speak = may_speak


def shutdown() -> None:
    global _watcher
    with _watcher_lock:
        if _watcher is not None:
            _watcher.stop()
            _watcher = None


# -- the tools -------------------------------------------------------------


def watch_for(reading: str, direction: str, threshold: float,
              message: str = "", name: str = "") -> str:
    """Tell me when something crosses a line."""
    way = str(direction).strip().lower()
    if way in ("above", "over", "more", "higher", "up", ">"):
        above = True
    elif way in ("below", "under", "less", "lower", "down", "<"):
        above = False
    else:
        raise WatchError("Say 'above' or 'below'.")
    return watcher().add(reading, above, threshold, message, name)


def list_watches() -> str:
    return watcher().listing()


def stop_watching(name: str) -> str:
    return watcher().remove(name)


WATCHES = {
    "watch_for": (
        watch_for,
        "Have Jarvis speak up by itself when a reading crosses a line, and "
        "keep watching until told otherwise. Readings: cpu, memory, battery, "
        "disk_free, download, upload, latency. The condition has to hold for "
        "several seconds before it says anything, and it will not repeat "
        "itself until the reading has recovered. Use this whenever the user "
        "says to tell them if something happens.",
        {"reading": ("string", "What to watch", True),
         "direction": ("string", "above or below", True),
         "threshold": ("number", "The value to cross", True),
         "message": ("string", "What to say when it does. Keep it to a "
                               "sentence -- it is spoken aloud", False),
         "name": ("string", "A short name, so it can be cancelled", False)}),
    "list_watches": (
        list_watches, "List the standing orders Jarvis is watching for.", {}),
    "stop_watching": (
        stop_watching, "Cancel a standing order, or 'all' of them.",
        {"name": ("string", "Its name, or 'all'", True)}),
}
