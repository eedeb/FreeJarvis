"""The widgets Jarvis puts on the screen, and the board that arranges them.

Two kinds of card, one renderer:

  * **Monitors** watch something and redraw themselves -- CPU, memory, the
    network, the battery, the disks, what is running. Jarvis opens one and
    then stops thinking about it; it keeps being true on its own. This is the
    difference between an assistant that can tell you the CPU is hot and one
    that is watching the CPU.
  * **Notes** are whatever Jarvis wants to leave on the screen: a countdown,
    a shopping list, the answer to something you will want again in ten
    minutes. It writes the contents and updates them when it has reason to.

**Why one thread for all of them.**  Each card is its own layered window,
because that is the only way to be see-through and still be over the wallpaper
rather than painted into a camera frame.  But a thread per card would mean six
threads waking sixty times a second to redraw six small bitmaps, and a monitor
whose producer blocks -- `netsh` takes 40ms -- would be six chances to stutter
instead of one.  So the board has one clock, ticks every card on it, and skips
any card whose picture has not changed.

**Placement.**  Down the left edge, because the right edge is the orb, the
session log and the audio widget.  Cards are placed when they are opened and
do not move afterwards: a board that re-flowed every time something opened
would move the thing you were reading, and the whole value of a card is that
it stays where you last saw it.  When a column fills, the next card starts
another column inward.
"""

from __future__ import annotations

import ctypes
import threading
import time

import cv2
import numpy as np

from . import holo, sensors
from .terminal import Glyphs, plain

WIDTH = 316
MARGIN = 14
GAP = 12
HEAD_H = 24
PAD_X, PAD_Y = 12, 7

# A card is at most this tall, and a note shows at most this many lines.
# Both exist so that one card cannot take the screen, which is the failure the
# session log had before it became a window with a size.
MAX_LINES = 14
GRAPH_H = 42
BAR_H = 7

# Pictures. A card is 316 wide, so a tall photograph would be most of the
# screen if it were allowed its own aspect ratio unbounded.
IMAGE_MAX_H = 400
IMAGE_KINDS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
# How often a file-backed card looks at the disk. The board ticks ten times a
# second and a stat() per tick per card is wasteful for something a person
# edits; half a second still feels immediate.
FILE_POLL = 0.5

# How often the board wakes, and how often a card bothers asking what it
# should say. These are different numbers on purpose: the sensors behind a
# monitor only move once a second, so producing a card ten times a second
# threw away nine tenths of the work -- and one of those producers was opening
# a socket. The board's own tick stays quicker than the slowest card so that a
# card added mid-run appears without a visible wait.
REDRAW_HZ = 5
PRODUCE_EVERY = 1.0      # monitors: the sampler's own cadence
NOTE_EVERY = 1.0         # notes only change when something writes them

_board = None
_board_lock = threading.Lock()
# Every card needs a window class of its own. Counted rather than hashed from
# the name: Python salts string hashes per run, and two cards landing on the
# same class name would mean the second window runs the first one's handler.
_serial = 0


class CardError(Exception):
    """Something about a widget could not be done, said in a sentence."""


# -- what a card knows how to draw -----------------------------------------


def _spec(title, lines=None, gauge=None, series=None, ceiling=None,
          note="", tint=None, image=None) -> dict:
    """One card's contents. Every producer returns this shape."""
    return {"title": title, "lines": list(lines or []), "gauge": gauge,
            "series": list(series or []), "ceiling": ceiling, "note": note,
            "tint": tint, "image": image}


class Card:
    """One see-through panel, its own window, redrawn by the board."""

    def __init__(self, name: str, kind: str, x: int, y: int, height: int,
                 produce, period: float = PRODUCE_EVERY) -> None:
        from .widget import HoloWindow

        self.name = name
        self.kind = kind                      # "monitor", "note" or "file"
        self.produce = produce                # () -> spec
        self.period = float(period)
        self._asked = 0.0
        self.opened_at = time.time()
        self.glyphs = _glyphs()
        self.width, self.height = WIDTH, height
        self.colour = np.empty((height, WIDTH, 3), np.uint8)
        self.alpha = np.empty((height, WIDTH), np.uint8)
        self._cover = np.empty((height, WIDTH), np.uint8)
        self._ink = np.empty((height, WIDTH, 3), np.uint8)
        self._wide = np.empty((height, WIDTH, 3), np.uint8)
        self._lit = np.empty((height, WIDTH, 3), np.uint8)
        self._base = holo.chrome(WIDTH, height, HEAD_H)
        self._drawn = None
        global _serial
        _serial += 1
        self.x, self.y = x, y
        self.window = HoloWindow(x, y, WIDTH, height,
                                 name=f"GestureControlCard{_serial}")
        self.available = self.window.available
        self.unavailable_reason = self.window.unavailable_reason

    # -- drawing -----------------------------------------------------------

    def tick(self, force: bool = False) -> None:
        """Ask the producer what to say, and put it on the screen if it moved.

        `force` for the calls that follow something changing -- opening a card,
        or writing to one -- where waiting out the period would show a card
        that is briefly blank or stale at the one moment someone is looking.
        """
        now = time.monotonic()
        if not force and now - self._asked < self.period:
            return
        self._asked = now
        spec = self.produce()
        key = (spec["title"], tuple(spec["lines"]), spec["gauge"],
               spec["note"], id(spec.get("image")),
               tuple(round(v, 2) for v in spec["series"][-64:]))
        if key == self._drawn:
            return
        self._drawn = key
        self._render(spec)
        if self.window.bgra is not None:
            from .widget import premultiply

            premultiply(self.colour, self.alpha, self.window.bgra)
            self.window.blit()

    def _render(self, spec: dict) -> None:
        base_colour, base_alpha = self._base
        np.copyto(self.colour, base_colour)
        np.copyto(self.alpha, base_alpha)
        cover, ink = self._cover, self._ink
        cover[:] = 0
        ink[:] = holo.FAINT
        glyph = self.glyphs
        tint = spec.get("tint") or holo.WARM

        # Title, and the kind of card it is on the right of the strip.
        top = (HEAD_H - glyph.cell_h) // 2 + 1
        end = glyph.stamp(cover, spec["title"][:26].upper(), PAD_X, top)
        ink[top:top + glyph.cell_h, PAD_X:end] = holo.HOT
        if spec["note"]:
            mark = spec["note"][:12]
            mx = self.width - PAD_X - len(mark) * glyph.cell_w
            if mx > end + glyph.cell_w:
                stop = glyph.stamp(cover, mark, mx, top)
                ink[top:top + glyph.cell_h, mx:stop] = tint

        y = HEAD_H + PAD_Y
        if spec["gauge"] is not None:
            holo.bar(self.colour, self.alpha,
                     (PAD_X, y, self.width - PAD_X, y + BAR_H),
                     float(spec["gauge"]), tint)
            y += BAR_H + PAD_Y

        picture = spec.get("image")
        if picture is not None:
            y = self._draw_image(picture, y)

        cols = (self.width - 2 * PAD_X) // glyph.cell_w
        room = self.height - PAD_Y - y
        if spec["series"]:
            room -= GRAPH_H + PAD_Y
        for line in spec["lines"][:MAX_LINES]:
            if room < glyph.cell_h:
                break
            text = str(line)
            # A label and a value separated by a tab are laid out as two
            # columns: the values line up down the card, which is what makes
            # a stack of readings scannable instead of a paragraph.
            if "\t" in text:
                label, _, value = text.partition("\t")
                glyph.stamp(cover, label[:cols], PAD_X, y)
                ink[y:y + glyph.cell_h, PAD_X:self.width - PAD_X] = holo.FAINT
                vx = self.width - PAD_X - len(value) * glyph.cell_w
                if vx > PAD_X + len(label) * glyph.cell_w:
                    stop = glyph.stamp(cover, value, vx, y)
                    ink[y:y + glyph.cell_h, vx:stop] = holo.COOL
            else:
                glyph.stamp(cover, text[:cols], PAD_X, y)
                ink[y:y + glyph.cell_h, PAD_X:self.width - PAD_X] = holo.COOL
            y += glyph.cell_h
            room -= glyph.cell_h
        if len(spec["lines"]) > MAX_LINES:
            more = f"+{len(spec['lines']) - MAX_LINES} more"
            glyph.stamp(cover, more, PAD_X, y)
            ink[y:y + glyph.cell_h, PAD_X:self.width - PAD_X] = holo.FAINT

        if spec["series"]:
            gy = self.height - PAD_Y - GRAPH_H
            holo.sparkline(self.colour, self.alpha,
                           (PAD_X, gy, self.width - PAD_X, gy + GRAPH_H),
                           spec["series"], spec.get("ceiling"), tint)

        cv2.merge((cover, cover, cover), dst=self._wide)
        cv2.multiply(self._wide, ink, dst=self._lit, scale=1.0 / 255.0)
        cv2.add(self.colour, self._lit, dst=self.colour)
        cv2.max(self.alpha, cover, dst=self.alpha)

    def _draw_image(self, picture: np.ndarray, y: int) -> int:
        """Fit a picture across the card. Returns the y below it."""
        room_w = self.width - 2 * PAD_X
        room_h = self.height - PAD_Y - y
        if room_w < 8 or room_h < 8:
            return y
        high, wide = picture.shape[:2]
        scale = min(room_w / max(wide, 1), room_h / max(high, 1))
        out_w, out_h = max(1, int(wide * scale)), max(1, int(high * scale))
        shrink = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        fitted = cv2.resize(picture, (out_w, out_h), interpolation=shrink)
        x = PAD_X + (room_w - out_w) // 2
        self.colour[y:y + out_h, x:x + out_w] = fitted
        # Opaque: a photograph you can see your wallpaper through is not a
        # photograph. The glass stays see-through around it.
        self.alpha[y:y + out_h, x:x + out_w] = 255
        cv2.rectangle(self.colour, (x - 1, y - 1), (x + out_w, y + out_h),
                      tuple(int(c * 0.6) for c in holo.WARM), 1)
        cv2.rectangle(self.alpha, (x - 1, y - 1), (x + out_w, y + out_h), 255, 1)
        return y + out_h + PAD_Y

    def close(self) -> None:
        self.window.close()
        self.available = False


_glyph_cache = None


def _glyphs() -> Glyphs:
    """One atlas for every card -- building it is the expensive part."""
    global _glyph_cache
    if _glyph_cache is None:
        _glyph_cache = Glyphs()
    return _glyph_cache


# -- the built-in monitors -------------------------------------------------


def _cpu_card() -> dict:
    tap = sensors.sampler()
    reading = sensors.cpu()
    percent = reading.get("percent", 0.0)
    lines = [f"now\t{percent:.0f}%"]
    if reading.get("average_1m") is not None:
        lines.append(f"1 min average\t{reading['average_1m']:.0f}%")
        lines.append(f"1 min peak\t{reading['peak_1m']:.0f}%")
    if reading.get("mhz"):
        lines.append(f"clock\t{reading['mhz']:.0f} MHz")
    lines.append(f"cores\t{reading.get('cores')} / {reading.get('threads')}t")
    busiest = [p for p in sensors.processes(5)
               if p["cpu"] > 1.0 and p["name"].lower() not in IDLE][:3]
    for proc in busiest:
        lines.append(f"{proc['name'][:18]}\t{proc['cpu']:.0f}%")
    return _spec("CPU", lines, gauge=percent / 100.0,
                 series=tap.history("cpu"), ceiling=100.0,
                 note=f"{percent:.0f}%",
                 tint=holo.level_colour(percent / 100.0))


def _memory_card() -> dict:
    tap = sensors.sampler()
    ram = sensors.memory()
    if not ram:
        return _spec("MEMORY", ["psutil is not installed."])
    lines = [f"used\t{sensors.human_bytes(ram['used'])}",
             f"free\t{sensors.human_bytes(ram['available'])}",
             f"total\t{sensors.human_bytes(ram['total'])}"]
    if ram.get("swap_total"):
        lines.append(f"swap\t{ram['swap_percent']:.0f}%")
    for proc in sensors.processes(3, "memory"):
        lines.append(f"{proc['name'][:18]}\t{sensors.human_bytes(proc['memory'])}")
    return _spec("MEMORY", lines, gauge=ram["percent"] / 100.0,
                 series=tap.history("memory"), ceiling=100.0,
                 note=f"{ram['percent']:.0f}%",
                 tint=holo.level_colour(ram["percent"] / 100.0))


def _network_card() -> dict:
    tap = sensors.sampler()
    link = sensors.wifi()
    down, up = tap.now("net_down"), tap.now("net_up")
    lines = [f"down\t{sensors.human_rate(down)}",
             f"up\t{sensors.human_rate(up)}"]
    if link.get("connected"):
        lines.append(f"network\t{link['ssid'][:16]}")
        if link.get("signal_percent") is not None:
            lines.append(f"signal\t{link['signal_percent']}%")
        if link.get("rx_mbps"):
            lines.append(f"link\t{link['rx_mbps']:.0f} Mbps")
        lines.append(f"band\t{link.get('band', '?')}")
    elif link:
        lines.append(f"wi-fi\t{link.get('detail', 'off')}")
    # The cached one. This card used to open a socket to a DNS resolver every
    # time it redrew, which was ten times a second for a number that cannot
    # meaningfully change that fast.
    reached, latency = sensors.online()
    lines.append(f"internet\t{f'{latency:.0f} ms' if reached else 'DOWN'}")
    return _spec("NETWORK", lines, series=tap.history("net_down"),
                 note=sensors.human_rate(down),
                 tint=holo.WARM if reached else holo.ALERT)


def _battery_card() -> dict:
    tap = sensors.sampler()
    power = sensors.battery()
    if not power.get("present"):
        return _spec("BATTERY", ["No battery -- this machine runs",
                                 "on mains power."], note="mains")
    lines = [f"charge\t{power['percent']:.0f}%",
             f"state\t{'charging' if power['plugged'] else 'on battery'}"]
    if power.get("seconds_left"):
        lines.append(f"remaining\t{sensors.human_span(power['seconds_left'])}")
    if power.get("percent_per_hour"):
        drift = power["percent_per_hour"]
        lines.append(f"rate\t{drift:+.1f}%/h")
    fraction = power["percent"] / 100.0
    return _spec("BATTERY", lines, gauge=fraction,
                 series=tap.history("battery"), ceiling=100.0,
                 note=f"{power['percent']:.0f}%",
                 tint=holo.GOOD if power["plugged"]
                 else holo.level_colour(fraction, inverted=True))


def _disk_card() -> dict:
    found = sensors.disks()
    if not found:
        return _spec("DISKS", ["No drives could be read."])
    lines = []
    worst = 0.0
    for disk in found[:5]:
        lines.append(f"{disk['device']}\t{sensors.human_bytes(disk['free'])} free")
        lines.append(f"  of {sensors.human_bytes(disk['total'])}"
                     f"\t{disk['percent']:.0f}% used")
        worst = max(worst, disk["percent"])
    tap = sensors.sampler()
    lines.append(f"reading\t{sensors.human_rate(tap.now('disk_read'))}")
    lines.append(f"writing\t{sensors.human_rate(tap.now('disk_write'))}")
    return _spec("DISKS", lines, gauge=worst / 100.0,
                 series=tap.history("disk_write"),
                 note=f"{worst:.0f}%",
                 tint=holo.level_colour(worst / 100.0))


def _process_card() -> dict:
    rows = [p for p in sensors.processes(9)
            if p["name"].lower() not in IDLE][:7]
    lines = [f"{p['name'][:17]}\t{p['cpu']:.0f}% {sensors.human_bytes(p['memory'])}"
             for p in rows]
    busiest = rows[0]["cpu"] if rows else 0.0
    return _spec("RUNNING", lines or ["Nothing to report."],
                 note=f"top {busiest:.0f}%")


# Windows accounts for doing nothing as a process, and it is the busiest one
# on the machine whenever the machine is idle.
IDLE = {"system idle process", "idle"}

MONITORS = {
    "cpu": _cpu_card,
    "memory": _memory_card,
    "network": _network_card,
    "battery": _battery_card,
    "disk": _disk_card,
    "processes": _process_card,
}

# How tall each monitor's window is. Fixed per kind rather than measured,
# because a window cannot be resized without rebuilding its bitmap, and a card
# that grew and shrank as the numbers changed would be worse than one that is
# occasionally half empty.
MONITOR_HEIGHT = {
    "cpu": 250, "memory": 250, "network": 236, "battery": 200,
    "disk": 250, "processes": 178,
}


# -- the board -------------------------------------------------------------


class Board:
    """Every card on the screen, and the one clock that redraws them."""

    def __init__(self, screen: tuple[int, int] | None = None) -> None:
        if screen is None:
            user32 = ctypes.WinDLL("user32")
            user32.SetProcessDPIAware()
            screen = (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
        self.screen = screen
        self.cards: dict[str, Card] = {}
        self._notes: dict[str, dict] = {}
        self._files: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="jarvis-cards",
                                        daemon=True)
        self._thread.start()

    # -- placement ---------------------------------------------------------

    def _free_slot(self, height: int) -> tuple[int, int]:
        """Somewhere this card fits that nothing is already using."""
        with self._lock:
            taken = [(c.x, c.y, c.height) for c in self.cards.values()]
        column, bottom = MARGIN, self.screen[1] - MARGIN
        while column + WIDTH < self.screen[0] // 2:
            y = MARGIN
            for cx, cy, ch in sorted([t for t in taken if t[0] == column],
                                     key=lambda t: t[1]):
                if y + height <= cy:
                    break                       # a gap something else left
                y = max(y, cy + ch + GAP)
            if y + height <= bottom:
                return column, y
            column += WIDTH + GAP
        # Every column is full. Rather than refuse, stack on the first one --
        # a card that is there and overlapping beats a card that silently
        # did not open.
        return MARGIN, MARGIN

    # -- opening and closing -----------------------------------------------

    def open_monitor(self, kind: str) -> str:
        kind = str(kind).strip().lower()
        if kind not in MONITORS:
            raise CardError(f"There is no monitor called '{kind}'. "
                            f"I have: {', '.join(sorted(MONITORS))}.")
        with self._lock:
            if kind in self.cards:
                return f"The {kind} monitor is already on the screen."
            height = MONITOR_HEIGHT.get(kind, 220)
            if kind == "battery" and not sensors.battery().get("present"):
                height = 112        # two lines saying there is nothing to watch
            x, y = self._free_slot(height)
            card = Card(kind, "monitor", x, y, height, MONITORS[kind])
            if not card.available:
                raise CardError(card.unavailable_reason
                                or f"The {kind} monitor could not open.")
            self.cards[kind] = card
        card.tick(force=True)
        return f"The {kind} monitor is on the screen."

    def open_note(self, title: str, body: str, gauge=None) -> str:
        title = " ".join(str(title).split())[:26] or "NOTE"
        key = title.lower()
        lines = _body_lines(body)
        with self._lock:
            self._notes[key] = {"title": title, "lines": lines, "gauge": gauge}
            if key in self.cards:
                self.cards[key].tick(force=True)
                return f"Updated the '{title}' card."
            height = min(HEAD_H + 2 * PAD_Y + (len(lines) + 1) * _glyphs().cell_h
                         + (BAR_H + PAD_Y if gauge is not None else 0),
                         HEAD_H + 2 * PAD_Y + (MAX_LINES + 1) * _glyphs().cell_h)
            height = max(height, 96)
            x, y = self._free_slot(height)
            card = Card(key, "note", x, y, height,
                        lambda k=key: self._note_spec(k))
            if not card.available:
                self._notes.pop(key, None)
                raise CardError(card.unavailable_reason
                                or f"The '{title}' card could not open.")
            self.cards[key] = card
        card.tick(force=True)
        return f"'{title}' is on the screen."

    def open_file(self, title: str, path: str) -> str:
        """Show a file on the screen, and keep showing it as it changes.

        The point of following the file rather than snapshotting it is that it
        gives Jarvis a widget it can *update*: write the file, and what is on
        the screen changes. A card it has to re-issue a tool call to refresh
        is one that is silently stale the moment anything else happens.
        """
        title = " ".join(str(title).split())[:26] or "FILE"
        key = title.lower()
        lines, picture, note = read_file_card(path)      # raises if unreadable

        from .tools import _resolve
        real = _resolve(str(path))
        with self._lock:
            self._files[key] = {"title": title, "path": real, "stamp": None,
                                "lines": lines, "image": picture, "note": note,
                                "checked": 0.0}
            if key in self.cards:
                self.cards[key].tick(force=True)
                return f"'{title}' is on the screen, showing {real.name}."
            if picture is not None:
                high, wide = picture.shape[:2]
                room = WIDTH - 2 * PAD_X
                height = HEAD_H + 2 * PAD_Y + min(
                    int(high * room / max(wide, 1)), IMAGE_MAX_H)
            else:
                height = HEAD_H + 2 * PAD_Y + (
                    min(len(lines), MAX_LINES) + 1) * _glyphs().cell_h
            height = max(int(height), 96)
            x, y = self._free_slot(height)
            card = Card(key, "file", x, y, height,
                        lambda k=key: self._file_spec(k), period=FILE_POLL)
            if not card.available:
                self._files.pop(key, None)
                raise CardError(card.unavailable_reason
                                or f"The '{title}' card could not open.")
            self.cards[key] = card
        card.tick(force=True)
        return f"'{title}' is on the screen, showing {real.name}."

    def _file_spec(self, key: str) -> dict:
        """What a file card says now, re-reading only when the file moved."""
        with self._lock:
            held = self._files.get(key)
        if held is None:
            return _spec("GONE", [])
        now = time.monotonic()
        if now - held["checked"] >= FILE_POLL:
            held["checked"] = now
            try:
                stamp = held["path"].stat().st_mtime_ns
            except OSError:
                return _spec(held["title"], ["The file is no longer there."],
                             note="missing", tint=holo.ALERT)
            if stamp != held["stamp"]:
                held["stamp"] = stamp
                try:
                    (held["lines"], held["image"],
                     held["note"]) = read_file_card(held["path"])
                except CardError as exc:
                    held["lines"], held["image"] = [str(exc)], None
        return _spec(held["title"], held["lines"], note=held["note"],
                     image=held["image"])

    def _note_spec(self, key: str) -> dict:
        with self._lock:
            held = self._notes.get(key)
        if held is None:
            return _spec("GONE", [])
        return _spec(held["title"], held["lines"], gauge=held.get("gauge"))

    def close(self, name: str) -> str:
        key = str(name).strip().lower()
        with self._lock:
            if key in ("all", "*", "everything"):
                names = list(self.cards)
                for one in names:
                    self.cards.pop(one).close()
                self._notes.clear()
                self._files.clear()
                return (f"Cleared {len(names)} card{'' if len(names) == 1 else 's'}."
                        if names else "There was nothing on the screen.")
            card = self.cards.pop(key, None)
            self._notes.pop(key, None)
            self._files.pop(key, None)
        if card is None:
            raise CardError(f"There is no card called '{name}'. "
                            f"{self.listing()}")
        card.close()
        return f"Closed the '{name}' card."

    def listing(self) -> str:
        with self._lock:
            if not self.cards:
                return "There is nothing on the screen right now."
            rows = [f"  {c.name} ({c.kind}, at {c.x},{c.y}, open for "
                    f"{sensors.human_span(time.time() - c.opened_at)})"
                    for c in self.cards.values()]
        return "On the screen now:\n" + "\n".join(rows)

    # -- the clock ---------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(1.0 / REDRAW_HZ):
            with self._lock:
                cards = list(self.cards.values())
            for card in cards:
                try:
                    card.tick()
                except Exception:                                  # noqa: BLE001
                    pass       # one broken producer must not stop the others

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            for card in self.cards.values():
                card.close()
            self.cards.clear()


def read_file_card(path) -> tuple[list[str], "np.ndarray | None", str]:
    """A file as something a card can show: (lines, image, note).

    Resolved through the same guard the file tools use, so a card cannot be
    pointed at anything the rest of Jarvis is not allowed to read. That
    matters more here than it looks: this is the one tool whose whole purpose
    is to put a file's contents on the screen.
    """
    from .tools import ToolError, _resolve

    try:
        real = _resolve(str(path))
    except ToolError as exc:
        raise CardError(str(exc)) from None
    if not real.exists():
        raise CardError(f"There is no file at {real}.")
    if real.is_dir():
        raise CardError(f"{real} is a folder, not a file.")

    if real.suffix.lower() in IMAGE_KINDS:
        picture = cv2.imread(str(real), cv2.IMREAD_COLOR)
        if picture is None:
            raise CardError(f"{real.name} is not a picture I can read.")
        return [], picture, f"{picture.shape[1]}x{picture.shape[0]}"

    try:
        text = real.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise CardError(f"{real.name} could not be read ({exc}).") from None
    lines = []
    for raw in text.splitlines():
        line = plain(raw)
        if not line:
            continue
        # Markdown, lightly: the heading marks are noise on a card that
        # already has a title bar, and nothing here can render bold anyway.
        line = line.lstrip("#").strip() if line.startswith("#") else line
        lines.append(line)
    return (lines or ["(empty)"]), None, real.name[:12]


def _body_lines(body) -> list[str]:
    """A note's text as lines, however the model handed it over."""
    if isinstance(body, (list, tuple)):
        raw = [str(x) for x in body]
    else:
        raw = str(body).replace("\\n", "\n").splitlines()
    return [line.rstrip() for line in raw if line.strip()] or ["(empty)"]


def board() -> Board:
    """The one board, made on first use."""
    global _board
    with _board_lock:
        if _board is None:
            _board = Board()
        return _board


def shutdown() -> None:
    global _board
    with _board_lock:
        if _board is not None:
            _board.shutdown()
            _board = None
