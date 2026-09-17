"""The session log: a small terminal under the orb.

What it replaces, and why.  The conversation used to be painted straight onto
the overlay as a column of slabs growing downward from the reactor, which had
two problems.  It grew until it ran out of screen, so a turn that called six
tools covered everything; and it was painted by the camera's draw loop, so it
only advanced when the webcam produced a frame -- a reply streaming in while
the lens was covered simply stopped.

So this is its own window, with its own clock and a fixed size.  Output older
than the window is scrollback rather than gone: it follows the bottom while you
leave it alone, stops following the moment you scroll up, and starts again when
you come back down.  That is the behaviour of every terminal ever written, and
it is the right one -- new output must never yank the screen away from
something being read.

**Why a real character grid.**  The look being aimed at is a machine's log:
`open_app(name=chrome)`, `-> Chrome is open`.  That reads as code only if the
columns line up, and OpenCV's built-in fonts are all proportional, so an `i`
and an `M` take different widths and nothing ever aligns.  Consolas is on every
Windows machine, so every printable character is rendered once into a fixed
cell at startup and the whole body of the terminal is then drawn as one numpy
gather -- `atlas[indices]` -- rather than a putText per line.  A grid of cells
is also what makes wrapping arithmetic instead of measurement: with a fixed
advance, where a line breaks is a division, so the entire log can be re-wrapped
every frame for nothing.

If Consolas or Pillow is missing the atlas is built with OpenCV's Hershey font
instead, one glyph per cell.  The glyphs are uglier; the grid, the wrapping and
the scrollback are identical, because none of them depend on the shapes.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


# The cell, in pixels. Consolas at 16px happens to advance exactly 9px with a
# 12/5 ascent/descent, so the grid lands on whole pixels without rounding --
# which is what keeps the columns crisp instead of shimmering by a fraction.
FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_SIZE = 16
CELL_W, CELL_H = 9, 18
BASELINE = 13                    # where the glyph sits inside its cell

# Index 95 in the atlas, past the last printable character: a filled block,
# used as the cursor. Free, because the atlas is a lookup either way.
BLOCK = 95

# The same palette as the orb and the audio widget, so the whole interface
# reads as one machine: projected cyan light, hot and near-white at its
# brightest. BGR, because everything downstream is OpenCV -- these read
# blue-dominant for that reason, not by mistake.
WARM = (246, 170, 44)
HOT = (255, 240, 198)
FAINT = (158, 124, 92)
GLASS = (20, 14, 9)              # the panel's own tint, before its alpha

# How solid it is. The glass is see-through on purpose -- this is meant to look
# projected onto the desk -- but text and the frame are not: a log you have to
# squint past your wallpaper to read is not a log. 182 is where a pale
# wallpaper stops showing through far enough to swallow the dim lines; much
# below it the reasoning and the header go illegible over bright ground, and
# much above it stops looking projected and starts looking like a box.
GLASS_ALPHA, EDGE_ALPHA = 182, 240

# Per kind: the ink, and the two-character gutter that opens its lines. The
# gutter is what makes the column scannable without reading it -- you can see
# that three tools ran without parsing a word.
STYLE = {
    "you":       ((236, 226, 212), "> "),
    "note":      ((186, 154, 120), ":: "),
    "reasoning": ((168, 138, 108), ".. "),
    "tool":      ((255, 206, 122), "$ "),
    "jarvis":    ((255, 228, 168), "< "),
    "error":     ((120, 120, 255), "! "),
}

PAD_X, PAD_Y = 12, 8
HEAD_H = 24                      # the title strip
RAIL_W = 4                       # the scroll position rail
RAIL_GAP = 10

# How long a single entry may get before its middle is dropped. A tool that
# returns a directory listing is worth seeing; it is not worth a thousand
# lines of scrollback.
MAX_CHARS = 1200

# How far one wheel notch moves, and how much of the view a Page moves.
WHEEL_LINES = 3

REDRAW_HZ = 30

# Characters the log picks up from the model and the app that ASCII has a
# perfectly good spelling for. Replaced rather than rendered, because the
# atlas is 95 printable characters and a terminal that shows `?` where the
# reply had a quotation mark looks broken.
_PLAIN = str.maketrans({
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2013": "-", "\u2014": "--", "\u2026": "...", "\u2192": "->",
    "\u00a0": " ", "\u2022": "*",
})


def plain(text) -> str:
    """One line of ASCII, however the model spelled it."""
    return " ".join(str(text).translate(_PLAIN).split())


def shorten(text, limit: int = MAX_CHARS) -> str:
    """One entry's worth of any output, with the middle taken out.

    The middle rather than the end: the interesting parts of a long result are
    usually what it was and how it finished, and a plain truncation keeps only
    the first of those.
    """
    text = plain(text)
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]} ... {text[-(limit - head - 5):]}"


class Glyphs:
    """Every printable character, rendered once into a fixed cell.

    `atlas` is (96, CELL_H, CELL_W) of coverage 0..255: index `c - 32` for a
    printable character, and `BLOCK` for the cursor. Looking a string up is
    then subtraction, and looking a whole screen up is one fancy index.
    """

    def __init__(self, path: str = FONT_PATH, size: int = FONT_SIZE) -> None:
        self.cell_w, self.cell_h = CELL_W, CELL_H
        self.source = "consolas"
        try:
            self.atlas = self._truetype(path, size)
        except Exception:                                        # noqa: BLE001
            self.source = "hershey"
            self.atlas = self._hershey()
        self.atlas[BLOCK] = 255

    # -- building ----------------------------------------------------------

    @staticmethod
    def _truetype(path: str, size: int) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(path, size)
        advance = int(round(font.getlength("M")))
        if advance != CELL_W:
            # Not the font this grid was measured for. Rather than silently
            # drawing on top of the next column, fall back to the shapes we
            # can place ourselves.
            raise ValueError(f"{path} advances {advance}px, not {CELL_W}")
        sheet = Image.new("L", (CELL_W * 96, CELL_H), 0)
        draw = ImageDraw.Draw(sheet)
        for i in range(95):
            draw.text((i * CELL_W, BASELINE), chr(32 + i), font=font,
                      fill=255, anchor="ls")
        flat = np.asarray(sheet, np.uint8)
        return np.ascontiguousarray(
            flat.reshape(CELL_H, 96, CELL_W).transpose(1, 0, 2))

    @staticmethod
    def _hershey() -> np.ndarray:
        """The same grid without Pillow: one Hershey glyph centred per cell."""
        atlas = np.zeros((96, CELL_H, CELL_W), np.uint8)
        for i in range(95):
            ch = chr(32 + i)
            (w, _), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_PLAIN, 0.9, 1)
            cell = np.zeros((CELL_H, CELL_W), np.uint8)
            cv2.putText(cell, ch, (max(0, (CELL_W - w) // 2), BASELINE),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, 255, 1, cv2.LINE_AA)
            atlas[i] = cell
        return atlas

    # -- using -------------------------------------------------------------

    def index(self, text: str) -> np.ndarray:
        """`text` as atlas rows. Anything unprintable becomes a space."""
        raw = np.frombuffer(text.encode("ascii", "replace"), np.uint8)
        idx = raw.astype(np.int16) - 32
        # Control characters and DEL land outside the atlas; a space is a
        # better answer than an exception or a wrapped-around glyph.
        np.putmask(idx, (idx < 0) | (idx > 94), 0)
        return idx

    def stamp(self, cover: np.ndarray, text: str, x: int, y: int) -> int:
        """Draw one short run into `cover` at a pixel position. Returns its end."""
        idx = self.index(text)
        wide = min(len(idx), (cover.shape[1] - x) // self.cell_w)
        if wide <= 0:
            return x
        block = self.atlas[idx[:wide]]                      # (n, ch, cw)
        strip = block.transpose(1, 0, 2).reshape(self.cell_h, wide * self.cell_w)
        patch = cover[y:y + self.cell_h, x:x + wide * self.cell_w]
        np.maximum(patch, strip[:patch.shape[0], :patch.shape[1]], out=patch)
        return x + wide * self.cell_w



class Console:
    """The buffer, the scroll position, and the picture of both.

    Owns no window, and knows nothing about Win32: `colour` and `alpha` are
    two plain arrays, and `Terminal` is what puts them on the screen. Keeping
    the split means the whole look of this thing can be built and checked
    without a window ever existing, which is how the tests read it.
    """

    def __init__(self, width: int, height: int,
                 glyphs: Glyphs | None = None) -> None:
        self.width, self.height = int(width), int(height)
        self.glyphs = glyphs or Glyphs()
        cw, ch = self.glyphs.cell_w, self.glyphs.cell_h

        self._body_x = PAD_X
        self._body_y = HEAD_H + PAD_Y
        body_w = self.width - 2 * PAD_X - RAIL_W - RAIL_GAP
        body_h = self.height - self._body_y - PAD_Y
        self.cols = max(8, body_w // cw)
        self.rows = max(2, body_h // ch)

        # Lines from the bottom. 0 means pinned to the newest output, which is
        # the only state in which new output is allowed to move the view.
        self.scroll = 0
        self._entries: list[tuple[str, str]] = []
        self._wrapped: list[tuple[tuple[int, int, int], str]] = []
        self._stamp = None                       # what _wrapped was built from

        self.colour = np.empty((self.height, self.width, 3), np.uint8)
        self.alpha = np.empty((self.height, self.width), np.uint8)
        self._cover = np.empty((self.height, self.width), np.uint8)
        self._ink = np.empty((self.height, self.width, 3), np.uint8)
        self._wide = np.empty((self.height, self.width, 3), np.uint8)
        self._lit = np.empty((self.height, self.width, 3), np.uint8)
        # What render() last drew, so it can decline to draw it again.
        self._drawn = None
        self._build()

    # -- the parts that never change --------------------------------------

    def _build(self) -> None:
        """The glass, the frame and the scanlines, drawn once."""
        base = np.empty((self.height, self.width, 3), np.uint8)
        base[:] = GLASS
        # Scanlines: every third row a shade darker. Static, so it costs one
        # multiply at startup rather than a pass over the panel every frame.
        base[::3] = (base[::3].astype(np.uint16) * 7 // 10).astype(np.uint8)

        alpha = np.full((self.height, self.width), GLASS_ALPHA, np.uint8)
        edge = tuple(int(c * 0.82) for c in WARM)
        right, bottom = self.width - 1, self.height - 1
        cv2.rectangle(base, (0, 0), (right, bottom), edge, 1)
        cv2.rectangle(alpha, (0, 0), (right, bottom), EDGE_ALPHA, 1)
        # A rule under the title, so the header reads as a header.
        rule = tuple(int(c * 0.5) for c in WARM)
        cv2.line(base, (1, HEAD_H), (right - 1, HEAD_H), rule, 1)
        cv2.line(alpha, (1, HEAD_H), (right - 1, HEAD_H), EDGE_ALPHA, 1)

        # Corner brackets, the same detail the reactor is framed with.
        arm = 14
        for cx, cy, dx, dy in ((0, 0, 1, 1), (right, 0, -1, 1),
                               (0, bottom, 1, -1), (right, bottom, -1, -1)):
            for canvas, value in ((base, HOT), (alpha, 255)):
                cv2.line(canvas, (cx, cy), (cx + dx * arm, cy), value, 2)
                cv2.line(canvas, (cx, cy), (cx, cy + dy * arm), value, 2)

        self._base_colour, self._base_alpha = base, alpha
        self._rail_x = self.width - PAD_X - RAIL_W

    # -- what is in it -----------------------------------------------------

    def feed(self, entries) -> None:
        """Take the transcript as it stands. Re-wrapped only when it changed."""
        entries = [(k, t) for k, t in entries]
        stamp = (len(entries), sum(len(t) for _, t in entries))
        if stamp == self._stamp:
            return
        self._stamp = stamp
        self._entries = entries
        below = self._below()
        self._rewrap()
        # Scrolled up, and more output arrived: hold the same *text* still
        # rather than the same offset, or every token that streams in would
        # walk the view down a line.
        if self.scroll:
            self.scroll = min(max(len(self._wrapped) - below, 0),
                              max(len(self._wrapped) - self.rows, 0))

    def _below(self) -> int:
        """How many lines are currently below the top of the view."""
        return len(self._wrapped) - self.scroll

    def _rewrap(self) -> None:
        """Every entry as fixed-width lines, with its gutter and its ink."""
        out: list[tuple[tuple[int, int, int], str]] = []
        for kind, text in self._entries:
            ink, gutter = STYLE.get(kind, STYLE["jarvis"])
            body = shorten(text)
            room = max(4, self.cols - len(gutter))
            # A hanging indent under the gutter, so a wrapped line is visibly
            # a continuation and the gutter column stays readable down the
            # left edge.
            lead = " " * len(gutter)
            first = True
            for start in range(0, max(len(body), 1), room):
                chunk = body[start:start + room]
                out.append((ink, (gutter if first else lead) + chunk))
                first = False
        self._wrapped = out

    def scroll_by(self, lines: int) -> None:
        """Move the view. Positive is backward, into older output."""
        limit = max(0, len(self._wrapped) - self.rows)
        self.scroll = int(min(max(self.scroll + lines, 0), limit))

    def to_bottom(self) -> None:
        self.scroll = 0

    @property
    def live(self) -> bool:
        """Whether new output will move the view."""
        return self.scroll == 0

    # -- the picture -------------------------------------------------------

    def render(self, state: str = "idle", cursor: bool = False) -> bool:
        """Fill `colour` and `alpha` with the terminal as it stands now.

        Returns whether anything actually changed. Nothing here animates on
        its own, so a log nobody is adding to and nobody is scrolling redraws
        to the same picture thirty times a second -- and at panel size that is
        a few percent of a core spent proving a bitmap is what it already was.
        """
        key = (self._stamp, self.scroll, state, cursor)
        if key == self._drawn:
            return False
        self._drawn = key

        np.copyto(self.colour, self._base_colour)
        np.copyto(self.alpha, self._base_alpha)
        cover, ink = self._cover, self._ink
        cover[:] = 0
        ink[:] = FAINT

        self._header(cover, ink, state)
        self._body(cover, ink, cursor)
        self._rail(self.colour, self.alpha)

        # Ink where there is coverage, glass where there is not. Additive, so
        # the text sits *in* the glass rather than punching a hole in it.
        # Done with OpenCV rather than the obvious numpy expression: the
        # numpy one promotes the whole panel to uint16 and back for every
        # frame, which measured at 5ms where these three saturating SIMD
        # passes measure well under one.
        cv2.merge((cover, cover, cover), dst=self._wide)
        cv2.multiply(self._wide, ink, dst=self._lit, scale=1.0 / 255.0)
        cv2.add(self.colour, self._lit, dst=self.colour)
        cv2.max(self.alpha, cover, dst=self.alpha)
        return True

    def _header(self, cover, ink, state: str) -> None:
        g = self.glyphs
        y = (HEAD_H - g.cell_h) // 2 + 1
        x = g.stamp(cover, "JARVIS", PAD_X, y)
        ink[y:y + g.cell_h, PAD_X:x] = HOT
        after = g.stamp(cover, " // SESSION LOG", x, y)
        ink[y:y + g.cell_h, x:after] = FAINT

        # Right-aligned: what it is doing, and whether the view is following.
        # Held back, it also says by how much -- otherwise there is no way to
        # tell a log that has gone quiet from one you have scrolled away from.
        mark = "LIVE" if self.live else f"HOLD -{self.scroll}"
        tail = f"{state.upper()}  {mark}"
        tx = self.width - PAD_X - RAIL_W - RAIL_GAP - len(tail) * g.cell_w
        end = g.stamp(cover, tail, max(x, tx), y)
        # HOLD in the warning colour, so it is obvious why new output stopped
        # appearing at the bottom.
        ink[y:y + g.cell_h, max(x, tx):end] = WARM if self.live else (120, 120, 255)

    def _body(self, cover, ink, cursor: bool) -> None:
        g = self.glyphs
        if not self._wrapped:
            return
        end = len(self._wrapped) - self.scroll
        shown = self._wrapped[max(0, end - self.rows):end]
        # Anchored to the bottom of the body, so a short log sits on the floor
        # of the terminal the way a shell's output does rather than floating
        # at the top with a gap underneath.
        first = self.rows - len(shown)
        for row, (colour, text) in enumerate(shown):
            y = self._body_y + (first + row) * g.cell_h
            g.stamp(cover, text, self._body_x, y)
            ink[y:y + g.cell_h, self._body_x:self.width - PAD_X] = colour
        if cursor and self.live:
            last = shown[-1][1]
            x = self._body_x + min(len(last), self.cols) * g.cell_w
            y = self._body_y + (self.rows - 1) * g.cell_h
            if x + g.cell_w <= self._rail_x:
                # Half-height, and only while something is arriving: a block
                # that blinks forever reads as a prompt waiting for you, which
                # is the opposite of what this window is.
                patch = cover[y + 3:y + g.cell_h - 2, x:x + g.cell_w - 2]
                patch[:] = 255
                ink[y:y + g.cell_h, x:x + g.cell_w] = HOT

    def _rail(self, colour, alpha) -> None:
        """A bar down the right showing where in the scrollback the view is."""
        total = len(self._wrapped)
        top = self._body_y
        height = self.rows * self.glyphs.cell_h
        x0, x1 = self._rail_x, self._rail_x + RAIL_W
        cv2.rectangle(colour, (x0, top), (x1, top + height), (34, 26, 18), -1)
        cv2.rectangle(alpha, (x0, top), (x1, top + height), GLASS_ALPHA, -1)
        if total <= self.rows:
            return                               # everything fits; no position
        span = max(int(height * self.rows / total), 12)
        # 0 scroll is the bottom of the rail, which is where the newest output
        # is -- the handle and the text have to agree about which way is old.
        offset = int((height - span) * (1.0 - self.scroll / (total - self.rows)))
        y0 = top + min(max(offset, 0), height - span)
        shade = HOT if not self.live else WARM
        cv2.rectangle(colour, (x0, y0), (x1, y0 + span), shade, -1)
        cv2.rectangle(alpha, (x0, y0), (x1, y0 + span), 255, -1)


class Terminal:
    """The console in a window of its own. `available` is False if it won't open.

    Its own thread and its own clock, for the reason in the module docstring:
    output has to keep arriving while the camera is not producing frames.
    """

    def __init__(self, x: int, y: int, width: int, height: int,
                 source=None) -> None:
        # Imported here rather than at the top so that the Console and the
        # glyph atlas stay importable off Windows -- everything about the
        # picture can then be built and tested without a window existing.
        from .widget import HoloWindow

        self.available = False
        self.notes: list[str] = []
        self.console = Console(width, height)
        if self.console.glyphs.source != "consolas":
            self.notes.append("Consolas was not usable, so the terminal is "
                              "drawn in OpenCV's font instead.")
        self.source = source                     # () -> (entries, state)
        self._drag_from = None
        self._drag_scroll = 0
        self._stop = threading.Event()
        self._window = HoloWindow(x, y, width, height, on_mouse=self._mouse,
                                  name="GestureControlTerminal")
        if not self._window.available:
            self.notes.append(self._window.unavailable_reason)
            return
        self.available = True
        self._thread = threading.Thread(target=self._loop, name="jarvis-terminal",
                                        daemon=True)
        self._thread.start()

    # -- interaction -------------------------------------------------------

    def _mouse(self, event: str, x: int, y: int) -> None:
        """Runs on the window's thread. Only ever moves the scroll position.

        Both ways of scrolling are here on purpose. The wheel is what anyone
        would reach for, and Windows delivers it to whatever is under the
        pointer -- but only while "scroll inactive windows on hover" is on,
        and this window never takes focus, so with that setting off no wheel
        message ever arrives. Dragging always works, and is also the only one
        available to the hand-tracked cursor, which has no wheel at all.
        """
        if event == "wheel":
            self.console.scroll_by(y * WHEEL_LINES)
        elif event == "down":
            self._drag_from = y
            self._drag_scroll = self.console.scroll
        elif event == "drag" and self._drag_from is not None:
            moved = (y - self._drag_from) // self.console.glyphs.cell_h
            # Dragging down pulls older text down into view, like moving a
            # sheet of paper rather than moving a viewport over it.
            self.console.scroll = 0
            self.console.scroll_by(self._drag_scroll + moved)
        elif event == "up":
            self._drag_from = None

    # -- drawing -----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.draw()
            except Exception:                                    # noqa: BLE001
                pass              # never let a drawing slip kill the window
            time.sleep(1.0 / REDRAW_HZ)

    def draw(self) -> None:
        entries, state = ([], "idle") if self.source is None else self.source()
        self.console.feed(entries)
        # The cursor blinks on the wall clock rather than a frame count, so it
        # keeps the same rhythm whatever the redraw rate turns out to be.
        blink = state in ("thinking", "speaking") and time.monotonic() % 1.0 < 0.6
        if not self.console.render(state, cursor=blink):
            return                    # the same picture; nothing to push
        if self._window.bgra is not None:
            from .widget import premultiply

            premultiply(self.console.colour, self.console.alpha,
                        self._window.bgra)
            self._window.blit()

    def close(self) -> None:
        self._stop.set()
        if self._window is not None:
            self._window.close()
        self.available = False
