"""The audio widget: what the speakers are playing, and a slider to change it.

A translucent panel in the corner of the screen, drawn in the same cyan the
orb and the glove are projected in.  The top half is a spectrum of the real output, mirrored about a centre
line so it reads as a waveform rather than as a bar chart.  The bottom half is
the system volume, which really is the system volume -- dragging it moves the
same control the tray icon does.

It lives in a `HoloWindow` rather than being painted onto the overlay, because
the overlay is click-through and a slider has to be draggable.  That is also
why it redraws itself on its own clock rather than with the camera: the camera
only produces frames when it has something new, and a meter that stutters when
you cover the lens would look broken.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .audio import BARS, Tap, Volume
from .widget import HoloWindow, premultiply

WIDTH, HEIGHT = 300, 132
FONT = cv2.FONT_HERSHEY_SIMPLEX

# The same palette as the orb, so the two read as one interface.
WARM = (246, 170, 44)
HOT = (255, 240, 198)
FAINT = (128, 100, 74)
INK = (226, 200, 164)
GLASS = (22, 15, 9)             # the panel's own tint, before its alpha

# How solid the panel is: the glass, and the frame around it. Not opaque --
# the point is to look like something projected onto the desk.
GLASS_ALPHA, EDGE_ALPHA = 132, 224

PAD = 12
WAVE_TOP, WAVE_BOTTOM = 30, 84
SLIDER_Y = 108
SLIDER_LEFT, SLIDER_RIGHT = PAD + 44, WIDTH - PAD - 44
# How near the slider a click counts. Generous: this is a small target that
# gets aimed at with a hand-tracked cursor as often as with a mouse.
GRAB = 22

REDRAW_HZ = 30


class AudioPanel:
    """The widget. `available` is False if there is nothing to show or drive."""

    def __init__(self, x: int, y: int) -> None:
        self.available = False
        self.notes: list[str] = []
        self.volume = Volume()
        self.tap = Tap()
        if not self.tap.start():
            self.notes.append(self.tap.unavailable_reason)
        if not self.volume.available:
            self.notes.append(self.volume.unavailable_reason)
        if not (self.tap.available or self.volume.available):
            self.notes.append("The audio widget has nothing to show.")
            return

        self._shown = self.volume.get()          # what the slider is drawing
        self._dragging = False
        self._stop = threading.Event()
        self._window = HoloWindow(x, y, WIDTH, HEIGHT, on_mouse=self._mouse,
                                  name="GestureControlAudio")
        if not self._window.available:
            self.notes.append(self._window.unavailable_reason)
            return
        self.available = True
        self._colour = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        self._alpha = np.zeros((HEIGHT, WIDTH), np.uint8)
        self._thread = threading.Thread(target=self._loop, name="jarvis-audio-ui",
                                        daemon=True)
        self._thread.start()

    # -- interaction -------------------------------------------------------

    def _mouse(self, event: str, x: int, y: int) -> None:
        """Runs on the window's thread. Only ever touches the volume."""
        if not self.volume.available:
            return
        if event == "down":
            if abs(y - SLIDER_Y) > GRAB:
                return
            self._dragging = True
        elif event == "up":
            self._dragging = False
            return
        elif not self._dragging:
            return
        span = max(SLIDER_RIGHT - SLIDER_LEFT, 1)
        level = (x - SLIDER_LEFT) / span
        level = min(max(level, 0.0), 1.0)
        self._shown = level
        self.volume.set(level)

    # -- drawing -----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._draw()
            except Exception:                                # noqa: BLE001
                pass          # never let a drawing slip kill the widget
            time.sleep(1.0 / REDRAW_HZ)

    def _draw(self) -> None:
        # While dragging, trust the drag: reading the system back mid-gesture
        # makes the handle stutter, because the value only lands after the
        # call returns and the read can catch the old one.
        if not self._dragging and self.volume.available:
            self._shown = self.volume.get()
        muted = self.volume.muted()

        colour, alpha = self._colour, self._alpha
        colour[:] = GLASS
        alpha[:] = GLASS_ALPHA

        # Frame: corner brackets rather than a full box, matching the orb.
        arm = 20
        for cx, cy, dx, dy in ((2, 2, 1, 1), (WIDTH - 3, 2, -1, 1),
                               (2, HEIGHT - 3, 1, -1),
                               (WIDTH - 3, HEIGHT - 3, -1, -1)):
            for target, value in ((colour, WARM), (alpha, EDGE_ALPHA)):
                cv2.line(target, (cx, cy), (cx + dx * arm, cy), value, 1, cv2.LINE_AA)
                cv2.line(target, (cx, cy), (cx, cy + dy * arm), value, 1, cv2.LINE_AA)

        self._text(colour, alpha, "AUDIO OUTPUT", (PAD, 20), 0.4, FAINT)
        if not self.tap.available:
            self._text(colour, alpha, "no signal", (PAD, WAVE_TOP + 26), 0.42, FAINT)
        else:
            self._wave(colour, alpha)
            if not self.tap.spectral:
                # Say so, rather than let a level history pass for a spectrum.
                self._text(colour, alpha, "level", (WIDTH - PAD - 34, 20), 0.34, FAINT)

        self._slider(colour, alpha, muted)
        premultiply(colour, alpha, self._window.bgra)
        self._window.blit()

    def _wave(self, colour, alpha) -> None:
        """The spectrum, mirrored about the middle so it reads as a wave."""
        bars = self.tap.bars
        middle = (WAVE_TOP + WAVE_BOTTOM) // 2
        reach = (WAVE_BOTTOM - WAVE_TOP) // 2
        left = PAD + 2
        span = WIDTH - 2 * left
        step = span / BARS
        thick = max(2, int(step * 0.62))
        for i, value in enumerate(bars):
            height = int(max(1.0, value * reach))
            x = int(left + step * (i + 0.5))
            # Hotter towards the top of each bar, so a loud one visibly
            # brightens rather than just growing.
            tint = tuple(int(w + (h - w) * min(1.0, float(value) * 1.3))
                         for w, h in zip(WARM, HOT))
            cv2.line(colour, (x, middle - height), (x, middle + height), tint,
                     thick, cv2.LINE_AA)
            cv2.line(alpha, (x, middle - height), (x, middle + height), 255,
                     thick, cv2.LINE_AA)
        # The centre line, which is what makes the mirroring read as a wave
        # rather than as two rows of bars.
        cv2.line(colour, (left, middle), (WIDTH - left, middle), FAINT, 1, cv2.LINE_AA)
        cv2.line(alpha, (left, middle), (WIDTH - left, middle), 190, 1, cv2.LINE_AA)

    def _slider(self, colour, alpha, muted: bool) -> None:
        level = 0.0 if muted else self._shown
        knob = int(SLIDER_LEFT + (SLIDER_RIGHT - SLIDER_LEFT) * level)

        for target, dim, lit, full in ((colour, FAINT, WARM, HOT),
                                       (alpha, 150, 255, 255)):
            cv2.line(target, (SLIDER_LEFT, SLIDER_Y), (SLIDER_RIGHT, SLIDER_Y),
                     dim, 2, cv2.LINE_AA)
            if knob > SLIDER_LEFT:
                cv2.line(target, (SLIDER_LEFT, SLIDER_Y), (knob, SLIDER_Y),
                         lit, 2, cv2.LINE_AA)
            cv2.circle(target, (knob, SLIDER_Y), 6, full, -1, cv2.LINE_AA)
            cv2.circle(target, (knob, SLIDER_Y), 6, lit, 1, cv2.LINE_AA)

        icon = "MUTE" if muted else "VOL"
        self._text(colour, alpha, icon, (PAD, SLIDER_Y + 5), 0.38,
                   (90, 100, 235) if muted else FAINT)
        label = "--" if not self.volume.available else f"{round(level * 100):3d}%"
        self._text(colour, alpha, label, (SLIDER_RIGHT + 10, SLIDER_Y + 5), 0.42,
                   INK)

    @staticmethod
    def _text(colour, alpha, text, at, scale, tint) -> None:
        """Text has to be written into the mask as well, or it is invisible."""
        cv2.putText(colour, text, at, FONT, scale, tint, 1, cv2.LINE_AA)
        cv2.putText(alpha, text, at, FONT, scale, 255, 1, cv2.LINE_AA)

    # -- lifecycle ---------------------------------------------------------

    def sink(self) -> None:
        if self.available:
            self._window.sink()

    def close(self) -> None:
        self._stop.set()
        self.tap.stop()
        if getattr(self, "_window", None) is not None:
            self._window.close()
        self.available = False
