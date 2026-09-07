"""Fullscreen calibration: point at each dot, pinch, and the mapping is fitted.

The window collects a burst of frames per dot rather than a single one, and
takes the median of each burst, because the fingertip wobbles by a few pixels
and the pinch itself drags the hand slightly.  The bursts also double as
training data for the pinch thresholds, so click detection ends up tuned to the
hand that calibrated it.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from dataclasses import dataclass, field

import numpy as np

from . import landmarks as lm
from .config import CALIBRATION_PATH, Settings
from .mapping import (AimSample, PointerMap, describe_inputs,
                      fit_pointer_map)
from .tracker import HandFrame, HandTracker

BG = "#0b0d10"
FG = "#e8edf4"
MUTED = "#7c8798"
ACCENT = "#4da3ff"
GOOD = "#4ade80"
WARN = "#fbbf24"


def target_grid(width: int, height: int, cols: int, rows: int,
                margin: float) -> list[tuple[float, float]]:
    """Dot positions, row by row.

    The dots sit `margin` in from each edge: the fit extrapolates poorly
    outside the calibrated region, so the ring of dots should enclose as much
    of the screen as the user can comfortably point at.
    """
    xs = (np.linspace(margin, 1.0 - margin, cols) if cols > 1 else np.array([0.5])) * width
    ys = (np.linspace(margin, 1.0 - margin, rows) if rows > 1 else np.array([0.5])) * height
    return [(float(x), float(y)) for y in ys for x in xs]


@dataclass
class _Dot:
    position: tuple[float, float]
    samples: list[AimSample] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return bool(self.samples)


class CalibrationWindow:
    """The calibration screen.  Call :meth:`run`; it returns a model or None."""

    def __init__(self, tracker: HandTracker, settings: Settings):
        self.tracker = tracker
        self.settings = settings

        self.root = tk.Tk()
        self.root.title("Gesture Control - Calibration")
        self.root.configure(bg=BG)
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.config(cursor="none")
        self.root.update_idletasks()

        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0,
                                width=self.width, height=self.height)
        self.canvas.pack(fill="both", expand=True)

        self.dots = [_Dot(p) for p in target_grid(
            self.width, self.height, settings.grid_cols, settings.grid_rows,
            settings.margin_frac)]
        self.index = 0
        self.phase = "intro"          # intro -> collect -> verify -> done
        self.model: PointerMap | None = None
        self.result: PointerMap | None = None

        # Pinch-threshold training data, gathered as a side effect of clicking
        # the dots: ratios seen with the hand open vs. with it pinched.
        self.open_ratios: list[float] = []
        self.closed_ratios: list[float] = []
        self.thresholds: tuple[float, float] | None = None
        self.coverage = 0.0
        self.inputs: dict = {}
        self.raw_samples: list = []
        self.raw_targets: list = []

        self._pinching = False
        self._capturing = False
        self._dwell_anchor = None
        self._dwell_since = 0.0
        self._advance_tip = None
        self._needs_release = False
        self._last_ratio = float('nan')
        self._last_pinched = False
        self._settle_left = 0
        self._last_stamp = -1.0
        self._hand_seen_since: float | None = None
        self._message = ""
        self._message_until = 0.0
        self._t0 = time.monotonic()

        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("<Return>", self._on_enter)
        self.root.bind("<KP_Enter>", self._on_enter)
        self.root.bind("r", self._on_redo)
        self.root.bind("R", self._on_redo)
        self.root.bind("<BackSpace>", self._on_back)
        self.root.bind("<space>", self._on_start)
        self.root.protocol("WM_DELETE_WINDOW", self._on_escape)

    # -- public ------------------------------------------------------------

    def run(self) -> PointerMap | None:
        self.root.after(16, self._tick)
        self.root.mainloop()
        return self.result

    # -- key handlers ------------------------------------------------------

    def _on_escape(self, _event=None) -> None:
        self.result = None
        self._close()

    def _on_enter(self, _event=None) -> None:
        if self.phase == "verify" and self.model is not None:
            self.result = self.model
            self._close()

    def _on_redo(self, _event=None) -> None:
        if self.phase == "verify":
            for dot in self.dots:
                dot.samples.clear()
            self.open_ratios.clear()
            self.closed_ratios.clear()
            self.index = 0
            self.model = None
            self.phase = "collect"
            self._flash("Starting over")
        elif self.phase == "collect":
            self.dots[self.index].samples.clear()
            self._reset_capture()
            self._needs_release = True
            self._flash("Retrying this dot")

    def _on_start(self, _event=None) -> None:
        """Space begins calibration.

        The gesture way in is a pinch, which is precisely the thing that may be
        misbehaving when someone comes here to fix it. There has to be a way in
        that cannot be broken by a bad threshold.
        """
        if self.phase == "intro":
            self.phase = "collect"
            self._reset_capture()
            self._needs_release = True

    def _on_back(self, _event=None) -> None:
        if self.phase == "collect" and self.index > 0:
            self.index -= 1
            self.dots[self.index].samples.clear()
            self._reset_capture()
            self._needs_release = True
            self._flash("Back one dot")

    def _close(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # -- main loop ---------------------------------------------------------

    def _tick(self) -> None:
        try:
            self.tracker.raise_if_failed()
        except Exception as exc:  # noqa: BLE001 - report and bail out cleanly
            self._message = f"Tracking failed: {exc}"
            self._message_until = time.monotonic() + 5.0

        hand = self.tracker.latest()
        fresh = hand is not None and hand.stamp != self._last_stamp
        if hand is not None:
            self._last_stamp = hand.stamp

        if self.phase == "intro":
            self._update_intro(hand, fresh)
        elif self.phase == "collect":
            self._update_collect(hand, fresh)

        self._draw(hand)
        if self.root.winfo_exists():
            self.root.after(16, self._tick)

    def _update_intro(self, hand: HandFrame | None, fresh: bool) -> None:
        if hand is None:
            self._hand_seen_since = None
            return
        now = hand.stamp   # frame clock, like every other timing decision here
        if self._hand_seen_since is None:
            self._hand_seen_since = now
        # Keep the pinch readout live on the intro screen too, so a threshold
        # that reads an open hand as pinched is obvious before starting rather
        # than after twelve dots have filled themselves in.
        self._last_ratio = self._trigger_value(hand)
        self._last_pinched = self._is_pinched(hand, self._last_ratio)
        # A pinch starts calibration, but only once the hand has been tracked
        # steadily for a moment -- otherwise a half-detected hand fires it.
        if now - self._hand_seen_since > 0.6 and fresh:
            if self._is_pinched(hand):
                self.phase = "collect"
                self._reset_capture()
                self._needs_release = True

    def _update_collect(self, hand: HandFrame | None, fresh: bool) -> None:
        if hand is None or not fresh:
            return

        s = self.settings
        ratio = self._trigger_value(hand)
        pinched = self._is_pinched(hand, ratio)
        self._last_ratio, self._last_pinched = ratio, pinched
        (self.closed_ratios if pinched else self.open_ratios).append(ratio)

        # Holding the fingertip still captures a dot as well as pinching does.
        # Calibration is where the pinch thresholds come from, so making it
        # depend on those same thresholds is a loop with no way in when they
        # are wrong -- as they are for anyone whose hand the defaults do not
        # suit. Dwelling needs nothing calibrated at all.
        tip = hand.norm[lm.INDEX_TIP]
        if (self._dwell_anchor is None
                or float(np.linalg.norm(tip - self._dwell_anchor)) > s.dwell_radius):
            self._dwell_anchor = tip
            self._dwell_since = hand.stamp
        dwelling = (hand.stamp - self._dwell_since) * 1000.0 > s.dwell_ms

        # Each dot needs its own deliberate trigger: the hand must be seen
        # un-pinched and must have moved since the last capture.  Without this,
        # a threshold too loose to ever release turns one pinch into a cascade
        # that fills every dot from wherever the hand happens to be.
        if self._needs_release:
            moved = (self._advance_tip is None
                     or float(np.linalg.norm(tip - self._advance_tip)) > s.dwell_radius)
            if not pinched and moved:
                self._needs_release = False
            else:
                self._pinching = pinched
                self._dwell_since = hand.stamp     # no dwell credit while blocked
                return

        capturing = pinched or dwelling
        dot = self.dots[self.index]
        if capturing and not self._capturing:
            # A pinch tugs the index finger off target as the thumb closes, so
            # the first frames of one are discarded.  A dwell has no such
            # transient -- it is only recognised after the hand has already
            # been still -- so it needs no settling.
            self._settle_left = s.settle_frames if pinched else 0
            dot.samples.clear()
        elif not capturing and self._capturing and not dot.done:
            if dot.samples:
                self._flash("Keep holding until the ring fills")
            dot.samples.clear()

        self._pinching = pinched
        self._capturing = capturing

        if capturing:
            if self._settle_left > 0:
                self._settle_left -= 1
            else:
                dot.samples.append(hand.aim_sample(s.aim_mode))
                if len(dot.samples) >= s.samples_per_point:
                    self._advance_tip = tip
                    self._advance()

    def _advance(self) -> None:
        self.index += 1
        self._reset_capture()
        self._needs_release = True
        if self.index >= len(self.dots):
            self.index = len(self.dots) - 1
            self._finish_collection()

    def _reset_capture(self) -> None:
        self._pinching = False
        self._capturing = False
        self._dwell_anchor = None
        self._settle_left = 0

    def _is_pinched(self, hand: HandFrame, ratio: float | None = None) -> bool:
        """Is the hand making the capture gesture right now?

        In palm mode that is a closed fist, and the wide gap between the fist
        and open thresholds serves as the hysteresis. In finger mode it is a
        thumb-to-index pinch.
        """
        s = self.settings
        if s.aim_mode == "palm":
            c = hand.curl if ratio is None else ratio
            return c < (s.open_curl if self._pinching else s.fist_curl)
        r = hand.pinch(lm.INDEX_TIP) if ratio is None else ratio
        return r < (s.pinch_off if self._pinching else s.pinch_on)

    def _trigger_value(self, hand: HandFrame) -> float:
        """The number the capture gesture is judged on, for display."""
        return hand.curl if self.settings.aim_mode == "palm" else hand.pinch(lm.INDEX_TIP)

    # -- fitting -----------------------------------------------------------

    def _finish_collection(self) -> None:
        samples = [AimSample.average(dot.samples) for dot in self.dots if dot.done]
        targets = [dot.position for dot in self.dots if dot.done]
        self.coverage = self._sweep_coverage(samples)
        self.inputs = describe_inputs(samples)
        self.raw_samples, self.raw_targets = samples, targets
        try:
            self.model = fit_pointer_map(samples, targets, (self.width, self.height))
        except (ValueError, np.linalg.LinAlgError) as exc:
            # Every dot is already captured, so there is nothing to retry from
            # here; start the whole grid again rather than dead-ending.
            self._flash(f"Fit failed ({exc}) - starting over", 6.0)
            for dot in self.dots:
                dot.samples.clear()
            self.index = 0
            self._reset_capture()
            return
        self.thresholds = self._derive_thresholds()
        self.phase = "verify"

    @staticmethod
    def _sweep_coverage(samples) -> float:
        """Fraction of the camera frame the fingertip covered, as a width.

        This is the single biggest thing the user controls.  A hand that only
        travels a quarter of the frame has every pixel of tracking noise
        magnified four times on the way to the screen; one that travels half
        the frame halves that.  Measuring it here means the app can say so
        while the user is still sitting down to fix it.
        """
        if not samples:
            return 0.0
        tips = np.array([s.tip for s in samples], dtype=float)
        span = tips.max(axis=0) - tips.min(axis=0)
        return float(span[0])

    def _derive_thresholds(self) -> tuple[float, float] | None:
        """Set the click thresholds from this hand instead of a global guess.

        Hands differ: a long thumb closes to a smaller fraction of the palm
        than a short one.  Calibration has just watched both states, so the gap
        between them is measured rather than assumed.
        """
        if self.settings.aim_mode == "palm":
            return None
        if len(self.closed_ratios) < 20 or len(self.open_ratios) < 20:
            return None
        closed_hi = float(np.percentile(self.closed_ratios, 85))
        open_lo = float(np.percentile(self.open_ratios, 15))
        if open_lo - closed_hi < 0.12:
            return None   # the two states overlap; the defaults are safer
        on = float(np.clip(closed_hi + 0.35 * (open_lo - closed_hi), 0.18, 0.75))
        off = closed_hi + 0.75 * (open_lo - closed_hi)
        # Keep the hysteresis band narrow.  If the resting pose happens to hold
        # the thumb far from the finger, the raw numbers would put the release
        # threshold near a fully open hand, and a click would refuse to let go.
        off = float(np.clip(off, on + 0.06, on * 1.5))
        # The release threshold has to sit below the ratios seen with the hand
        # open, or the pinch can never end.  The arithmetic above should always
        # land there; checked explicitly because the cost of being wrong is a
        # mouse button stuck down.
        if off >= open_lo or on >= off:
            return None
        return on, off

    # -- drawing -----------------------------------------------------------

    def _flash(self, text: str, seconds: float = 2.0) -> None:
        self._message = text
        self._message_until = time.monotonic() + seconds

    def _draw(self, hand: HandFrame | None) -> None:
        c = self.canvas
        c.delete("all")
        now = time.monotonic()

        self._draw_hand_preview(hand)   # first, so dots draw over it
        if self.phase == "intro":
            self._draw_intro(hand)
        elif self.phase == "collect":
            self._draw_collect(hand, now)
        elif self.phase == "verify":
            self._draw_verify(hand)

        if self._message and now < self._message_until:
            c.create_text(self.width / 2, self.height - 90, text=self._message,
                          fill=WARN, font=("Segoe UI", 15))

    def _draw_intro(self, hand: HandFrame | None) -> None:
        c = self.canvas
        PALM = self.settings.aim_mode == "palm"
        cx, cy = self.width / 2, self.height / 2
        c.create_text(cx, cy - 150, text="Gesture Control",
                      fill=FG, font=("Segoe UI Light", 52))
        c.create_text(cx, cy - 70, fill=MUTED, font=("Segoe UI", 17),
                      text=f"About to calibrate {len(self.dots)} points. "
                           "Sit where you normally sit.")

        lines = [
            "1.  Point at each dot with your index finger.",
            ("2.  Close your hand into a fist and hold it closed --"
             if PALM else "2.  Pinch your thumb and index finger together and hold --"),
            "     or just hold your hand still on the dot, if that plays up.",
            "3.  Keep aiming until the ring around the dot fills, then open up.",
            "",
            "Space begins   ·   Backspace redoes a dot   ·   R restarts   ·   Esc cancels",
        ]
        for i, line in enumerate(lines):
            c.create_text(cx, cy + 10 + i * 34, text=line, fill=FG if i < 3 else MUTED,
                          font=("Segoe UI", 16))

        if hand is None:
            status, colour = "Raise your hand where the camera can see it...", WARN
        else:
            status, colour = (("Close your fist to begin, or press Space" if PALM
                               else "Pinch to begin, or press Space"), GOOD)
        c.create_text(cx, cy + 230, text=status, fill=colour, font=("Segoe UI", 20))

    def _draw_collect(self, hand: HandFrame | None, now: float) -> None:
        c = self.canvas
        PALM = self.settings.aim_mode == "palm"
        # Dots before the cursor are finished; the current one is only
        # part-captured, however many samples it happens to hold right now.
        done = self.index

        for i, dot in enumerate(self.dots):
            x, y = dot.position
            if i < self.index:
                c.create_oval(x - 6, y - 6, x + 6, y + 6, fill=GOOD, outline="")
            elif i > self.index:
                c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#243040", outline="")

        dot = self.dots[self.index]
        x, y = dot.position
        target = self.settings.samples_per_point
        progress = len(dot.samples) / target if target else 0.0

        # A slow pulse draws the eye to the live dot without being distracting.
        pulse = 4.0 * math.sin((now - self._t0) * 3.0)
        outer = 34 + pulse
        c.create_oval(x - outer, y - outer, x + outer, y + outer,
                      outline="#1e2a3a", width=2)
        if progress > 0:
            c.create_arc(x - outer, y - outer, x + outer, y + outer,
                         start=90, extent=-359.9 * min(progress, 1.0),
                         style=tk.ARC, outline=ACCENT, width=6)
        inner = 11 if self._pinching else 8
        c.create_oval(x - inner, y - inner, x + inner, y + inner,
                      fill=ACCENT if self._pinching else FG, outline="")

        if hand is None:
            hint, colour = "Hand lost - move back into the camera view", WARN
        elif self._pinching:
            hint, colour = "Hold...", ACCENT
        else:
            hint, colour = (("Aim at the dot with your palm, then close your fist"
                             if PALM else "Point at the dot, then pinch"), MUTED)
        c.create_text(self.width / 2, self.height - 140, text=hint,
                      fill=colour, font=("Segoe UI", 19))
        c.create_text(self.width / 2, self.height - 50,
                      text=f"{done} / {len(self.dots)}",
                      fill=MUTED, font=("Segoe UI", 15))

    def _draw_verify(self, hand: HandFrame | None) -> None:
        c = self.canvas
        assert self.model is not None
        meta = self.model.meta
        loocv = meta.get("loocv_rms_px")

        for dot in self.dots:
            x, y = dot.position
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#243040", outline="")

        cx, cy = self.width / 2, self.height / 2
        pointer = None
        if hand is not None:
            raw = self.model.predict(hand.aim_sample(self.settings.aim_mode))
            pointer = (min(max(raw[0], 0), self.width),
                       min(max(raw[1], 0), self.height))
        # The whole point of this screen is to sweep the pointer around, so the
        # text steps out of the way instead of fighting the crosshair for it.
        over_text = pointer is not None and cy - 190 < pointer[1] < cy + 150
        head, body, faint = (GOOD, FG, MUTED) if not over_text else ("#25402f", "#2b3340", "#232a35")

        c.create_text(cx, cy - 120, text="Calibrated", fill=head,
                      font=("Segoe UI Light", 44))

        poor = loocv is not None and loocv > self.height * 0.03
        if loocv is None:
            accuracy = "Accuracy unknown (too few dots to cross-validate)"
        else:
            accuracy = (f"Expected accuracy ±{loocv:.0f} px "
                        f"({loocv / self.height * 100:.1f}% of screen height)")
        c.create_text(cx, cy - 55, text=accuracy,
                      fill=WARN if poor else body, font=("Segoe UI", 19))

        # A bad fit is worth interrupting for.  Accepting one produces a cursor
        # that misses by inches, and the cause is almost never something the
        # user would guess at from a number alone.
        if poor:
            c.create_text(cx, cy - 90, text="This calibration is not usable",
                          fill=WARN, font=("Segoe UI", 20, "bold"))
            for i, line in enumerate([
                "The dots did not line up with a consistent view of your hand. Usually one of:",
                "a pinch registered while you were not yet pointing at the dot,",
                "you shifted position part-way through, or your hand was too dim to track.",
                "Press R and redo it, pausing on each dot before you pinch.",
            ]):
                c.create_text(cx, cy + 130 + i * 26, text=line, fill=WARN,
                              font=("Segoe UI", 13))
        aims = meta.get("aims_by", "position")
        moved = self.inputs.get("position_span_x", 0.0)
        turned = self.inputs.get("facing_span_deg", 0.0)
        c.create_text(cx, cy + 155, fill=faint, font=("Segoe UI", 13),
                      text=f"read as aiming by {aims}   ·   "
                           f"hand moved across {moved:.0%} of the camera view   ·   "
                           f"turned through {turned:.0f}°")

        c.create_text(cx, cy - 20, fill=faint, font=("Segoe UI", 13),
                      text=f"model: homography"
                           f"{' + ' + meta['feature_set'] + ' correction' if meta.get('feature_set') not in (None, 'none') else ' only'}"
                           f"   ·   {meta.get('points', 0)} dots"
                           f"   ·   fit residual {meta.get('fit_rms_px', 0):.0f} px")

        # Say what was missing, not just that something was. Too little of
        # either signal is fixable; too little of both is the real problem.
        moved = self.inputs.get("position_span_x", 0.0)
        turned = self.inputs.get("facing_span_deg", 0.0)
        advice = None
        if "ray" in aims and poor:
            # The ray model is the one the user asks for by name, and it is
            # also the one with a lever arm: the hand is half a metre from the
            # screen, so a degree of error in its angle is a centimetre of
            # error on the glass. Moving the hand has no such multiplier.
            advice = ("Aiming by angle alone magnifies small errors: a degree of "
                      "wobble in your hand becomes about a centimetre on screen. "
                      "Moving your hand around as well as turning it is steadier.")
        elif moved < 0.35 and turned < 20.0:
            advice = ("Your hand barely moved (%.0f%% of the view) and barely turned "
                      "(%.0f°). Aim by doing more of one or the other." % (moved * 100, turned))
        elif moved < 0.35 and "orientation" not in aims:
            advice = ("Your hand only crossed %.0f%% of the camera view. Bigger "
                      "arm movements, or sitting closer, would sharpen this." % (moved * 100))
        if advice:
            c.create_text(cx, cy + 22, fill=WARN, font=("Segoe UI", 14), text=advice)

        c.create_text(cx, cy + 60, fill=body, font=("Segoe UI", 17),
                      text="Move your hand around and check the ring tracks where you point.")
        c.create_text(cx, cy + 100, fill=faint, font=("Segoe UI", 15),
                      text="Enter to save and start controlling   ·   R to recalibrate   ·   Esc to cancel")

        if pointer is not None:
            px, py = pointer
            pinched = hand.pinch(lm.INDEX_TIP) < self.settings.pinch_on
            colour = ACCENT if pinched else FG
            r = 16 if pinched else 22
            c.create_oval(px - r, py - r, px + r, py + r, outline=colour, width=3)
            c.create_line(px - 34, py, px - r - 6, py, fill=colour)
            c.create_line(px + r + 6, py, px + 34, py, fill=colour)
            c.create_line(px, py - 34, px, py - r - 6, fill=colour)
            c.create_line(px, py + r + 6, px, py + 34, fill=colour)

    def _draw_hand_preview(self, hand: HandFrame | None) -> None:
        """A small mirrored skeleton, so it is obvious whether tracking is alive."""
        c = self.canvas
        PALM = self.settings.aim_mode == "palm"
        box = 150
        pad = 28
        x0, y0 = self.width - box - pad, self.height - box - pad
        if self.phase == "collect":
            # Sit in the corner diagonally opposite the dot being aimed at,
            # otherwise the box covers the very dot the user is trying to hit.
            dx, dy = self.dots[self.index].position
            x0 = pad if dx > self.width / 2 else self.width - box - pad
            y0 = pad if dy > self.height / 2 else self.height - box - pad
        c.create_rectangle(x0, y0, x0 + box, y0 + box, outline="#1b2430", width=1)

        if hand is None:
            c.create_text(x0 + box / 2, y0 + box / 2, text="no hand",
                          fill="#3a4657", font=("Segoe UI", 11))
            return

        pts = hand.norm.copy()
        pts[:, 0] = 1.0 - pts[:, 0]            # mirror: the user sees themselves
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = max(float((hi - lo).max()), 1e-6)
        pts = (pts - lo) / span * (box - 34) + np.array([x0 + 17, y0 + 17])

        for a, b in lm.CONNECTIONS:
            c.create_line(pts[a, 0], pts[a, 1], pts[b, 0], pts[b, 1],
                          fill="#2f4a6b", width=2)
        for i, (px, py) in enumerate(pts):
            hot = i in (lm.THUMB_TIP, lm.INDEX_TIP)
            r = 4 if hot else 2
            c.create_oval(px - r, py - r, px + r, py + r,
                          fill=ACCENT if hot else "#5b6c82", outline="")
        c.create_text(x0 + box / 2, y0 + box - 6,
                      text=f"{self.tracker.fps:.0f} fps", fill="#3a4657",
                      font=("Segoe UI", 9))

        # Showing the raw pinch number turns "the dots fill by themselves" from
        # a mystery into something with a visible cause.
        if self._last_ratio == self._last_ratio:      # not NaN
            state = ("FIST" if self._last_pinched else "open") if PALM else                     ("PINCH" if self._last_pinched else "open")
            colour = ACCENT if self._last_pinched else "#3a4657"
            c.create_text(x0 + box / 2, y0 - 12,
                          text=f"{state}  {self._last_ratio:+.2f}"
                               f"  (fires below {self.settings.fist_curl:+.2f})"
                          if PALM else
                          f"{state}  {self._last_ratio:.2f}"
                          f"  (fires below {self.settings.pinch_on:.2f})",
                          fill=colour, font=("Segoe UI", 10))


def _save_raw(window: "CalibrationWindow") -> None:
    """Write the per-dot measurements out.

    A calibration that comes out badly is nearly impossible to reason about
    from the single accuracy number it prints. The inputs it was given are
    small enough to just keep.
    """
    import json

    from .config import ROOT

    try:
        payload = {
            "screen": [window.width, window.height],
            "aim_mode": window.settings.aim_mode,
            "inputs": window.inputs,
            "dots": [
                {"target": list(target),
                 "palm": list(sample.tip),
                 "facing": list(sample.direction),
                 "scale": sample.scale}
                for sample, target in zip(window.raw_samples, window.raw_targets)
            ],
        }
        (ROOT / "calibration_data.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def calibrate(tracker: HandTracker, settings: Settings) -> PointerMap | None:
    """Run calibration and, if the user accepts it, save the model."""
    window = CalibrationWindow(tracker, settings)
    model = window.run()
    if model is None:
        return None

    if window.thresholds is not None:
        on, off = window.thresholds
        settings.pinch_on, settings.pinch_off = on, off
        settings.right_pinch_on, settings.right_pinch_off = on, off
        model.meta["pinch_on"] = on
        model.meta["pinch_off"] = off
        Settings.update_file(pinch_on=on, pinch_off=off,
                             right_pinch_on=on, right_pinch_off=off)
        print(f"Pinch thresholds tuned to your hand: click below {on:.2f}, "
              f"release above {off:.2f} (palm lengths).")

    _save_raw(window)
    model.meta["aim_mode"] = settings.aim_mode
    model.meta.update({f"input_{k}": v for k, v in window.inputs.items()})
    print(f"Read as aiming by {model.meta.get('aims_by', 'position')}: "
          f"your hand moved across {window.inputs.get('position_span_x', 0):.0%} of "
          f"the camera view and turned through "
          f"{window.inputs.get('facing_span_deg', 0):.0f} degrees.")
    model.meta["sweep_coverage"] = window.coverage
    if window.coverage and window.coverage < 0.35:
        print(f"Note: your hand covered only {window.coverage:.0%} of the camera view. "
              "Sitting closer or pointing with larger movements reduces jitter "
              "roughly in proportion.")

    model.save(CALIBRATION_PATH)
    print(f"Calibration saved to {CALIBRATION_PATH}")
    loocv = model.meta.get("loocv_rms_px")
    if loocv is not None:
        print(f"Expected accuracy: +/-{loocv:.0f} px "
              f"({loocv / model.screen[1] * 100:.1f}% of screen height)")
    return model
