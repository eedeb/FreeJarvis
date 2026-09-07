"""The run loop: hand pose in, real mouse events out.

Direct aiming (the default)
    Picture the camera's view blown up to fill the monitor, then made
    invisible. Wherever your hand sits on that picture is where the cursor
    goes. Nothing is calibrated, because there is nothing to fit.

    move your hand                   move the cursor
    pinch thumb + index              left click
    pinch, then move                 drag (holding still keeps it a click)
    index + middle up, moved up/down scroll
    pinch thumb + middle finger      right click
    thumb out, everything else shut on-screen keyboard, for typing

Palm aiming (aim_mode "palm", calibrated against dots)
    open hand                        move the cursor
    close a fist                     left click; keep it closed to drag
    index + middle up, moved up/down scroll
    index + thumb out, others closed right click

Finger aiming (aim_mode "finger")
    point with the index finger      move the cursor
    pinch thumb + index              left click (hold to drag)
    pinch thumb + middle             right click
    index + middle up, moved up/down scroll
    open palm, held ~1s              pause / resume control

Palm aiming has no pause pose: an open hand is how you aim, so it cannot also
mean stop. Use P in the preview window.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from . import gauntlet
from . import landmarks as lm
from .config import Settings
from . import keyboard
from .filters import OneEuroFilter
from .mapping import PointerMap
from .mouse import Mouse, double_click_time_ms
from .session import Frame, SessionLog
from .targets import Target, TargetFinder, snap
from .tracker import HandFrame, HandTracker

_HELP = {
    "direct": ("the camera view is the screen: wherever your hand is in it, "
               "that is where the cursor goes\n"
               "   move = move   ·   pinch = click, then move to drag   "
               "·   pinch middle = right click   ·   two fingers = scroll   "
               "·   thumbs-up = keyboard"),
    "palm": ("open hand = move   ·   close a fist = click, hold to drag   "
             "·   two fingers = scroll   ·   finger + thumb = right click"),
    "finger": ("point = move   ·   pinch index = click/drag   "
               "·   pinch middle = right click   ·   two fingers = scroll   "
               "·   open palm = pause"),
}


class _Debounced:
    """A boolean that only changes after the input has agreed for N frames.

    Hysteresis on the pinch distance stops slow drift from chattering, but it
    does nothing about a single badly-tracked frame in which the thumb appears
    to jump onto the fingertip.  That one frame is enough to fire a click, so
    the press also has to survive a couple of frames before it counts.
    Releasing is deliberately quicker than pressing: a click that takes an
    extra frame is invisible, a button left stuck down is not.
    """

    def __init__(self, press_frames: int = 2, release_frames: int = 1):
        self.press_frames = max(1, press_frames)
        self.release_frames = max(1, release_frames)
        self.state = False
        self._pending = False
        self._count = 0

    def update(self, raw: bool, release_frames: int | None = None) -> bool:
        if raw == self.state:
            self._count = 0
            return self.state
        if raw != self._pending:
            self._pending, self._count = raw, 1
        else:
            self._count += 1
        wait = self.release_frames if release_frames is None else max(1, release_frames)
        needed = self.press_frames if raw else wait
        if self._count >= needed:
            self.state = raw
            self._count = 0
        return self.state

    def force(self, value: bool) -> None:
        self.state = self._pending = value
        self._count = 0


class GestureController:
    def __init__(self, tracker: HandTracker, model: PointerMap, settings: Settings):
        self.tracker = tracker
        self.model = model
        self.settings = settings
        self.mouse = Mouse()

        self.filter = OneEuroFilter(settings.filter_min_cutoff,
                                    settings.filter_beta,
                                    settings.filter_dcutoff)
        # Recent cursor positions, so a click can be placed where the finger
        # was just *before* the pinch started rather than where the pinch
        # dragged it to.
        self.history: deque[tuple[float, np.ndarray]] = deque(maxlen=180)

        # Set by run(); left None in tests and when snapping is switched off.
        self.finder: TargetFinder | None = None
        self.snap_target: Target | None = None

        # Raw (unmapped, unsnapped) predictions, used to estimate how noisy the
        # tracking currently is and report it back to the user.
        self.raw_history: deque[np.ndarray] = deque(maxlen=15)

        self.paused = False
        self.running = True
        self.mode = "idle"

        self._left_pinched = False
        self._left_since = 0.0
        self._dragging = False
        self._right_pinched = False
        self._left_debounce = _Debounced(settings.pinch_press_frames)
        self._right_debounce = _Debounced(settings.pinch_press_frames)
        self._fist_debounce = _Debounced(settings.fist_press_frames)
        self._scroll_debounce = _Debounced(settings.pose_debounce_frames)
        self._menu_debounce = _Debounced(settings.pose_debounce_frames)
        self._held_pose = "open"
        self._menu_fired = False
        # Recent pinch measurements, median-filtered before thresholding.
        depth = max(settings.pinch_median_frames, settings.drag_hold_frames, 1)
        self._pinch_history: deque[float] = deque(maxlen=depth)
        self._middle_history: deque[float] = deque(maxlen=depth)
        # A longer trail of (time, gap), for the sustained-slack release.
        self._gap_trail: deque[tuple[float, float]] = deque(maxlen=120)
        # Running reference for the finger bones, and the last gap measured on
        # a frame whose bones looked right.
        self._bone_trail: deque[np.ndarray] = deque(maxlen=45)
        self._last_good_gap: float | None = None
        self._reject_run = 0
        self.rejected_frames = 0
        self._keyboard_debounce = _Debounced(8)
        self._keyboard_shown = False
        self._click_pos: np.ndarray | None = None
        self._press_origin = np.zeros(2)
        self._drag_offset = np.zeros(2)
        self._drag_run = 0
        self._last_click_at = -1e9
        self._double_click_ms = double_click_time_ms()
        self.log = SessionLog()
        self._signals = (0.0, 0.0, 0.0)   # index raw, index used, middle
        self._scrolling = False
        self._scroll_anchor = 0.0
        self._palm_since: float | None = None
        self._palm_anchor: np.ndarray | None = None
        self._pause_cooldown = 0.0
        self._last_stamp = -1.0
        self._last_seen = time.monotonic()
        self._position = np.array([self.model.screen[0] / 2.0,
                                   self.model.screen[1] / 2.0])

    # -- gesture helpers ---------------------------------------------------

    def _pinched(self, ratio: float, active: bool, on: float, off: float) -> bool:
        """Hysteresis: it takes a firmer pinch to start than to keep going."""
        return ratio < (off if active else on)

    def _anchor_position(self, when: float) -> np.ndarray:
        """Where the cursor was just before the pinch started.

        Averaging a short window rather than taking the single nearest frame
        cuts the click error roughly in half, because per-frame landmark jitter
        is the dominant error source and this is the one place it can be
        averaged away for free: the cursor is frozen from pinch-down until
        release anyway, so there is no lag to pay for it.  The median keeps one
        bad frame from dragging the click off target.
        """
        cutoff = when - self.settings.click_anchor_ms / 1000.0
        window = self.settings.click_anchor_window_ms / 1000.0
        tolerance = self.settings.click_anchor_tolerance_frac * self.model.screen[1]

        # Walk back from the cutoff, keeping frames while the hand was resting
        # near the same spot and stopping the moment it was somewhere else.
        # A fixed-length window would otherwise average in the approach when
        # someone pinches the instant they arrive, throwing the click far wider
        # than using a single frame would have.
        kept: list[np.ndarray] = []
        for stamp, pos in reversed(self.history):
            if stamp > cutoff:
                continue
            if not kept:
                kept.append(pos)
                continue
            if stamp < cutoff - window or np.linalg.norm(pos - kept[0]) > tolerance:
                break
            kept.append(pos)

        if kept:
            return np.median(np.stack(kept), axis=0)
        return self._position

    def _trustworthy(self, hand: HandFrame) -> bool:
        """Do this frame's finger bones still have the lengths they had?

        Fingers do not stretch. When the reported bone lengths jump, the
        tracker has lost the hand rather than the hand having moved, and the
        gap it reports that frame is meaningless -- which is how a plainly
        open hand comes to read 0.7cm and a firm pinch 8.2cm. Those frames are
        better dropped than thresholded.
        """
        bones = hand.bones()
        if bones is None or not np.all(np.isfinite(bones)):
            return True                     # nothing to judge with; allow it
        self._bone_trail.append(bones)
        if len(self._bone_trail) < 8:
            return True                     # no settled reference yet
        reference = np.median(np.stack(self._bone_trail), axis=0)
        good = reference > 1e-4
        if not np.any(good):
            return True
        drift = np.abs(bones[good] - reference[good]) / reference[good]
        return bool(drift.max() <= self.settings.bone_tolerance)

    def _pinch_signal(self, hand: HandFrame, tip: int, history: deque,
                      active: bool, min_frames: int | None = None) -> float:
        """Filtered thumb-to-fingertip gap, in centimetres where possible.

        Lopsided on purpose: the median to start a pinch, so one stray close
        reading cannot fire one, and the smallest of the recent frames to end
        it, so fingertips occluding each other cannot chop one gesture into
        several.
        """
        gap = hand.pinch_cm(tip)
        value = gap if gap is not None else hand.pinch(tip)
        if tip == lm.INDEX_TIP:
            bad = not self._trustworthy(hand)
            if bad and self._reject_run < self.settings.bone_reject_limit                     and self._last_good_gap is not None:
                # Hold the last trustworthy reading -- but only briefly, so a
                # persistently odd-looking hand cannot freeze the click state.
                self._reject_run += 1
                self.rejected_frames += 1
                value = self._last_good_gap
            else:
                self._reject_run = 0
                self._last_good_gap = value
        history.append(value)
        vals = list(history)
        if not active:
            short = vals[-self.settings.pinch_median_frames:]
            return float(np.median(short))
        window = min_frames or self.settings.pinch_median_frames
        return float(np.min(vals[-window:]))

    def _double_click_anchor(self, anchor: np.ndarray, now: float) -> np.ndarray:
        """Place a quick second click exactly where the first one landed.

        Windows only treats two clicks as a double click when they arrive
        within its double-click time *and* within a few pixels of each other --
        four, on a default setup. Cursor jitter here is around thirteen, so
        left alone almost every attempted double click reaches applications as
        two unrelated single clicks. Reusing the first position removes the
        distance test from the equation; the timing still has to be the user's.

        The reuse only applies within a small radius, so deliberately clicking
        somewhere else nearby is not dragged back to the previous spot.
        """
        s = self.settings
        if not s.double_click_assist or self._click_pos is None:
            return anchor
        elapsed_ms = (now - self._last_click_at) * 1000.0
        if elapsed_ms < 0 or elapsed_ms > self._double_click_ms:
            return anchor
        if float(np.linalg.norm(anchor - self._click_pos)) > s.double_click_slack_px:
            return anchor
        return self._click_pos.copy()

    def _classify(self, hand: HandFrame) -> str:
        """Name the hand pose: open, fist, scroll, menu, or between.

        The two poses that matter are separated by a wide deadband rather than
        a single threshold, and the hand is always somewhere in that gap on its
        way from one to the other.  Nothing fires there -- the last state
        simply persists -- which is what stops a click landing halfway through
        the act of closing your hand.
        """
        s = self.settings
        fingers = hand.extended()
        curled = [not fingers[f] for f in ("middle", "ring", "pinky")]

        # Two deliberate poses, checked first: both sit inside the deadband by
        # curl alone, so they would otherwise be invisible.
        if fingers["index"] and fingers["middle"] and not fingers["ring"] \
                and not fingers["pinky"]:
            return "scroll"
        if fingers["thumb"] and not fingers["index"] and all(curled):
            return "keyboard"

        c = hand.curl
        if c > s.open_curl:
            return "open"
        if c < s.fist_curl:
            return "fist"
        return "between"

    def tracking_jitter(self) -> float | None:
        """Current per-frame noise, in screen pixels.

        Estimated from the *second* difference of consecutive raw predictions:
        steady motion cancels out of it, so what remains is noise even while
        the hand is moving.  For white noise the second difference has standard
        deviation sqrt(6) times the underlying one, hence the divisor.
        """
        if len(self.raw_history) < 5:
            return None
        p = np.stack(self.raw_history)
        second = p[2:] - 2.0 * p[1:-1] + p[:-2]
        return float(np.median(np.linalg.norm(second, axis=1)) / np.sqrt(6.0))

    def _release_everything(self) -> None:
        self.mouse.release_all()
        self._left_pinched = self._dragging = False
        self._right_pinched = False
        self._scrolling = False
        self._left_debounce.force(False)
        self._right_debounce.force(False)
        self._fist_debounce.force(False)
        self._scroll_debounce.force(False)
        self._menu_debounce.force(False)
        self._held_pose = "open"

    # -- per-frame update --------------------------------------------------

    def _update(self, hand: HandFrame) -> None:
        now = hand.stamp
        fingers = hand.extended()
        s = self.settings

        # Open palm toggles pause.  Three conditions, not one: every finger
        # straight, the fingers actually splayed apart, and the hand held still
        # throughout.  Straightness alone was too easy to hit by accident --
        # a hand caught mid-gesture, or one being re-acquired after a dropout,
        # can read as five straight fingers for long enough to fire, and
        # pausing itself unprompted is the most annoying thing this can do.
        palm_pose = (s.aim_mode == "finger" and all(fingers.values())
                     and hand.spread() > s.palm_spread_min)
        if palm_pose:
            if self._palm_since is None:
                self._palm_since = now
                self._palm_anchor = self._position.copy()
            elif (self._palm_anchor is not None
                  and np.linalg.norm(self._position - self._palm_anchor) > s.palm_still_px):
                self._palm_since = None      # drifting: not a deliberate hold
                self._palm_anchor = None
            elif ((now - self._palm_since) * 1000.0 > s.palm_pause_ms
                  and now > self._pause_cooldown):
                self.paused = not self.paused
                self._pause_cooldown = now + 1.5
                self._palm_since = None
                self._palm_anchor = None
                self._release_everything()
                print("Paused." if self.paused else "Resumed.")
        else:
            self._palm_since = None
            self._palm_anchor = None

        if self.paused:
            self.mode = "paused"
            return

        raw = np.array(self.model.predict(hand.aim_sample(s.aim_mode)), dtype=float)
        self.raw_history.append(raw)
        smooth = self.filter(now, raw)
        smooth[0] = min(max(smooth[0], 0.0), self.model.screen[0] - 1)
        smooth[1] = min(max(smooth[1], 0.0), self.model.screen[1] - 1)

        # Curve the cursor onto nearby clickable things -- but never mid-drag
        # (it would yank whatever is being dragged) and never mid-scroll.
        if (self.finder is not None and s.snap_enabled
                and not self._dragging and not self._scrolling):
            position, self.snap_target = snap(
                smooth, self.finder.targets(), s.snap_radius_px,
                sticky=self.snap_target, inset=s.snap_inset_px,
                unlock_radius=s.snap_unlock_px,
                sticky_margin=s.snap_sticky_margin_px)
            self.finder.note_cursor(position[0], position[1])
        else:
            position, self.snap_target = smooth, None

        self._position = position
        # History feeds the click anchor, so it has to hold where the cursor
        # actually went, snapping included.
        self.history.append((now, position.copy()))

        if s.aim_mode != "finger":
            self._act_on_pose(hand, now, smooth, position)
            return

        index_ratio = hand.pinch(lm.INDEX_TIP)
        middle_ratio = hand.pinch(lm.MIDDLE_TIP)

        # --- scroll -------------------------------------------------------
        two_fingers = (fingers["index"] and fingers["middle"]
                       and not fingers["ring"] and not fingers["pinky"])
        pose = hand.pose
        together = (np.linalg.norm(pose[lm.INDEX_TIP] - pose[lm.MIDDLE_TIP])
                    / max(hand.scale, 1e-6)) < 0.55
        if two_fingers and together and not self._left_pinched and index_ratio > s.pinch_off:
            if not self._scrolling:
                self._scrolling = True
                self._scroll_anchor = smooth[1]
            travel = self._scroll_anchor - smooth[1]
            if abs(travel) > 1.0:
                self.mouse.scroll(travel / self.model.screen[1] * s.scroll_gain)
                self._scroll_anchor = smooth[1]
            self.mode = "scroll"
            return
        self._scrolling = False

        # --- left click / drag --------------------------------------------
        left = self._left_debounce.update(
            self._pinched(index_ratio, self._left_pinched, s.pinch_on, s.pinch_off))
        if left and not self._left_pinched:
            anchor = self._anchor_position(now)
            self.mouse.move_to(anchor[0], anchor[1])
            self.mouse.left_down()
            self.log.clicks += 1
            self._left_since = now
            self._dragging = False
        elif not left and self._left_pinched:
            self.mouse.left_up()
            self._dragging = False
        self._left_pinched = left

        if left:
            # Freeze briefly, then let the cursor follow: a quick pinch is a
            # clean click, a held one becomes a drag.
            if not self._dragging and (now - self._left_since) * 1000.0 > s.drag_after_ms:
                self._dragging = True
            if self._dragging:
                self.mouse.move_to(position[0], position[1])
            self.mode = "drag" if self._dragging else "click"
            return

        # --- right click ---------------------------------------------------
        right = self._right_debounce.update(
            self._pinched(middle_ratio, self._right_pinched,
                          s.right_pinch_on, s.right_pinch_off))
        if right and not self._right_pinched and index_ratio > s.pinch_off:
            anchor = self._anchor_position(now)
            self.mouse.move_to(anchor[0], anchor[1])
            self.mouse.right_click()
            self.mode = "right click"
        self._right_pinched = right
        if right:
            return

        # --- move ----------------------------------------------------------
        if fingers["index"]:
            self.mouse.move_to(position[0], position[1])
            self.mode = "move"
        else:
            self.mode = "idle"

    def _act_on_pose(self, hand: HandFrame, now: float, smooth, position) -> None:
        """The palm aims; a pinch (or a fist) clicks."""
        s = self.settings
        pose = self._classify(hand)

        # "between" is the hand mid-transition. Hold whatever is already
        # happening and start nothing new.
        if pose == "between":
            pose = self._held_pose
        self._held_pose = pose

        if s.click_gesture == "pinch":
            # Pinching leaves the palm where it was, and the palm is what aims,
            # so unlike a fist this cannot tug the cursor as it closes.
            metric = hand.pinch_cm(lm.INDEX_TIP) is not None
            on = s.pinch_on_cm if metric else s.pinch_on
            off = s.pinch_off_cm if metric else s.pinch_off
            drag_off = s.drag_release_cm if metric else s.pinch_off * (
                s.drag_release_cm / s.pinch_off_cm)
            clear = s.right_click_clearance_cm if metric else (
                s.right_click_clearance_cm / s.pinch_off_cm * s.pinch_off)

            index_gap = self._pinch_signal(
                hand, lm.INDEX_TIP, self._pinch_history, self._left_pinched,
                min_frames=(s.drag_hold_frames if self._dragging
                            else s.pinch_median_frames))
            middle_gap = self._pinch_signal(hand, lm.MIDDLE_TIP,
                                            self._middle_history, self._right_pinched)

            # Right click is the thumb on the *middle* finger. The index must
            # be clearly further from the thumb than the middle is, which is
            # what keeps the two gestures apart: the old right-click pose put
            # the thumb alongside the index finger, a shape indistinguishable
            # from a pinch, so making it correctly fired a left click instead.
            right_squeeze = (middle_gap < (off if self._right_pinched else on)
                             and index_gap > middle_gap + clear)

            raw_index = hand.pinch_cm(lm.INDEX_TIP)
            self._signals = (raw_index if raw_index is not None else -1.0,
                             index_gap, middle_gap)

            release_at = drag_off if self._dragging else off
            squeeze = self._pinched(index_gap, self._left_pinched, on, release_at)

            # The escape hatch. `index_gap` uses the smallest of the recent
            # frames while pinched, which is what stops occlusion chopping one
            # click into several -- but it can also hold on forever when the
            # reading is merely unstable rather than pinched. If the gap has
            # not been clearly closed for a while, by the median so that spikes
            # cannot force it, the press is over whatever the min says.
            self._gap_trail.append((now, self._signals[0]))
            if self._left_pinched:
                # Only frames from inside this press count. Including the ones
                # before it would fill the window with open-hand readings and
                # end every press the moment it began.
                window = [g for stamp, g in self._gap_trail
                          if now - stamp <= s.pinch_slack_ms / 1000.0
                          and stamp >= self._left_since and g >= 0]
                if len(window) >= 5 and float(np.median(window)) > on:
                    squeeze = False
            if right_squeeze or self._right_pinched:
                squeeze = False
        else:
            squeeze = pose == "fist"
            right_squeeze = False     # fist mode has no middle-finger pinch

        # A pinch outranks the other poses. Thumb and finger meeting changes
        # the shape of the hand, and the shape is what those poses are read
        # from, so without this a click could be mistaken for one of them
        # halfway through being made.
        if squeeze or self._left_pinched:
            pose = "open"

        if pose == "scroll":
            fresh = self._scroll_debounce.update(True)
            self._fist_debounce.force(False)
            if fresh:
                if not self._scrolling:
                    self._scrolling = True
                    self._scroll_anchor = smooth[1]
                travel = self._scroll_anchor - smooth[1]
                if abs(travel) > 1.0:
                    self.mouse.scroll(travel / self.model.screen[1] * s.scroll_gain)
                    self._scroll_anchor = smooth[1]
                self.mode = "scroll"
                return
        else:
            self._scroll_debounce.update(False)
            self._scrolling = False

        if pose == "keyboard":
            if self._keyboard_debounce.update(True) and not self._keyboard_shown:
                self._keyboard_shown = True
                showing = keyboard.toggle()
                print("On-screen keyboard shown." if showing
                      else "On-screen keyboard hidden.")
            self.mode = "keyboard"
            return
        self._keyboard_debounce.update(False)
        self._keyboard_shown = False

        # Never while the left button is down. This branch returns early, so
        # letting it run mid-click would skip the release below and leave the
        # button stuck -- and a right click during a drag is not a thing anyone
        # is asking for anyway.
        if right_squeeze and not self._left_pinched:
            if self._menu_debounce.update(True) and not self._right_pinched:
                anchor = self._anchor_position(now)
                self.mouse.move_to(anchor[0], anchor[1])
                self.mouse.right_click()
                self._right_pinched = True
                self.log.right_clicks += 1
                self.mode = "right click"
            return
        if not self._menu_debounce.update(False):
            self._right_pinched = False

        if (squeeze and not self._left_pinched
                and (now - self._last_click_at) * 1000.0 < s.click_refractory_ms):
            squeeze = False          # too soon after the last one to be meant

        closed = self._fist_debounce.update(
            squeeze,
            release_frames=(s.drag_release_frames if self._dragging
                            else s.pinch_release_frames))
        if closed and not self._left_pinched:
            # Click where the hand was before it started closing.
            anchor = self._anchor_position(now)
            anchor = self._double_click_anchor(anchor, now)
            self.mouse.move_to(anchor[0], anchor[1])
            self.mouse.left_down()
            self.log.clicks += 1
            self._left_since = now
            self._click_pos = anchor.copy()
            self._press_origin = position.copy()
            self._drag_offset = np.zeros(2)
            self._drag_run = 0
            self._dragging = False
        elif not closed and self._left_pinched:
            self.mouse.left_up()
            # Only a real click starts a double click; a drag ends somewhere
            # else entirely and must not anchor the next one.
            self._last_click_at = -1e9 if self._dragging else now
            self._dragging = False
        self._left_pinched = closed

        if closed:
            travelled = float(np.linalg.norm(position - self._press_origin))
            held_ms = (now - self._left_since) * 1000.0
            far = (travelled > s.drag_start_px
                   or (held_ms > s.drag_after_ms and travelled > s.drag_creep_px))
            # The cursor can jump hundreds of pixels in one frame, so a single
            # frame past the threshold means nothing. Only travel that stays
            # travelled counts.
            self._drag_run = self._drag_run + 1 if far else 0
            if not self._dragging and self._drag_run >= s.drag_confirm_frames:
                self._dragging = True
                self.log.drags += 1
                # Carry on from where the cursor actually is rather than
                # snapping to the hand. The gap between them is the distance
                # the hand travelled while the cursor was held still for the
                # click, and jumping it would fling whatever is being dragged.
                self._drag_offset = self._click_pos - position
            if self._dragging:
                target = position + self._drag_offset
                target[0] = min(max(target[0], 0.0), self.model.screen[0] - 1)
                target[1] = min(max(target[1], 0.0), self.model.screen[1] - 1)
                self.mouse.move_to(target[0], target[1])
            self.mode = "drag" if self._dragging else "click"
            return

        if s.click_gesture == "pinch" or pose == "open":
            self.mouse.move_to(position[0], position[1])
            self.mode = "move"
        else:
            self.mode = "idle"

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        print(_HELP.get(self.settings.aim_mode, _HELP["finger"]))
        print("Press Q or Esc in the preview window (or Ctrl+C here) to quit.")
        window = "Gesture Control"
        if self.settings.show_preview:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 640, 360)

        if self.settings.snap_enabled:
            self.finder = TargetFinder(self.model.screen,
                                       self.settings.snap_refresh_s).start()
            if not self.finder.available:
                print(self.finder.unavailable_reason)
                self.finder = None
            else:
                print("Snapping to on-screen controls is on (press S to toggle).")

        self._warn_about_frame_rate()

        try:
            while self.running:
                self.tracker.raise_if_failed()
                hand = self.tracker.latest()
                now = time.monotonic()

                if hand is not None and hand.stamp != self._last_stamp:
                    self._last_stamp = hand.stamp
                    self._last_seen = now
                    self._update(hand)
                    raw, used, middle = self._signals
                    self.log.add(Frame(
                        t=hand.stamp, hand=True, index_raw=raw, index_used=used,
                        middle_raw=middle, mode=self.mode,
                        down=self.mouse.left_is_down, dragging=self._dragging,
                        x=self._position[0], y=self._position[1]))
                elif hand is None and (now - self._last_seen) * 1000.0 > self.settings.lost_hand_grace_ms:
                    # Never leave a button stuck down because the hand left frame.
                    if self.mouse.left_is_down or self._scrolling:
                        self._release_everything()
                    self.filter.reset()
                    self.mode = "no hand"

                if self.settings.show_preview:
                    if not self._draw_preview(window, hand):
                        break
                else:
                    time.sleep(0.004)
        except KeyboardInterrupt:
            print()
        finally:
            self._release_everything()
            self._write_session()
            if self.finder is not None:
                self.finder.stop()
            if self.settings.show_preview:
                cv2.destroyAllWindows()

    def _write_session(self) -> None:
        """Dump the recording and say what it shows.

        Printed on the way out rather than tucked in a file nobody opens: the
        point is to turn "clicking feels wrong" into a set of numbers that
        names which part is wrong.
        """
        from .config import ROOT

        try:
            path = ROOT / "session_log.csv"
            rows = self.log.save(path)
        except OSError:
            rows = 0
        if not rows:
            return
        print()
        print("-" * 62)
        if self.rejected_frames:
            print(f"{self.rejected_frames} frame(s) discarded as implausible "
                  f"(the hand model had collapsed)")
        for line in self.log.summary(self.settings):
            print(line)
        print("-" * 62)
        print(f"Frame-by-frame recording of the last few minutes: {path.name}")

    def _warn_about_frame_rate(self) -> None:
        """The default camera resolution trades frame rate for precision.

        On a slower machine that trade can go the wrong way, so measure it once
        the tracker has settled and say so plainly rather than leaving the user
        wondering why the cursor feels sticky.
        """
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and self.tracker.fps <= 0.0:
            time.sleep(0.1)
        fps = self.tracker.fps
        if 0.0 < fps < self.settings.min_useful_fps:
            # Deliberately not naming a resolution to switch to: which ones are
            # fast is a property of the camera, not something to guess at.
            # `check` measures them.
            print(f"Tracking is running at only {fps:.0f} fps, which will feel "
                  f"sluggish. Run `run.bat check` to see which resolutions your "
                  f"camera delivers quickly -- a smaller one is not always faster.")

    # -- preview -----------------------------------------------------------

    def _pinch_fraction(self, hand) -> float:
        """How closed the pinch is, 0 (open) to 1 (clicking).

        Drives the glow on the stones, and nothing else. Deliberately reads the
        same numbers the click does rather than sampling its state, so the
        preview cannot end up holding a reference into the click machinery.
        """
        s = self.settings
        cm = hand.pinch_cm(lm.INDEX_TIP)
        if cm is None:
            value, on, off = hand.pinch(lm.INDEX_TIP), s.pinch_on, s.pinch_off
        else:
            value, on, off = cm, s.pinch_on_cm, s.pinch_off_cm
        if off <= on:
            return 0.0
        return float(min(max((off - value) / (off - on), 0.0), 1.0))



    def _draw_preview(self, window: str, hand: HandFrame | None) -> bool:
        frame = self.tracker.latest_preview()
        if frame is None:
            return True
        frame = cv2.flip(frame, 1)                 # mirror, so it reads like a mirror
        h, w = frame.shape[:2]

        if hand is not None:
            pts = hand.norm.copy()
            pts[:, 0] = 1.0 - pts[:, 0]
            pts = pts * np.array([w, h])
            if self.settings.gauntlet:
                gauntlet.draw(frame, pts, hand.handedness,
                              charge=self._pinch_fraction(hand))
            else:
                ipts = pts.astype(int)
                for a, b in lm.CONNECTIONS:
                    cv2.line(frame, tuple(ipts[a]), tuple(ipts[b]), (120, 90, 40), 2)
                for i, p in enumerate(ipts):
                    hot = i in (lm.THUMB_TIP, lm.INDEX_TIP, lm.MIDDLE_TIP)
                    cv2.circle(frame, tuple(p), 6 if hot else 3,
                               (255, 163, 77) if hot else (140, 160, 190), -1)

        # In direct mode the preview *is* the map: this window scaled up to the
        # monitor is exactly where the cursor goes. Drawing the cursor on it
        # makes that correspondence visible rather than something to take on
        # trust, and gives the edges of your reach a visible boundary.
        if self.settings.aim_mode == "direct" and not self.paused:
            m = self.settings.direct_margin
            if m > 0:
                # Dim the overscan border: everything outside the bright
                # rectangle is already pinned to the edge of the screen, so
                # the rectangle is the part of the view that still has
                # anywhere left to go.
                x0, y0 = int(w * m), int(h * m)
                x1, y1 = int(w * (1 - m)), int(h * (1 - m))
                shade = frame.copy()
                for box in ((0, 0, w, y0), (0, y1, w, h),
                            (0, y0, x0, y1), (x1, y0, w, y1)):
                    cv2.rectangle(shade, box[:2], box[2:], (12, 14, 18), -1)
                cv2.addWeighted(shade, 0.55, frame, 0.45, 0, frame)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (110, 125, 145), 1)
            px = int(self._position[0] / max(self.model.screen[0] - 1, 1) * w)
            py = int(self._position[1] / max(self.model.screen[1] - 1, 1) * h)
            cv2.line(frame, (px - 14, py), (px + 14, py), (255, 163, 77), 2)
            cv2.line(frame, (px, py - 14), (px, py + 14), (255, 163, 77), 2)
            cv2.circle(frame, (px, py), 20, (255, 163, 77), 1)

        colour = (80, 200, 120) if not self.paused else (60, 160, 250)
        cv2.rectangle(frame, (0, 0), (w, 34), (18, 20, 26), -1)
        cv2.putText(frame, f"{self.mode.upper():<11} {self.tracker.fps:4.0f} fps",
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)

        # Jitter is the thing the user can actually do something about (light,
        # distance), so show it, colour-coded, rather than burying it.
        jitter = self.tracking_jitter()
        if jitter is not None:
            jcolour = ((90, 200, 120) if jitter < 8 else
                       (60, 190, 235) if jitter < 18 else (70, 110, 250))
            cv2.putText(frame, f"jitter {jitter:4.0f}px", (150, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, jcolour, 1, cv2.LINE_AA)

        # In palm mode the open/closed number is the click signal, so show it.
        if self.settings.aim_mode != "finger" and hand is not None:
            if self.settings.click_gesture == "pinch":
                cm = hand.pinch_cm(lm.INDEX_TIP)
                if cm is None:
                    c, on, off = (hand.pinch(lm.INDEX_TIP),
                                  self.settings.pinch_on, self.settings.pinch_off)
                    label = f"pinch {c:.2f}"
                else:
                    c, on, off = cm, self.settings.pinch_on_cm, self.settings.pinch_off_cm
                    label = f"pinch {c:4.1f}cm"
                ccolour = ((255, 163, 77) if c < on else
                           (150, 160, 175) if c < off else (90, 200, 120))
            else:
                c = hand.curl
                ccolour = ((90, 200, 120) if c > self.settings.open_curl else
                           (255, 163, 77) if c < self.settings.fist_curl else
                           (150, 160, 175))
                label = f"hand {c:+.2f}"
            cv2.putText(frame, label, (270, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ccolour, 1, cv2.LINE_AA)

        if self.snap_target is not None:
            cv2.putText(frame, f"-> {self.snap_target.kind}", (380, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 163, 77), 1, cv2.LINE_AA)

        cv2.putText(frame, f"({self._position[0]:.0f}, {self._position[1]:.0f})",
                    (w - 150, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 160, 175), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, h - 26), (w, h), (18, 20, 26), -1)
        if self.finder is not None and self.settings.snap_enabled:
            snap_state = f"on, {len(self.finder.targets())} targets"
        else:
            snap_state = "off"
        cv2.putText(frame, f"Q/Esc: quit   P: pause   K: keyboard   "
                           f"S: snapping ({snap_state})",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (120, 132, 150), 1, cv2.LINE_AA)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord("p"):
            self.paused = not self.paused
            self._release_everything()
        if key == ord("k"):
            print("On-screen keyboard shown." if keyboard.toggle()
                  else "On-screen keyboard hidden.")
        if key == ord("s"):
            self.settings.snap_enabled = not self.settings.snap_enabled
            self.snap_target = None
            print("Snapping on." if self.settings.snap_enabled else "Snapping off.")
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True
