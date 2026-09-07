"""Paths and user-tunable settings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
CALIBRATION_PATH = ROOT / "calibration.json"
SETTINGS_PATH = ROOT / "settings.json"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class Settings:
    # --- camera -------------------------------------------------------
    camera_index: int = 0
    # More pixels on the hand means less landmark jitter, which is the dominant
    # source of pointing error.  Drop to 1280x720 if the frame rate suffers --
    # the app warns when it does.
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fps: int = 60
    min_useful_fps: float = 20.0

    # --- tracking -----------------------------------------------------
    detection_confidence: float = 0.5
    # Presence and tracking are deliberately looser than detection: they decide
    # whether to keep following a hand already being tracked, and a hand
    # pointing at the camera is a hard, self-occluded view that scores low.
    # Giving up on it means the cursor stalls and then jumps when the hand is
    # re-acquired, which reads as the app losing your hand.
    presence_confidence: float = 0.3
    tracking_confidence: float = 0.3

    # --- calibration --------------------------------------------------
    grid_cols: int = 4
    grid_rows: int = 3
    margin_frac: float = 0.09        # dot inset from the screen edge
    samples_per_point: int = 18      # frames averaged per calibration dot
    settle_frames: int = 6           # frames discarded right after the pinch
    # Holding the finger still also captures a dot, so calibration does not
    # depend on the pinch thresholds it is partly there to establish.
    dwell_ms: float = 1200.0         # hold this long on a dot to capture it
    dwell_radius: float = 0.012      # "still" means within this much of the frame

    # --- pointer smoothing (One Euro filter) --------------------------
    filter_min_cutoff: float = 1.6   # lower = smoother but laggier
    filter_beta: float = 0.035       # higher = snappier on fast motion
    filter_dcutoff: float = 1.0

    # --- how you aim --------------------------------------------------
    # "direct": the camera view fills the screen; your hand's place in the
    #   frame is the cursor's place on the monitor. Nothing to calibrate.
    # "palm": a fitted mapping from the palm, calibrated against dots.
    # "finger": aim with the index fingertip, pinch thumb-to-index to click.
    #
    # Palm is the default because every part of it is easier to see. An open
    # hand shows the camera all of itself, where a pointing finger aimed at the
    # lens is foreshortened and half hidden behind itself. A fist is four
    # fingers agreeing, where a pinch is one small distance between two
    # landmarks. And the palm barely moves as the fingers close, so clicking
    # does not tug the aim off target.
    aim_mode: str = "direct"

    # Mirrored, so moving your hand right moves the cursor right.
    mirror: bool = True
    # Overscan: this much of each edge of the frame is treated as already off
    # the screen, and the middle is stretched to fill it. At 0.1 the outer
    # tenth on every side pins the cursor to that edge, so the corners are
    # reachable without stretching to the very limits of the camera's view --
    # which is both a shorter reach and a kinder place to be tracked, since
    # landmark quality falls off at the frame edges.
    #
    # The cost is a 1/(1 - 2*margin) magnification: 1.25x at 0.1, applied to
    # hand movement and to tracking jitter alike. 0 maps the frame one to one.
    direct_margin: float = 0.20

    # Open above this, fist below the other; the wide gap between them is a
    # deadband the hand passes through on the way, in which nothing fires.
    # What closes a click: "pinch" (thumb to index) or "fist" (whole hand).
    click_gesture: str = "pinch"

    open_curl: float = 0.35
    fist_curl: float = -0.15
    fist_press_frames: int = 2
    pose_debounce_frames: int = 5    # for the rarer, more deliberate poses

    # --- gestures -----------------------------------------------------
    # Click thresholds in centimetres, which is how a hand is actually
    # imagined. The world landmarks give the palm's real size, so the gap
    # between fingertips converts to real units without calibration.
    # Fingertips in contact are about a centimetre apart, pad to pad.
    # Set from measured sessions rather than from what a pinch "should" be.
    # A tight threshold looks right on paper and fails in use: at 2.1cm, 167
    # frames of a genuinely pinched hand read as open, against 6 open frames
    # misread the other way. The readings are noisy enough that the honest
    # trade is a generous threshold -- 4cm misses 38 instead of 167, and still
    # sits below the fifth percentile of a real open hand (5.1cm).
    pinch_on_cm: float = 4.0         # closer than this clicks
    pinch_off_cm: float = 5.2        # must open past this to release
    # A frame whose finger bones have changed length by more than this much is
    # one where the tracker has lost the hand, not one where the hand moved.
    # Measured across a real session, the gap read anywhere from 0.7cm with the
    # hand plainly open to 8.2cm with the fingers plainly together; those
    # frames are not worth thresholding, they are worth discarding.
    #
    # Deliberately loose. A tight bound also rejects ordinary motion -- the
    # estimated bone lengths breathe a little as a finger bends -- and a gate
    # that throws away good frames is worse than the outliers it removes. This
    # is set to catch only frames where the hand model has plainly collapsed.
    bone_tolerance: float = 0.60
    # However bad a frame looks, no more than this many in a row may be
    # discarded, so the check can never freeze the reading indefinitely.
    bone_reject_limit: int = 3

    # Used only when world landmarks are unavailable, in palm lengths.
    pinch_on: float = 0.18
    pinch_off: float = 0.36
    # Finger aiming keeps its own thumb-to-middle thresholds, in palm lengths.
    right_pinch_on: float = 0.18
    right_pinch_off: float = 0.36

    # Right click is the thumb on the middle finger. For it to count, the
    # index finger has to be this much further from the thumb than the middle
    # is -- which is what tells the two gestures apart, rather than hoping a
    # hand shape will read cleanly.
    right_click_clearance_cm: float = 2.0

    # Windows only counts two clicks as a double click if they land within a
    # few pixels of each other -- four, typically. Hand tracking cannot repeat
    # a position that precisely, so a second click arriving quickly is placed
    # exactly where the first one went.
    double_click_assist: bool = True
    double_click_slack_px: float = 90.0
    pinch_press_frames: int = 2      # frames a pinch must survive to count
    # Fingertips in contact occlude each other, and MediaPipe's estimate of
    # where they are gets jumpy exactly then -- the measured gap can spike wide
    # for a frame or two in the middle of a perfectly steady pinch. Taking the
    # median of the last few frames throws those spikes away before they reach
    # the threshold, rather than leaving the debounce to mop up afterwards.
    pinch_median_frames: int = 3
    # One is enough: the min-filter above already requires every recent frame
    # to agree before a release, which is the same guarantee a debounce gives.
    # Stacking both delays the button-up enough that a quick second click
    # arrives before the first has finished letting go.
    pinch_release_frames: int = 1
    # ...but "every recent frame must agree" can latch. Measured on a real
    # session, a hand that had plainly let go read as open in 20 frames out of
    # 98 and never three in a row, so the button stayed down for 3.7 seconds.
    # This is the escape hatch: if the gap has not been *clearly* pinched for
    # this long -- judged by the median, so spikes do not trigger it -- the
    # press ends regardless. A firm pinch reads about 1.2cm throughout and is
    # nowhere near it, so genuine long drags survive.
    # 600, not 300: at 300 a five-frame tracking dropout is enough of the
    # window to move the median, so this hatch would undo the longer drag
    # window below and split a drag in two anyway. Measured across two
    # sessions, 600 tells a dropout from a hand that has genuinely let go.
    pinch_slack_ms: float = 600.0
    # Nothing may click again this soon after a click, which stops one pinch
    # from stuttering into several. Kept well under the double-click time so
    # deliberate double clicks still get through.
    click_refractory_ms: float = 130.0
    click_anchor_ms: float = 110.0   # click where the finger was this long ago
    click_anchor_window_ms: float = 300.0  # most history a click may average
    click_anchor_tolerance_frac: float = 0.02  # ...but only while the hand held still
    # A pinch becomes a drag by *moving*, not by being held. Holding still is
    # how someone clicks carefully, and how they double click, so time alone
    # must not turn a click into a drag.
    # Set from a real session: the cursor wanders about 42px in 200ms and 92px
    # in 400ms just from tracking noise and a hand held in the air, so a 26px
    # threshold with a 5px creep made essentially every held click a drag.
    drag_start_px: float = 110.0     # move this far while pinched and it drags
    # ...and stay moved for this many frames. The cursor can teleport: measured
    # over a real session it jumped 270px in a single frame at the 99th
    # percentile and 889px at worst, so any distance threshold is crossed now
    # and then by one bad frame alone. Requiring the travel to persist is what
    # separates a real drag from a glitch; on that session 110px held for three
    # frames picked out exactly the three intended drags and nothing else.
    drag_confirm_frames: int = 3
    drag_after_ms: float = 700.0     # or hold this long, having moved a fair way
    drag_creep_px: float = 50.0
    # Ending a drag takes a more deliberate opening than ending a click: a
    # dropped file is much more annoying than a click that lingers a frame.
    drag_release_cm: float = 6.2
    # None extra: the eight-frame window below is already ample evidence, and
    # stacking a debounce on it pushes letting go of a drag past a third of a
    # second, which is long enough to feel like the app is arguing.
    drag_release_frames: int = 1
    # How many frames must all agree the pinch is over, while dragging. In a
    # measured session a steady pinch read wide for five frames running and
    # then went straight back to 0.5cm -- the fingers never parted, the
    # tracker just lost them -- and a three-frame window let that end a drag.
    # Eight covers it with margin. Plain clicks keep the shorter window, since
    # there a slow button-up costs double clicks and nothing is being carried.
    drag_hold_frames: int = 8
    scroll_gain: float = 9.0         # wheel notches per screen-height of travel
    palm_pause_ms: float = 900.0     # open palm held this long toggles pause
    palm_spread_min: float = 0.30    # ...and the fingers must be splayed this far
    palm_still_px: float = 90.0      # ...and the hand must not wander this much

    # --- snapping to clickable UI -------------------------------------
    snap_enabled: bool = True
    # How far a control reaches out for the cursor. This is the number you feel
    # most directly: too large and the cursor is grabbed from across a gap you
    # were only passing through, which reads as the snap acting on its own.
    # Roughly a fingernail's width on screen -- close enough that reaching for
    # the control is what triggers it. This is now smaller than the hand's own
    # jitter, so the snap corrects far less than it used to: on a 60x30 button
    # with 12px of shake it recovers 69% -> 93% of aims, where the original
    # 85px basin recovered 100%. Deliberate -- reach was what felt wrong.
    snap_radius_px: float = 11.0
    snap_inset_px: float = 4.0       # land this far inside the control, not on its edge
    # Letting go of a lock takes more than acquiring it. Without the gap the
    # cursor sits on the boundary and rattles in and out. Kept below the pitch
    # of a typical toolbar (~60px) so the lock always expires in the gap
    # between two controls rather than carrying across it -- so it is set well
    # above the acquire radius on purpose. Grabbing late and letting go late is
    # what gives a small basin without a twitchy one: the cursor commits only
    # when you have really arrived, then stays committed while your hand shakes.
    snap_unlock_px: float = 50.0
    # How much nearer a rival control must be before the lock moves to it.
    # With a release radius this wide the lock outlives the gap between two
    # controls, so this number arbitrates the handoff and both failure modes
    # ride on it. Measured on a 60px-pitch row, worst of 8 jitter seeds at 6px
    # of shake -- (margin: handoff lag, swaps per 300 frames):
    #   14: 0px, 45   19: 0px, 9   24: 4px, 2   32: 12px, 0   40: 20px, 0
    # Below 24 the lock flickers between neighbours; above it the cursor clings
    # to the button you just left while your hand is already well inside the
    # next one, which is what "it skips over buttons" felt like.
    snap_sticky_margin_px: float = 24.0
    snap_refresh_s: float = 0.6      # how often the on-screen controls are re-read

    # --- behaviour ----------------------------------------------------
    show_preview: bool = True
    # Draw the nano gauntlet over your hand in the preview instead of the bare
    # tracking skeleton. Cosmetic only -- the control loop never reads it.
    gauntlet: bool = True
    max_hands: int = 1
    lost_hand_grace_ms: float = 500.0

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "Settings":
        s = cls()
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return s
            known = {f.name for f in fields(cls)}
            for key, value in raw.items():
                if key in known:
                    setattr(s, key, value)
        return s

    def save(self, path: Path = SETTINGS_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def update_file(path: Path = SETTINGS_PATH, **values) -> None:
        """Merge a few keys into settings.json, leaving the rest alone.

        Calibration persists the pinch thresholds it measured, but the live
        Settings object may also hold one-off command line overrides
        (--camera, --grid) that have no business being written to disk.
        """
        current: dict = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    current = loaded
            except (OSError, json.JSONDecodeError):
                current = {}
        current.update(values)
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")
