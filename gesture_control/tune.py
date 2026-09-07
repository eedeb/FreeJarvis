"""Measure this particular hand and set the gesture thresholds from it.

Default thresholds are a guess about an average hand at an average distance in
average light.  When clicks fire by themselves, or refuse to fire, the honest
fix is not to nudge a constant but to record what the hand actually does and
put the threshold where the two states genuinely separate.

The routine watches two phases -- pointing without pinching, then pinching --
and reports, for this hand:

* how far apart the pinched and unpinched states are, and therefore how often
  a click can be expected to misfire at any given threshold;
* the same figure computed the old 2D way, as a check that measuring pose in
  3D was worth it;
* how often tracking drops out, how often the finger reads as curled while
  pointing, and how often the pause gesture would have fired by accident.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import landmarks as lm
from .config import Settings
from .tracker import HandFrame, HandTracker

WINDOW = "Gesture Control - Tuning"
PHASE_SECONDS = 7.0


class _Samples:
    """Per-frame measurements for one phase."""

    def __init__(self) -> None:
        self.curl: list[float] = []
        self.pinch3d: list[float] = []
        self.pinch2d: list[float] = []
        self.middle3d: list[float] = []
        self.index_extended: list[bool] = []
        self.palm_pose: list[bool] = []
        self.frames = 0
        self.detected = 0

    def add(self, hand: HandFrame | None, settings: Settings) -> None:
        self.frames += 1
        if hand is None:
            return
        self.detected += 1
        self.curl.append(hand.curl)
        self.pinch3d.append(hand.pinch(lm.INDEX_TIP))
        self.pinch2d.append(lm.pinch_ratio(hand.metric, lm.INDEX_TIP))
        self.middle3d.append(hand.pinch(lm.MIDDLE_TIP))
        fingers = hand.extended()
        self.index_extended.append(fingers["index"])
        self.palm_pose.append(all(fingers.values())
                              and hand.spread() > settings.palm_spread_min)

    @property
    def dropout(self) -> float:
        return 1.0 - self.detected / max(self.frames, 1)


def _separation(open_vals, closed_vals) -> float:
    """How many standard deviations apart the two states are (d-prime).

    Above about 3 the states barely overlap and clicking is reliable; below 2
    they blur into each other and no threshold can work well.
    """
    if len(open_vals) < 5 or len(closed_vals) < 5:
        return 0.0
    o, c = np.asarray(open_vals), np.asarray(closed_vals)
    spread = np.sqrt((o.std() ** 2 + c.std() ** 2) / 2.0)
    if spread < 1e-9:
        return float("inf")
    return float(abs(o.mean() - c.mean()) / spread)


def _thresholds(open_vals, closed_vals) -> tuple[float, float] | None:
    """Press and release thresholds, placed inside the gap between the states.

    Both are pinned into the corridor between the pinched values and the open
    ones, which is the safeguard that matters: the release threshold *must*
    sit below the ratios seen while merely pointing.  If it does not, the hand
    can enter the pinched state and have no way back out -- the button latches
    down, and in calibration every dot fills itself without being aimed at.

    An earlier version chose the threshold by minimising a weighted error rate.
    That optimises the right quantity and still produced an unusable pair,
    because nothing in the objective knew that releasing has to be possible.
    Returns None when the two states overlap too much for any pair to work.
    """
    o, c = np.asarray(open_vals), np.asarray(closed_vals)
    closed_hi = float(np.percentile(c, 90))   # a firm pinch, at its loosest
    open_lo = float(np.percentile(o, 10))     # pointing, at its tightest
    gap = open_lo - closed_hi
    if gap < 0.10:
        return None

    on = closed_hi + 0.35 * gap
    off = closed_hi + 0.70 * gap              # below open_lo by construction
    on = float(np.clip(on, 0.12, 0.70))
    off = float(np.clip(max(off, on + 0.05), 0.16, 0.80))
    if off >= open_lo:                        # clamping must not undo the point
        return None
    return on, off


def _error_rates(open_vals, closed_vals, on: float) -> tuple[float, float]:
    o, c = np.asarray(open_vals), np.asarray(closed_vals)
    return float((o < on).mean()), float((c >= on).mean())


def _phase(tracker: HandTracker, settings: Settings, title: str,
           subtitle: str) -> _Samples | None:
    """Collect one phase; returns None if the user aborted."""
    samples = _Samples()
    start = time.monotonic()
    # Count one sample per camera frame, not per pass of this loop. The loop
    # spins far faster than the camera, and a frame in which no hand was found
    # has no timestamp to deduplicate on -- counting those per iteration would
    # report a dropout rate made of nothing but idle spins.
    last_frame = tracker.frames_processed
    while True:
        remaining = PHASE_SECONDS - (time.monotonic() - start)
        if remaining <= 0:
            return samples
        tracker.raise_if_failed()
        hand = tracker.latest()
        processed = tracker.frames_processed
        if processed != last_frame:
            last_frame = processed
            samples.add(hand, settings)

        frame = tracker.latest_preview()
        if frame is not None:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 96), (18, 20, 26), -1)
            cv2.putText(frame, title, (14, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (235, 240, 245), 2, cv2.LINE_AA)
            cv2.putText(frame, subtitle, (14, 68), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (150, 165, 185), 1, cv2.LINE_AA)
            colour = (90, 200, 120) if hand is not None else (70, 110, 250)
            cv2.putText(frame, f"{remaining:.0f}", (w - 60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2, cv2.LINE_AA)
            if hand is None:
                cv2.putText(frame, "hand not visible", (14, h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 110, 250), 1,
                            cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            return None


def run_tuning(tracker: HandTracker, settings: Settings) -> bool:
    """Run both phases, report, and save the thresholds.  True if it completed."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 900, 520)
    by_fist = settings.click_gesture == "fist"
    try:
        print("\nMeasuring your hand. Two short phases, 7 seconds each.\n")
        time.sleep(1.0)
        # Each phase asks for one steady pose held throughout. An earlier
        # version asked for repeated pinches in the second phase, which mixed
        # open frames into the closed sample and dragged the threshold up into
        # the range of an ordinary open hand.
        if by_fist:
            first = _phase(tracker, settings,
                           "1/2  Hold your hand OPEN, palm to the screen.",
                           "Move it around a little, but keep it open throughout.")
            if first is None:
                return False
            second = _phase(tracker, settings,
                            "2/2  Now hold a CLOSED FIST.",
                            "Keep it closed the whole time. Move it around a little.")
        else:
            first = _phase(tracker, settings,
                           "1/2  Hold your hand up, thumb and finger APART.",
                           "Move around as you normally would while using it.")
            if first is None:
                return False
            second = _phase(tracker, settings,
                            "2/2  Now PINCH and hold it closed.",
                            "Keep thumb and finger together the whole time.")
        if second is None:
            return False
    finally:
        cv2.destroyWindow(WINDOW)

    if by_fist:
        return _report_palm(first, second, settings)
    return _report_and_save(first, second, settings)


def _report_palm(open_hand: _Samples, fist: _Samples, settings: Settings) -> bool:
    """Report how cleanly this hand's open and closed states separate."""
    print("\n" + "=" * 68)
    print("Tracking")
    print("=" * 68)
    print(f"  hand detected in {1 - open_hand.dropout:.0%} of frames, open")
    print(f"  hand detected in {1 - fist.dropout:.0%} of frames, in a fist")
    if max(open_hand.dropout, fist.dropout) > 0.1:
        print("  ! Tracking is dropping out. More light on your hand will help")
        print("    more than any setting here.")

    if len(open_hand.curl) < 30 or len(fist.curl) < 30:
        print("\nNot enough of your hand was seen to measure. Try again in "
              "better light.")
        return False

    print("\n" + "=" * 68)
    print("Clicking (open hand versus fist)")
    print("=" * 68)
    o, f = np.asarray(open_hand.curl), np.asarray(fist.curl)
    print(f"  open : median {np.median(o):+.2f}, 10th pct {np.percentile(o, 10):+.2f}, "
          f"min {o.min():+.2f}")
    print(f"  fist : median {np.median(f):+.2f}, 90th pct {np.percentile(f, 90):+.2f}, "
          f"max {f.max():+.2f}")
    print(f"  separation: {_separation(o, f):.1f} sigma")

    _write_report(open_hand, fist)
    print(f"  full measurements written to tune_report.json")

    gap_low, gap_high = float(np.percentile(f, 90)), float(np.percentile(o, 10))
    print(f"\n  current thresholds: fist below {settings.fist_curl:+.2f}, "
          f"open above {settings.open_curl:+.2f}")
    if gap_high - gap_low < 0.25:
        print("  ! Your open hand and your fist measure too much alike. That is")
        print("    unusual and points at the hand being small in frame or dim.")
        return False

    fist_curl = gap_low + 0.3 * (gap_high - gap_low)
    open_curl = gap_low + 0.7 * (gap_high - gap_low)
    if not (settings.fist_curl < gap_high and settings.open_curl > gap_low):
        print("  ! The defaults do not sit in your hand's gap. Adjusting.")
    print(f"  suggested for you : fist below {fist_curl:+.2f}, "
          f"open above {open_curl:+.2f}")
    Settings.update_file(fist_curl=fist_curl, open_curl=open_curl)
    print("\nSaved to settings.json. Calibrate next: calibrate.bat")
    return True


def _report_and_save(pointing: _Samples, closed: _Samples,
                     settings: Settings) -> bool:
    print("\n" + "=" * 68)
    print("Tracking")
    print("=" * 68)
    print(f"  hand detected in {1 - pointing.dropout:.0%} of frames while pointing")
    if pointing.dropout > 0.1:
        print("  ! Tracking is dropping out. More light on your hand, or moving")
        print("    a little closer to the camera, will help more than any setting.")

    if pointing.index_extended:
        stalled = 1.0 - float(np.mean(pointing.index_extended))
        print(f"  pointing finger read as curled in {stalled:.0%} of frames")
        if stalled > 0.1:
            print("  ! That is the cursor freezing. Point a little more across the")
            print("    camera rather than straight down the lens.")

    if pointing.palm_pose:
        accidental = float(np.mean(pointing.palm_pose))
        print(f"  open-palm pause pose seen in {accidental:.0%} of pointing frames")
        if accidental > 0.02:
            print("  ! Risk of pausing by accident; raising palm_spread_min helps.")

    if len(pointing.pinch3d) < 30 or len(closed.pinch3d) < 30:
        print("\nNot enough of your hand was seen to tune clicking. "
              "Try again in better light.")
        return False

    print("\n" + "=" * 68)
    print("Clicking")
    print("=" * 68)
    sep3d = _separation(pointing.pinch3d, closed.pinch3d)
    sep2d = _separation(pointing.pinch2d, closed.pinch2d)
    print(f"  pinched vs not, measured in 3D: {sep3d:.1f} sigma apart")
    print(f"  the same thing measured in 2D:  {sep2d:.1f} sigma apart")
    if sep2d > 0:
        print(f"  ({sep3d / sep2d:.1f}x better in 3D)" if sep3d > sep2d
              else "  (2D was not the problem here)")

    print(f"\n  pointing : median {np.median(pointing.pinch3d):.2f}, "
          f"10th pct {np.percentile(pointing.pinch3d, 10):.2f}, "
          f"min {np.min(pointing.pinch3d):.2f}")
    print(f"  pinched  : median {np.median(closed.pinch3d):.2f}, "
          f"90th pct {np.percentile(closed.pinch3d, 90):.2f}, "
          f"max {np.max(closed.pinch3d):.2f}")

    report_path = _write_report(pointing, closed)
    print(f"\n  full measurements written to {report_path.name}")

    chosen = _thresholds(pointing.pinch3d, closed.pinch3d)
    if chosen is None:
        print("\n  ! Your pinched and pointing hands measure too much alike for a")
        print("    threshold to separate them, so nothing has been saved -- the")
        print("    existing settings are left as they were. Clicking cannot be")
        print("    made reliable by tuning until the two states pull apart.")
        print("    More light on the hand, or filling more of the frame, is what")
        print("    moves this. Re-run tune.bat after changing either.")
        return False

    on, off = chosen
    false_rate, miss_rate = _error_rates(pointing.pinch3d, closed.pinch3d, on)
    print(f"  threshold: click below {on:.2f}, release above {off:.2f}")
    print(f"  expected : {false_rate:.1%} of pointing frames misread as a click, "
          f"{miss_rate:.1%} of pinches missed")
    debounce = settings.pinch_press_frames
    if false_rate > 0:
        print(f"             a real misfire needs {debounce} bad frames in a row: "
              f"about {false_rate ** debounce:.2%} of the time")

    Settings.update_file(pinch_on=on, pinch_off=off,
                         right_pinch_on=on, right_pinch_off=off)
    print("\nSaved to settings.json. Calibrate next: calibrate.bat")
    return True


def _write_report(pointing: _Samples, closed: _Samples):
    """Dump the raw distributions so a bad result can be diagnosed, not guessed at."""
    import json

    from .config import ROOT

    def describe(values):
        if not values:
            return None
        a = np.asarray(values, dtype=float)
        return {
            "n": int(a.size),
            "min": round(float(a.min()), 4),
            "p10": round(float(np.percentile(a, 10)), 4),
            "median": round(float(np.median(a)), 4),
            "p90": round(float(np.percentile(a, 90)), 4),
            "max": round(float(a.max()), 4),
            "std": round(float(a.std()), 4),
        }

    report = {
        "first_phase": {
            "frames": pointing.frames,
            "dropout": round(pointing.dropout, 3),
            "index_extended_frac": (round(float(np.mean(pointing.index_extended)), 3)
                                    if pointing.index_extended else None),
            "palm_pose_frac": (round(float(np.mean(pointing.palm_pose)), 3)
                               if pointing.palm_pose else None),
            "curl": describe(pointing.curl),
            "pinch_3d": describe(pointing.pinch3d),
            "pinch_2d": describe(pointing.pinch2d),
            "thumb_middle_3d": describe(pointing.middle3d),
        },
        "second_phase": {
            "frames": closed.frames,
            "dropout": round(closed.dropout, 3),
            "curl": describe(closed.curl),
            "pinch_3d": describe(closed.pinch3d),
            "pinch_2d": describe(closed.pinch2d),
        },
    }
    path = ROOT / "tune_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
