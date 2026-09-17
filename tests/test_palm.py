"""Palm aiming: open hand moves the cursor, a closed fist clicks."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dataclasses import replace

import numpy as np

from gesture_control import landmarks as lm
from gesture_control.config import Settings
from gesture_control.controller import GestureController
from gesture_control.calibrate import CalibrationWindow, _Dot, target_grid
from gesture_control.mapping import fit_pointer_map
from synthetic_hand import (FIST, OPEN_PALM, SCREEN, TWO_FINGERS,
                            FakeTracker, make_hand)

ok = True


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


# ---------------------------------------------------------------- aim stability
print("Closing the hand changes its shape. How much does each aim point move?")
# Measured on the templates themselves, anchored at the wrist, so this is the
# shape change alone and not an artefact of where the hand was placed.
def tpl(pose):
    a = np.zeros((21, 2))
    for i, xy in pose.items():
        a[i] = xy
    return a - a[lm.WRIST]

open_t, fist_t = tpl(OPEN_PALM), tpl(FIST)
palm_shift = float(np.linalg.norm(lm.palm_center(fist_t) - lm.palm_center(open_t)))
tip_shift = float(np.linalg.norm(fist_t[lm.INDEX_TIP] - open_t[lm.INDEX_TIP]))
print(f"      palm centre moves {palm_shift:.3f} palm-lengths, "
      f"index tip moves {tip_shift:.3f}")
check("the palm centre is far steadier through a click than a fingertip",
      palm_shift < tip_shift / 10)

# ---------------------------------------------------------------- pose reading
print("\nPose classification:")
settings = replace(Settings(), aim_mode="palm")
tgt = target_grid(SCREEN[0], SCREEN[1], 4, 3, settings.margin_frac)
model = fit_pointer_map(
    [make_hand(t, OPEN_PALM, anchor="palm", noise=0.0004).aim_sample("palm")
     for t in tgt], tgt, SCREEN)
model.meta["aim_mode"] = "palm"


class FakeMouse:
    def __init__(self): self.events = []; self._left = False
    def move_to(self, x, y): self.events.append(("move", x, y))
    def left_down(self):
        if not self._left: self._left = True; self.events.append(("left_down",))
    def left_up(self):
        if self._left: self._left = False; self.events.append(("left_up",))
    @property
    def left_is_down(self): return self._left
    def right_click(self): self.events.append(("right_click",))
    def scroll(self, n): self.events.append(("scroll", round(n, 3)))
    def release_all(self): self.left_up()


def controller():
    c = GestureController(FakeTracker(), model, settings)
    c.mouse = FakeMouse()
    return c


c = controller()
for name, pose, want in [("open palm", OPEN_PALM, "open"), ("fist", FIST, "fist"),
                         ("two fingers", TWO_FINGERS, "scroll")]:
    got = c._classify(make_hand((960, 540), pose, anchor="palm"))
    check(f"{name} reads as '{got}'", got == want)


def drive(c, steps):
    t = [100.0]
    for pose, n in steps:
        for _ in range(n):
            t[0] += 1 / 30
            c._update(make_hand(pose[1], pose[0], anchor="palm", stamp=t[0]))
    return c.mouse.events


# ---------------------------------------------------------------- clicking
print("\nClicking:")
c = controller()
ev = drive(c, [((OPEN_PALM, (900, 500)), 18), ((FIST, (900, 500)), 5),
               ((OPEN_PALM, (900, 500)), 6)])
names = [e[0] for e in ev]
check("closing the fist clicks exactly once",
      names.count("left_down") == 1 and names.count("left_up") == 1)
at = ev[names.index("left_down") - 1]
check(f"the click lands on target ({at[1]:.0f}, {at[2]:.0f}) vs (900, 500)",
      abs(at[1] - 900) < 60 and abs(at[2] - 500) < 60)
check("the cursor is frozen while the fist is closed",
      not any(e[0] == "move"
              for e in ev[names.index("left_down"):names.index("left_up")]))

print("\nThe deadband between open and closed:")
# Sweep the hand slowly from open to shut and back, through every intermediate
# shape, and count how many clicks come out. It must be exactly one.
c = controller()
t = 100.0
for cycle in range(3):
    for frac in list(np.linspace(0, 1, 14)) + list(np.linspace(1, 0, 14)):
        t += 1 / 30
        blend = {k: tuple(np.array(OPEN_PALM[k]) * (1 - frac) + np.array(FIST[k]) * frac)
                 for k in OPEN_PALM}
        c._update(make_hand((900, 500), blend, anchor="palm", stamp=t))
clicks = [e[0] for e in c.mouse.events].count("left_down")
print(f"      three slow open-close-open cycles produced {clicks} clicks")
check("each cycle gives exactly one click, none from the transition", clicks == 3)

print("\nDragging:")
c = controller()
ev = drive(c, [((OPEN_PALM, (500, 400)), 14), ((FIST, (500, 400)), 18),
               ((FIST, (1400, 900)), 22), ((OPEN_PALM, (1400, 900)), 14)])
names = [e[0] for e in ev]
during = [e for e in ev[names.index("left_down"):names.index("left_up")]
          if e[0] == "move"]
check(f"a held fist drags ({len(during)} moves while closed)", len(during) > 8)
check("and releases when the hand opens", names.count("left_up") == 1)
check("nothing is left held down", not c.mouse.left_is_down)

print("\nOther poses:")
c = controller()
ev = drive(c, [((OPEN_PALM, (900, 900)), 10), ((TWO_FINGERS, (900, 900)), 8),
               ((TWO_FINGERS, (900, 400)), 20)])
scrolls = [e[1] for e in ev if e[0] == "scroll"]
check(f"two fingers moving up scrolls ({sum(scrolls):+.1f} notches)",
      len(scrolls) > 3 and sum(scrolls) > 0)
check("no clicks while scrolling",
      "left_down" not in [e[0] for e in ev])

# Right click is a thumb-to-middle-finger pinch, told from a left click by the
# index being clearly further from the thumb than the middle is.
c = controller()
t = [100.0]
for pinch, n in [(None, 12), ("middle", 8), (None, 6)]:
    for _ in range(n):
        t[0] += 1 / 30
        c._update(make_hand((900, 500), OPEN_PALM, pinch, anchor="palm",
                            stamp=t[0]))
names = [e[0] for e in c.mouse.events]
check(f"pinching the middle finger right clicks "
      f"({names.count('right_click')} right, {names.count('left_down')} left)",
      names.count("right_click") == 1 and names.count("left_down") == 0)

# ---------------------------------------------------------------- calibration
print("\nCalibrating in palm mode (fist as the capture trigger):")
cal = replace(settings, samples_per_point=6, settle_frames=2)
tracker = FakeTracker()
win = CalibrationWindow(tracker, cal)
win.root.withdraw()
win.width, win.height = SCREEN
win.dots = [_Dot(p) for p in target_grid(SCREEN[0], SCREEN[1],
                                         cal.grid_cols, cal.grid_rows,
                                         cal.margin_frac)]
clock = [0.0]


def feed(pose, target, n=1):
    for _ in range(n):
        clock[0] += 1 / 30
        tracker.hand = make_hand(target, pose, anchor="palm", stamp=clock[0])
        win._tick()


feed(OPEN_PALM, win.dots[0].position, 25)
check(f"an open hand does not start calibration by itself (phase={win.phase})",
      win.phase == "intro")
feed(FIST, win.dots[0].position, 4)
check(f"a fist starts it (phase={win.phase})", win.phase == "collect")

guard, last = 0, -1
while win.phase == "collect" and guard < 6000:
    guard += 1
    dot = win.dots[win.index]
    if win.index != last:
        feed(OPEN_PALM, dot.position, 4)      # open up between dots
        last = win.index
    feed(FIST, dot.position)
check(f"all dots captured with fists (phase={win.phase})", win.phase == "verify")
if win.model is not None:
    acc = win.model.meta["loocv_rms_px"]
    print(f"      accuracy {acc:.0f}px on a {SCREEN[1]}px-tall screen")
    check("and the fit is accurate", acc < SCREEN[1] * 0.02)
try:
    win.root.destroy()
except Exception:
    pass

# ------------------------------------------------- aiming by turning the hand
print("\nA hand that stays put and only turns to aim:")
# The palm centre never moves; only the facing changes. Position carries no
# signal at all here, so the fit has to notice that and use orientation.
def turned(target, noise=0.0006):
    fx = target[0] / SCREEN[0] - 0.5
    fy = target[1] / SCREEN[1] - 0.5
    return make_hand((960, 540), OPEN_PALM, anchor="palm", noise=noise,
                     tilt=(-fy * 40.0, fx * 50.0))


turn_samples = [turned(t).aim_sample("palm") for t in tgt]
centres = np.array([s.tip for s in turn_samples])
print(f"      palm centre varies by {np.ptp(centres, axis=0).max():.4f} of the frame "
      f"(i.e. not at all)")
turn_model = fit_pointer_map(turn_samples, tgt, SCREEN)
print(f"      chosen model: aims by {turn_model.meta['aims_by']}, "
      f"accuracy {turn_model.meta['loocv_rms_px']:.0f}px")
check(f"the fit does not try to explain this by position alone "
      f"(chose {turn_model.meta['aims_by']})",
      turn_model.meta["aims_by"] != "position")
check("and calibrates accurately from it",
      turn_model.meta["loocv_rms_px"] < SCREEN[1] * 0.03)

probe = np.column_stack([np.random.default_rng(2).uniform(0.15, 0.85, 120) * SCREEN[0],
                         np.random.default_rng(3).uniform(0.15, 0.85, 120) * SCREEN[1]])
err = np.linalg.norm(
    turn_model.predict_many([turned(t).aim_sample("palm") for t in probe]) - probe,
    axis=1)
print(f"      unseen points: rms {err.mean():.0f}px, p95 {np.percentile(err, 95):.0f}px")
check("and generalises to points it never saw", np.percentile(err, 95) < SCREEN[1] * 0.06)

# And the reverse: a hand that only moves must still choose position.
move_samples = [make_hand(t, OPEN_PALM, anchor="palm", noise=0.0006).aim_sample("palm")
                for t in tgt]
move_model = fit_pointer_map(move_samples, tgt, SCREEN)
print(f"      a hand that only moves: aims by {move_model.meta['aims_by']}, "
      f"{move_model.meta['loocv_rms_px']:.0f}px")
check("a hand that moves is still fitted by position",
      move_model.meta["aims_by"] == "position")

print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
