"""End-to-end test of finger aiming: calibration UI + gesture state machine.

No camera, no real mouse. The Tk window is withdrawn so nothing appears on screen.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from dataclasses import replace

from gesture_control import landmarks as lm
from gesture_control.config import Settings

from synthetic_hand import (ASPECT, POINTING, OPEN_PALM, TWO_FINGERS,
                            SCREEN, FakeTracker, make_hand, rng)


def check(label, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    return cond


# ---------------------------------------------------------------- sanity
print("Synthetic hand poses:")
h = make_hand((1280, 720))
ext = h.extended()
ok = True
ok &= check("pointing pose: index extended, others curled",
            ext["index"] and not ext["middle"] and not ext["ring"] and not ext["pinky"])
ok &= check(f"pointing pose: not pinched (ratio {h.pinch():.2f})", h.pinch() > 0.55)
hp = make_hand((1280, 720), pinch="index")
ok &= check(f"pinched pose: pinched (ratio {hp.pinch():.2f})", hp.pinch() < 0.42)
op = make_hand((1280, 720), pose=OPEN_PALM)
ok &= check("open palm: all five extended", all(op.extended().values()))
tf = make_hand((1280, 720), pose=TWO_FINGERS)
e2 = tf.extended()
ok &= check("two fingers: index+middle up, ring+pinky down",
            e2["index"] and e2["middle"] and not e2["ring"] and not e2["pinky"])

# ---------------------------------------------------------------- calibration UI
print("\nCalibration window (withdrawn, driven frame by frame):")
from gesture_control.calibrate import CalibrationWindow

# This file covers the original finger-aiming mode, which is no longer the
# default; palm aiming has its own suite in test_palm.py.
settings = Settings(aim_mode="finger")
settings.samples_per_point = 8
settings.settle_frames = 3
tracker = FakeTracker()
win = CalibrationWindow(tracker, settings)
win.root.withdraw()
win.width, win.height = SCREEN
from gesture_control.calibrate import target_grid, _Dot
win.dots = [_Dot(p) for p in target_grid(SCREEN[0], SCREEN[1],
                                         settings.grid_cols, settings.grid_rows,
                                         settings.margin_frac)]

clock = [0.0]
def feed(target, pose=POINTING, pinch=None, scale=0.30):
    clock[0] += 1 / 30
    tracker.hand = make_hand(target, pose, pinch, scale, stamp=clock[0])
    win._tick()

# intro: hand visible, then a pinch to start
for _ in range(30):
    feed(win.dots[0].position)
check(f"intro waits for a steady hand (phase={win.phase})", win.phase == "intro")
for _ in range(3):
    feed(win.dots[0].position, pinch="index")
check(f"pinch starts collection (phase={win.phase})", win.phase == "collect")

guard = 0
last_index = -1
while win.phase == "collect" and guard < 6000:
    guard += 1
    dot = win.dots[win.index]
    # Hand size varies a little between dots, to exercise the depth term.
    scale = 0.30 + 0.02 * np.sin(win.index)
    if win.index != last_index:
        # Move to the new dot with the hand open, the way a person does: each
        # dot needs its own pinch, and calibration insists on seeing a release.
        for _ in range(4):
            feed(dot.position, scale=scale)
        last_index = win.index
    feed(dot.position, pinch="index", scale=scale)

ok &= check(f"all {len(win.dots)} dots collected", win.phase == "verify")
ok &= check("model was fitted", win.model is not None)
if win.model is not None:
    m = win.model.meta
    print(f"      model={m['feature_set']}  lambda={m['ridge_lambda']:.3f}  "
          f"fit={m['fit_rms_px']:.1f}px  loocv={m['loocv_rms_px']:.1f}px")
    ok &= check("cross-validated accuracy under 1% of screen height",
                m["loocv_rms_px"] < SCREEN[1] * 0.01)
    # Accuracy at fresh, off-grid points.
    tgt = np.column_stack([rng.uniform(0.12, 0.88, 200) * SCREEN[0],
                           rng.uniform(0.12, 0.88, 200) * SCREEN[1]])
    pred = np.array([win.model.predict(make_hand(t).sample) for t in tgt])
    err = np.linalg.norm(pred - tgt, axis=1)
    print(f"      unseen points: rms {err.mean():.0f}px  p95 {np.percentile(err,95):.0f}px")
    ok &= check("unseen-point p95 error under 60 px", np.percentile(err, 95) < 60)

ok &= check(f"pinch thresholds derived from the hand: {win.thresholds}",
            win.thresholds is not None)

# A threshold loose enough that the hand never reads as released used to fill
# every dot in one cascade, recording the whole calibration from wherever the
# hand happened to be. Each dot must demand its own fresh pinch.
print("\nA hand stuck in the pinched state must not fill the dots by itself:")
stuck = replace(settings, samples_per_point=8, settle_frames=3,
                pinch_on=1.6, pinch_off=1.9)      # everything reads as pinched
tracker2 = FakeTracker()
win2 = CalibrationWindow(tracker2, stuck)
win2.root.withdraw()
win2.width, win2.height = SCREEN
win2.dots = [_Dot(p) for p in target_grid(SCREEN[0], SCREEN[1],
                                          stuck.grid_cols, stuck.grid_rows,
                                          stuck.margin_frac)]
clock2 = [0.0]
for _ in range(400):
    clock2[0] += 1 / 30
    tracker2.hand = make_hand(win2.dots[0].position, POINTING, None, stamp=clock2[0])
    win2._tick()
filled = sum(1 for d in win2.dots if d.done)
print(f"      after 400 frames of a permanently-pinched hand: "
      f"{filled}/{len(win2.dots)} dots filled, phase={win2.phase}")
ok &= check("no dots are captured without a release first",
            filled == 0 and win2.phase == "collect")
try:
    win2.root.destroy()
except Exception:
    pass

# Calibration must survive pinch detection being useless, because that is the
# state someone is in when they come here to fix it.
print("\nCalibrating with pinch detection completely broken (dwell only):")
broken = replace(settings, samples_per_point=6, settle_frames=2,
                 dwell_ms=300.0, pinch_on=0.0, pinch_off=0.0)   # never pinched
tracker3 = FakeTracker()
win3 = CalibrationWindow(tracker3, broken)
win3.root.withdraw()
win3.width, win3.height = SCREEN
win3.dots = [_Dot(p) for p in target_grid(SCREEN[0], SCREEN[1],
                                          broken.grid_cols, broken.grid_rows,
                                          broken.margin_frac)]
win3._on_start()                       # Space, since a pinch can never fire
clock3 = [0.0]
guard = last = -1
while win3.phase == "collect" and guard < 8000:
    guard += 1
    if win3.index != last:             # travel to the new dot, then hold still
        for step in range(4):
            clock3[0] += 1 / 30
            drift = np.array([0.06, 0.06]) * (4 - step)
            tracker3.hand = make_hand(np.array(win3.dots[win3.index].position) + drift * 300,
                                      POINTING, None, noise=0.0, stamp=clock3[0])
            win3._tick()
        last = win3.index
    clock3[0] += 1 / 30
    tracker3.hand = make_hand(win3.dots[win3.index].position, POINTING, None,
                              noise=0.0, stamp=clock3[0])
    win3._tick()
print(f"      phase={win3.phase}, dots captured="
      f"{sum(1 for d in win3.dots if d.done)}/{len(win3.dots)}")
ok &= check("holding still captures every dot without a single pinch",
            win3.phase == "verify" and win3.model is not None)
if win3.model is not None:
    ok &= check(f"and the fit is good "
                f"({win3.model.meta['loocv_rms_px']:.0f}px)",
                win3.model.meta["loocv_rms_px"] < SCREEN[1] * 0.03)
try:
    win3.root.destroy()
except Exception:
    pass
win._on_enter()
ok &= check("Enter accepts the model", win.result is not None)
model = win.result

# ---------------------------------------------------------------- gestures
print("\nGesture state machine (mouse calls recorded, not sent):")
from gesture_control.controller import GestureController

class FakeMouse:
    def __init__(self): self.events = []; self._left = False
    def move_to(self, x, y): self.events.append(("move", round(x), round(y)))
    def left_down(self):
        if not self._left: self._left = True; self.events.append(("left_down",))
    def left_up(self):
        if self._left: self._left = False; self.events.append(("left_up",))
    @property
    def left_is_down(self): return self._left
    def right_click(self): self.events.append(("right_click",))
    def scroll(self, n): self.events.append(("scroll", round(n, 3)))
    def release_all(self): self.left_up()

def new_controller():
    c = GestureController(FakeTracker(), model, settings)
    c.mouse = FakeMouse()
    return c

def drive(c, frames):
    # Carry the clock across calls. Restarting it rewinds time for the
    # controller, which then sees a press that began in the future.
    t = [getattr(c, "_clock", 100.0)]
    for target, pose, pinch, n in frames:
        for _ in range(n):
            t[0] += 1 / 30
            c._update(make_hand(target, pose, pinch, stamp=t[0]))
    c._clock = t[0]
    return c.mouse.events

# move
c = new_controller()
ev = drive(c, [((600, 400), POINTING, None, 20), ((1900, 1000), POINTING, None, 20)])
moves = [e for e in ev if e[0] == "move"]
ok &= check(f"pointing moves the cursor ({len(moves)} moves, mode={c.mode})",
            len(moves) > 30 and c.mode == "move")
ok &= check(f"cursor converges on the aimed point {moves[-1][1:]} vs (1900, 1000)",
            abs(moves[-1][1] - 1900) < 60 and abs(moves[-1][2] - 1000) < 60)

# click
c = new_controller()
ev = drive(c, [((800, 500), POINTING, None, 15),
               ((800, 500), POINTING, "index", 4),
               ((800, 500), POINTING, None, 6)])
names = [e[0] for e in ev]
ok &= check("quick pinch = one left_down + left_up",
            names.count("left_down") == 1 and names.count("left_up") == 1)
ok &= check("click order is down then up",
            names.index("left_down") < names.index("left_up"))
down_at = ev[names.index("left_down") - 1]
ok &= check(f"click lands on target {down_at[1:]} vs (800, 500)",
            abs(down_at[1] - 800) < 70 and abs(down_at[2] - 500) < 70)
held = ev[names.index("left_down"):names.index("left_up")]
ok &= check("cursor is frozen between press and release (short pinch)",
            not any(e[0] == "move" for e in held))


def click_error(window_ms, trials=60):
    """RMS distance from the intended target over many simulated clicks."""
    errs = []
    for i in range(trials):
        target = (400 + (i * 137) % 1700, 300 + (i * 71) % 800)
        c = new_controller()
        c.settings = replace(settings, click_anchor_window_ms=window_ms)
        ev = drive(c, [(target, POINTING, None, 14),
                       (target, POINTING, "index", 3),
                       (target, POINTING, None, 3)])
        names = [e[0] for e in ev]
        if "left_down" not in names:
            continue
        at = ev[names.index("left_down") - 1]
        errs.append((at[1] - target[0]) ** 2 + (at[2] - target[1]) ** 2)
    return float(np.sqrt(np.mean(errs)))


single = click_error(0.0)      # nearest single frame, the old behaviour
averaged = click_error(settings.click_anchor_window_ms)
print(f"      click RMS: single frame {single:.1f} px -> averaged window {averaged:.1f} px")
ok &= check("averaging the pre-pinch anchor reduces click error",
            averaged < single)

# drag
c = new_controller()
ev = drive(c, [((500, 400), POINTING, None, 12),
               ((500, 400), POINTING, "index", 20),
               ((1500, 900), POINTING, "index", 25),
               ((1500, 900), POINTING, None, 5)])
names = [e[0] for e in ev]
after_down = ev[names.index("left_down"):names.index("left_up")]
drag_moves = [e for e in after_down if e[0] == "move"]
ok &= check(f"held pinch becomes a drag ({len(drag_moves)} moves while held)",
            len(drag_moves) > 10)
ok &= check("drag releases at the end", names.count("left_up") == 1)

# right click
c = new_controller()
ev = drive(c, [((900, 600), POINTING, None, 12),
               ((900, 600), POINTING, "middle", 5),
               ((900, 600), POINTING, None, 5)])
ok &= check("thumb+middle pinch = exactly one right click",
            [e[0] for e in ev].count("right_click") == 1)

# scroll
c = new_controller()
ev = drive(c, [((1280, 900), TWO_FINGERS, None, 10),
               ((1280, 400), TWO_FINGERS, None, 30)])
scrolls = [e[1] for e in ev if e[0] == "scroll"]
ok &= check(f"two fingers moving up scrolls up (total {sum(scrolls):+.1f} notches)",
            len(scrolls) > 5 and sum(scrolls) > 0)
ok &= check("no cursor movement while scrolling",
            not any(e[0] == "move" for e in ev[-len(scrolls):]))

# pause
c = new_controller()
drive(c, [((900, 600), POINTING, None, 5), ((900, 600), OPEN_PALM, None, 40)])
ok &= check(f"open palm pauses (paused={c.paused})", c.paused)
before = len(c.mouse.events)
drive(c, [((300, 300), POINTING, None, 20)])
ok &= check("nothing is sent while paused", len(c.mouse.events) == before)

# stuck-button safety
c = new_controller()
drive(c, [((900, 600), POINTING, None, 10), ((900, 600), POINTING, "index", 20)])
ok &= check("button is down mid-drag", c.mouse.left_is_down)
c._release_everything()
ok &= check("release_everything lifts the button", not c.mouse.left_is_down)

try:
    win.root.destroy()
except Exception:
    pass
# ------------------------------------------------- snapping, through the controller
print("\nClicking small buttons, with and without snapping:")
from gesture_control.targets import Target, TargetFinder

# A toolbar of 28x28 icons, the case hand pointing is worst at.
buttons = [Target(300 + i * 40, 200, 328 + i * 40, 228, "button") for i in range(8)]
buttons += [Target(300 + i * 40, 600, 328 + i * 40, 628, "button") for i in range(8)]


class FakeFinder:
    def __init__(self, targets): self._t = tuple(targets)
    def targets(self): return self._t
    def note_cursor(self, x, y): pass


def hit_rate(snapping, jitter, repeats=6):
    """Fraction of deliberate clicks that land inside the aimed-at button.

    Aim is deliberately imperfect: the hand targets a point a few pixels off
    each button centre, which is the situation snapping exists to rescue.
    """
    hits = total = 0
    for repeat in range(repeats):
        for button in buttons:
            c = new_controller()
            c.settings = replace(settings, snap_enabled=snapping)
            if snapping:
                c.finder = FakeFinder(buttons)
            cx, cy = button.center
            aim = (cx + 11 * np.cos(repeat * 1.7), cy + 11 * np.sin(repeat * 1.7))
            t = 100.0
            for pinch, n in [(None, 20), ("index", 3), (None, 3)]:
                for _ in range(n):
                    t += 1 / 30
                    c._update(make_hand(aim, POINTING, pinch,
                                        noise=jitter, stamp=t))
            names = [e[0] for e in c.mouse.events]
            if "left_down" not in names:
                continue
            at = c.mouse.events[names.index("left_down") - 1]
            total += 1
            hits += bool(button.contains((at[1], at[2])))
    return hits / max(total, 1)


for label, jitter in [("typical", 0.0031), ("poor", 0.0055)]:
    plain = hit_rate(False, jitter)
    snapped = hit_rate(True, jitter)
    print(f"      28x28 buttons, {label} tracking: "
          f"{plain:.0%} of clicks land on target -> {snapped:.0%} with snapping")
    ok &= check(f"snapping improves small-button clicks ({label})", snapped > plain)

# It must stay out of the way of a drag.
c = new_controller()
c.finder = FakeFinder(buttons)
drive(c, [((500, 400), POINTING, None, 12), ((500, 400), POINTING, "index", 20)])
drive(c, [((332, 214), POINTING, "index", 25)])     # dragged right over a button
mid_drag_target = c.snap_target
ev = c.mouse.events
names = [e[0] for e in ev]
during = [e for e in ev[names.index("left_down"):] if e[0] == "move"]
ok &= check(f"a drag is not yanked around by the buttons it passes over "
            f"({len(during)} moves, target={mid_drag_target})",
            len(during) > 10 and mid_drag_target is None)
drive(c, [((332, 214), POINTING, None, 4)])
ok &= check("the drag still releases cleanly", not c.mouse.left_is_down)

print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
