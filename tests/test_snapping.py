"""Geometry of cursor snapping. No desktop, no camera, fully deterministic."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from gesture_control.config import Settings
from gesture_control.targets import Target, snap

ok = True


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


R = 45.0
button = Target(500, 300, 560, 330, "button")     # a small 60x30 button
far = Target(1500, 900, 1560, 930, "button")

print("Basic behaviour:")
p, t = snap((100, 100), [], R)
check("no targets leaves the point alone", np.allclose(p, (100, 100)) and t is None)

p, t = snap((100, 100), [far], R)
check("a target beyond the radius is ignored", np.allclose(p, (100, 100)) and t is None)

inside = (530, 315)
p, t = snap(inside, [button], R)
check("a point already inside the target is not moved",
      np.allclose(p, inside) and t is button)

start = (530, 360)          # 30 px below the button
p, t = snap(start, [button], R)
check(f"a point {button.distance(start):.0f}px away is pulled onto the button "
      f"(y {start[1]} -> {p[1]:.0f})", t is button and p[1] < start[1])

print("\nThe lock: the cursor goes to the centre and stays there.")
for d in (40, 30, 20, 10, 5):
    q = (530, button.bottom + d)
    p_, tgt = snap(q, [button], R)
    good = tgt is button and float(np.linalg.norm(p_ - button.center)) < 0.01
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] hand {d:2d}px away -> cursor at the "
          f"button centre ({p_[0]:.0f}, {p_[1]:.0f})")

rng0 = np.random.default_rng(11)
held = np.array([snap(button.center + rng0.normal(0, 14, 2), [button], R,
                      sticky=button)[0] for _ in range(400)])
print(f"      400 jittery frames moved it {np.abs(held - button.center).max():.2f}px")

# With the default inset the cursor aims a little way inside the control, so it
# arrives on the button sooner than the bare curve would put it.
check("the inset lands the cursor inside the control, not on its edge",
      all(button.contains(snap((530, button.bottom + d), [button], R)[0])
          for d in (5, 10)))

print("\nHolding on, and letting go:")
# A lock is a jump by definition -- the old gradual pull was continuous but
# never actually held still. What matters instead is that it does not rattle:
# it is acquired at one radius and released at a wider one.
UNLOCK = R * 1.8
check("stays locked well past the radius it was acquired at",
      snap((530, button.bottom + R + 25), [button], R, sticky=button)[1] is button)
check("but does let go eventually",
      snap((530, button.bottom + UNLOCK + 20), [button], R, sticky=button)[1] is None)
rng1 = np.random.default_rng(3)
edge = [snap((530, button.bottom + R + rng1.normal(0, 4)), [button], R,
             sticky=button)[1] for _ in range(200)]
check("and does not flicker on and off at the boundary",
      all(e is button for e in edge))

print("\nLong controls lock across, and run free along:")
bar = Target(200, 100, 1400, 132, "custom")          # a window title bar
scroll = Target(1900, 200, 1917, 800, "scrollbar")
for nm, tg, want in (("title bar", bar, (False, True)),
                     ("scrollbar", scroll, (True, False)),
                     ("button", button, (True, True))):
    got = tg.locked_axes()
    check(f"{nm} {tg.width}x{tg.height} -> lock x={got[0]}, y={got[1]}", got == want)
pts = [snap((x, 118), [bar], R, sticky=bar)[0] for x in (300, 700, 1100)]
print(f"      along the bar: x = {[int(q[0]) for q in pts]}, "
      f"y pinned at {pts[0][1]:.0f}")
check("the title bar pins y and lets x run freely",
      len({round(float(q[1]), 3) for q in pts}) == 1
      and [int(q[0]) for q in pts] == [300, 700, 1100])

print("\nWhat it is actually for -- jitter near a target:")
rng = np.random.default_rng(4)
truth = np.array([530.0, button.bottom + 12.0])
noisy = truth + rng.normal(0, 12.0, (6000, 2))
snapped = np.array([snap(q, [button], R)[0] for q in noisy])
hit_before = float(np.mean([button.contains(q) for q in noisy]))
hit = float(np.mean([button.contains(q) for q in snapped]))
print(f"      aiming 12px off a 60x30 button, hand jitter 12px:")
print(f"      lands on the button {hit_before:.0%} -> {hit:.0%} of the time")
check("snapping roughly doubles the hit rate", hit > hit_before * 1.8)

# The curve compresses near the target and must therefore expand further out.
# Bound how much, so the cost is known rather than discovered in use.
print("\n      jitter gain by distance from the target (1.0 = unchanged):")
worst = 0.0
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 45)]:
    base = np.array([[530.0, button.bottom + (lo + hi) / 2]])
    probe = base + rng.normal(0, 4.0, (3000, 2))
    moved = np.array([snap(q, [button], R)[0] for q in probe])
    gain = float(moved.std(0).mean() / probe.std(0).mean())
    worst = max(worst, gain)
    print(f"        {lo:2d}-{hi:2d}px away   gain {gain:.2f}")
check(f"jitter is never amplified more than 1.7x (worst {worst:.2f})", worst < 1.7)

print("\nTwo targets side by side:")
a = Target(500, 300, 560, 330, "button")
b = Target(575, 300, 635, 330, "button")
mid = (567, 315)                       # almost exactly between them
_, first = snap(mid, [a, b], R)
_, sticky = snap(mid, [a, b], R, sticky=b)
_, gone = snap((100, 100), [a, b], R, sticky=b)
check("the previously chosen target is preferred at a tie",
      first is a and sticky is b)
check("a sticky target out of range does not suppress the others", gone is None)
# and the choice must not flicker frame to frame under jitter
choices = [snap(mid + rng.normal(0, 3, 2), [a, b], R, sticky=b)[1] for _ in range(300)]
share = sum(1 for c in choices if c is b) / len(choices)
print(f"      stayed on the sticky target {share:.0%} of frames")
check("stickiness suppresses flicker between neighbours", share > 0.85)

print("\nSweeping along a row of buttons, none may be skipped:")
# The failure this guards against: hold the lock unconditionally until the
# hand passes the wider release radius, and that radius has to be smaller than
# the spacing between controls -- otherwise the cursor is still latched to the
# first button as it passes the second, and lands on the third.
s = Settings()
SR, SU, SM = s.snap_radius_px, s.snap_unlock_px, s.snap_sticky_margin_px
print(f"      shipped: radius {SR:.0f}px, release {SU:.0f}px, sticky margin {SM:.0f}px")
row = [Target(400 + i * 60, 300, 440 + i * 60, 340, "button") for i in range(5)]
centres = [int(t.center[0]) for t in row]
held, visited = None, []
for x in range(380, 720, 2):
    _, held = snap((x, 320), row, SR, sticky=held, unlock_radius=SU,
                   sticky_margin=SM)
    name = None if held is None else int(held.center[0])
    if not visited or visited[-1] != name:
        visited.append(name)
print(f"      buttons at x = {centres}")
print(f"      cursor visited  {[v for v in visited if v is not None]}")
check("every button in the row gets its turn",
      [v for v in visited if v is not None] == centres)

# Visiting every button is necessary but not sufficient: the lock can hand over
# so late that the cursor sits on the button you have already left. That is the
# milder form of the same complaint, so measure the depth directly.
worst_lag = 0.0
held = None
for x10 in range(3800, 7200):
    x = x10 / 10.0
    _, held = snap((x, 320), row, SR, sticky=held, unlock_radius=SU, sticky_margin=SM)
    for tgt in row:
        if tgt.contains((x, 320)) and held is not tgt:
            worst_lag = max(worst_lag, min(x - tgt.left, tgt.right - x))
print(f"      the old lock holds up to {worst_lag:.0f}px into the next button")
check("the lock hands over near the edge of the next control, not deep inside",
      worst_lag <= 8.0)

# ...while still not trading the lock back and forth between two neighbours.
pair = [Target(500, 300, 560, 330, "button"), Target(575, 300, 635, 330, "button")]
worst = 0
for seed in range(8):
    rng2 = np.random.default_rng(seed)
    held, swaps = pair[1], 0
    for _ in range(300):
        _, new = snap(np.array([567.0, 315.0]) + rng2.normal(0, 6, 2), pair, SR,
                      sticky=held, unlock_radius=SU, sticky_margin=SM)
        swaps += new is not held
        held = new
    worst = max(worst, swaps)
print(f"      between two touching buttons, 6px of shake, worst of 8 seeds:")
print(f"      the lock changed {worst} times in 300 frames")
# Not zero, and it cannot be: a basin this small keeps the hand near a boundary
# for more of its travel. The bound records what the shipped radius costs.
check("a hand between two buttons does not rattle badly", worst <= 6)

print("\nRejecting things that are not really targets:")
from gesture_control.targets import TargetFinder
f = TargetFinder((1920, 1080))
check("a full-window pane is rejected",
      not f._plausible(Target(0, 0, 1900, 1000, "custom")))
check("a 2px sliver is rejected", not f._plausible(Target(10, 10, 12, 12, "button")))
check("an ordinary button is accepted",
      f._plausible(Target(100, 100, 180, 130, "button")))
check("something on another monitor is rejected",
      not f._plausible(Target(2200, 100, 2280, 130, "button")))

print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
