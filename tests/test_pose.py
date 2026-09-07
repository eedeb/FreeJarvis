"""Pose reading: why it is done in 3D, and why clicks are debounced.

Pointing at a screen means pointing roughly at the camera, so the hand is
seen almost end-on. These tests reproduce that view and check the two
illusions it creates are gone.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from gesture_control import landmarks as lm
from gesture_control.controller import _Debounced

ok = True


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


def rot_x(deg):
    t = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


# A hand with a straight index finger, built in its own frame: the finger runs
# along -y, the palm sits at the origin.
hand = np.zeros((21, 3))
hand[lm.WRIST] = (0.00, 0.00, 0)
hand[lm.THUMB_CMC] = (-0.03, -0.01, 0)
hand[lm.THUMB_MCP] = (-0.05, -0.04, 0)
hand[lm.THUMB_IP] = (-0.06, -0.06, 0)
hand[lm.THUMB_TIP] = (-0.07, -0.08, 0)
hand[lm.INDEX_MCP] = (-0.03, -0.09, 0)
hand[lm.INDEX_PIP] = (-0.03, -0.14, 0)
hand[lm.INDEX_DIP] = (-0.03, -0.17, 0)
hand[lm.INDEX_TIP] = (-0.03, -0.20, 0)
hand[lm.MIDDLE_MCP] = (0.00, -0.09, 0)
hand[lm.MIDDLE_PIP] = (0.00, -0.14, 0)
hand[11] = (0.00, -0.11, 0)
hand[lm.MIDDLE_TIP] = (0.00, -0.08, 0)          # curled
hand[lm.RING_MCP] = (0.03, -0.09, 0)
hand[lm.RING_PIP] = (0.03, -0.13, 0)
hand[15] = (0.03, -0.10, 0)
hand[lm.RING_TIP] = (0.03, -0.075, 0)           # curled
hand[lm.PINKY_MCP] = (0.05, -0.08, 0)
hand[lm.PINKY_PIP] = (0.05, -0.12, 0)
hand[19] = (0.05, -0.09, 0)
hand[lm.PINKY_TIP] = (0.05, -0.07, 0)           # curled


def old_style_extended(points, finger):
    """The distance-from-wrist test this used to use, for comparison."""
    tip, pip = {"index": (lm.INDEX_TIP, lm.INDEX_PIP)}[finger]
    wrist = points[lm.WRIST]
    return bool(np.linalg.norm(points[tip] - wrist)
                > np.linalg.norm(points[pip] - wrist) * 1.05)


print("A straight finger, seen more and more end-on:")
print(f"  {'tilt':>6} {'2D looks':>10}  {'old test':>9} {'new test':>9} {'3D':>5}")
old_failures = new_failures = 0
for tilt in (0, 30, 50, 65, 75, 85):
    world = hand @ rot_x(tilt).T
    flat = world[:, :2]                       # what the camera sees
    apparent = np.linalg.norm(flat[lm.INDEX_TIP] - flat[lm.INDEX_MCP]) * 100
    old = old_style_extended(flat, "index")
    new = lm.finger_extended(flat, "index")
    in3d = lm.finger_extended(world, "index")
    old_failures += not old
    new_failures += not new
    print(f"  {tilt:5d}° {apparent:8.1f}cm  {str(old):>9} {str(new):>9} {str(in3d):>5}")
    check(f"  {tilt}°: 3D still sees the finger as straight", in3d)

check("the straightness test never loses it", new_failures == 0)
# The distance-from-wrist proxy survives this case too: a pure rotation scales
# the wrist-to-tip and wrist-to-pip distances alike, so their ratio holds. It
# is kept here as the honest comparison -- the reason for changing it is that
# it answers a different question from the one being asked, not that it fails
# on a tilt. Where it does break is a curled finger seen end-on:
curled_at_camera = hand @ rot_x(80).T
flat_curled = curled_at_camera[:, :2]
old_says = (np.linalg.norm(flat_curled[lm.MIDDLE_TIP] - flat_curled[lm.WRIST])
            > np.linalg.norm(flat_curled[lm.MIDDLE_PIP] - flat_curled[lm.WRIST]) * 1.05)
new_says = lm.finger_extended(flat_curled, "middle")
in3d_says = lm.finger_extended(curled_at_camera, "middle")
print(f"      a curled middle finger seen end-on: "
      f"old test says extended={old_says}, new says {new_says}, 3D says {in3d_says}")
check("the curled finger is correctly read as curled in 3D", not in3d_says)

print("\nA thumb held well clear of the index finger, but behind it:")
# Thumb 5 cm from the index tip in 3D, but almost on top of it once flattened.
false_pinch = hand.copy()
false_pinch[lm.THUMB_TIP] = false_pinch[lm.INDEX_TIP] + np.array([0.002, 0.002, 0.05])
flat = false_pinch[:, :2]
r2d = lm.pinch_ratio(flat, lm.INDEX_TIP)
r3d = lm.pinch_ratio(false_pinch, lm.INDEX_TIP)
gap = np.linalg.norm(false_pinch[lm.THUMB_TIP] - false_pinch[lm.INDEX_TIP]) * 100
print(f"      real gap {gap:.1f}cm   2D ratio {r2d:.2f}   3D ratio {r3d:.2f}")
check("2D reports a pinch that is not happening", r2d < 0.42)
check("3D does not", r3d > 0.55)

print("\nAnd a genuine pinch is still detected in 3D:")
real = hand.copy()
real[lm.THUMB_TIP] = real[lm.INDEX_TIP] + np.array([0.004, 0.008, 0.002])
print(f"      3D ratio {lm.pinch_ratio(real, lm.INDEX_TIP):.2f}")
check("a real pinch reads as pinched", lm.pinch_ratio(real, lm.INDEX_TIP) < 0.42)

print("\nFinger splay (the second gate on the pause gesture):")
splayed = hand.copy()
for i, x in ((lm.INDEX_TIP, -0.06), (lm.MIDDLE_TIP, -0.01),
             (lm.RING_TIP, 0.04), (lm.PINKY_TIP, 0.08)):
    splayed[i] = (x, -0.19, 0)
together = hand.copy()
for i, x in ((lm.INDEX_TIP, -0.025), (lm.MIDDLE_TIP, -0.005),
             (lm.RING_TIP, 0.015), (lm.PINKY_TIP, 0.035)):
    together[i] = (x, -0.19, 0)
print(f"      splayed {lm.fingers_spread(splayed):.2f}   "
      f"together {lm.fingers_spread(together):.2f}")
check("splayed and together are distinguishable",
      lm.fingers_spread(splayed) > 0.30 >= lm.fingers_spread(together))

print("\nClick debounce:")
d = _Debounced(press_frames=2, release_frames=1)
check("a single bad frame does not press", d.update(True) is False)
check("two frames in a row do", d.update(True) is True)
check("it stays pressed while held", d.update(True) is True)
check("one clean frame releases", d.update(False) is False)

d = _Debounced(press_frames=2)
d.update(True)
check("an isolated glitch is forgotten", d.update(False) is False)
check("...and does not carry over", d.update(True) is False)

d = _Debounced(press_frames=3)
glitches = [True, False, True, False, True, False] * 4     # alternating noise
states = [d.update(g) for g in glitches]
check("alternating noise never presses", not any(states))

print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
