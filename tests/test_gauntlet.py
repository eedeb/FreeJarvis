"""The gauntlet overlay. Cosmetic, so what is tested is that it stays cosmetic.

No camera and no desktop: the renderer takes pixel landmarks and a frame.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from gesture_control import gauntlet
from gesture_control import landmarks as lm
from gesture_control.config import Settings
from synthetic_hand import OPEN_PALM, POINTING, FIST

ok = True


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


def px(template, scale=110.0, cx=320.0, cy=300.0):
    pts = np.zeros((21, 2))
    for i, xy in template.items():
        pts[i] = xy
    return pts * scale + np.array([cx, cy])


def blank(w=640, h=480):
    return np.full((h, w, 3), 28, np.uint8)


POSES = {"open palm": OPEN_PALM, "pointing": POINTING, "fist": FIST}

print("It draws, for every pose the tracker can report:")
for name, tpl in POSES.items():
    frame = blank()
    gauntlet.draw(frame, px(tpl), "Right")
    painted = float(np.mean(np.any(frame != 28, axis=2)))
    check(f"{name}: covers {painted:.0%} of the frame", 0.02 < painted < 0.60)

print("\nIt is a renderer, so it must not disturb what it is given:")
pts = px(OPEN_PALM)
before = pts.copy()
frame = blank()
result = gauntlet.draw(frame, pts, "Right")
check("the landmark array comes back unmodified", np.array_equal(pts, before))
check("draw() returns nothing for the caller to act on", result is None)

print("\nIt paints on the hand and nowhere else:")
frame = blank()
gauntlet.draw(frame, pts, "Right")
lo = pts.min(axis=0) - 130
hi = pts.max(axis=0) + 130
outside = frame.copy()
x0, y0 = max(int(lo[0]), 0), max(int(lo[1]), 0)
outside[y0:int(hi[1]), x0:int(hi[0])] = 28
check("nothing is drawn beyond the hand and its glow",
      bool(np.all(outside == 28)))

print("\nIt refuses rather than smearing when the hand is unusable:")
for label, bad in (
        ("a hand only a few pixels across", px(OPEN_PALM, scale=8.0)),
        ("landmarks containing NaN", np.full((21, 2), np.nan)),
        ("the wrong array shape", np.zeros((21, 3)))):
    frame = blank()
    gauntlet.draw(frame, bad, "Right")
    check(f"{label} leaves the frame untouched", bool(np.all(frame == 28)))

print("\nThe two sides of the hand are different objects:")
faces = {}
for hand in ("Right", "Left"):
    frame = blank()
    gauntlet.draw(frame, pts, hand)
    faces[hand] = frame.copy()


def has_stones(img):
    """A stone is the only thing on the glove that is vividly not red or gold."""
    b = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    r = img[:, :, 2].astype(int)
    green = (g > 170) & (r < 150) & (b < 150)
    violet = (b > 150) & (r > 150) & (g < 120)
    return bool(green.any() and violet.any())


# For a given hand shape one label is the back and the other the palm; which is
# which depends on the chirality, so ask rather than assume.
BACK = "Right" if has_stones(faces["Right"]) else "Left"
PALM = "Left" if BACK == "Right" else "Right"
check(f"one side carries the stones (here, the {BACK.lower()} hand)",
      has_stones(faces[BACK]))
check("and the other carries none of them", not has_stones(faces[PALM]))
differ = float(np.mean(np.any(faces["Right"] != faces["Left"], axis=2)))
print(f"      the two faces differ over {differ:.0%} of the frame")
check("they are different armour, not one drawing mirrored", differ > 0.05)
# The silhouette is the same hand either way: only what is drawn on it changes.
shape = {k: np.any(v != 28, axis=2) for k, v in faces.items()}
same = float(np.mean(shape["Right"] == shape["Left"]))
check(f"but both still fit the same hand ({same:.0%} of pixels)", same > 0.93)

print("\nThe pinch lights the stones:")
lit = {}
for charge in (0.0, 1.0):
    frame = blank()
    gauntlet.draw(frame, pts, BACK, charge=charge, now=0.0)
    lit[charge] = float(frame.mean())
print(f"      mean brightness {lit[0.0]:.1f} open -> {lit[1.0]:.1f} pinched")
check("closing the pinch brightens the frame", lit[1.0] > lit[0.0] * 1.02)
# ...and the palm, having no stones, has nothing to light up.
palm_lit = []
for charge in (0.0, 1.0):
    frame = blank()
    gauntlet.draw(frame, pts, PALM, charge=charge, now=0.0)
    palm_lit.append(float(frame.mean()))
check("the palm side ignores the charge, having no stones to light",
      palm_lit[0] == palm_lit[1])

print("\nIt has to keep up with the camera:")
big = np.full((1080, 1920, 3), 28, np.uint8)
hd = px(OPEN_PALM, scale=260.0, cx=900.0, cy=600.0)
gauntlet.draw(big, hd, "Right")            # warm up
started = time.perf_counter()
for _ in range(50):
    gauntlet.draw(big, hd, "Right")
ms = (time.perf_counter() - started) / 50 * 1000.0
print(f"      {ms:.2f} ms per frame at 1080p")
# A 30 fps loop has 33 ms to spend, so the overlay gets a small share of it.
# This bound is not decoration: drawn naively the bloom alone took 390 ms, and
# the first two attempts at fixing it landed at 9 ms. Measured at 4.5 ms.
check(f"draws in a small fraction of a frame ({ms:.2f} ms)", ms < 6.0)

print("\nIt stays out of the control path:")
check("on by default", Settings().gauntlet is True)
check("and can be turned off", Settings(gauntlet=False).gauntlet is False)
src = (pathlib.Path(__file__).resolve().parent.parent
       / "gesture_control" / "gauntlet.py").read_text(encoding="utf-8")
check("the module imports nothing that could move the mouse",
      "mouse" not in src and "Mouse" not in src)

print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
