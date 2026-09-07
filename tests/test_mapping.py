"""Synthetic check of the calibration fit: does it recover a known mapping?"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from gesture_control.mapping import AimSample, fit_pointer_map, fit_homography, _apply

rng = np.random.default_rng(7)
SCREEN = (2560, 1440)

# A plausible camera: screen -> image plane homography (off-axis, slight tilt).
H_true = np.array([
    [-0.00026, 0.00002, 0.78],
    [0.00001, 0.00030, 0.12],
    [0.00000, 0.00000, 1.0],
])


def grid(cols, rows, margin=0.09):
    xs = np.linspace(margin, 1 - margin, cols) * SCREEN[0]
    ys = np.linspace(margin, 1 - margin, rows) * SCREEN[1]
    return np.array([[x, y] for y in ys for x in xs])


def simulate(targets, distortion=0.25, noise=0.0015, depth=0.0):
    """Screen point -> where the fingertip actually lands in the image."""
    pts = _apply(H_true, targets)
    c = np.array([0.5, 0.5])
    d = pts - c
    r2 = (d ** 2).sum(1, keepdims=True)
    pts = c + d * (1 + distortion * r2)          # barrel distortion
    pts = pts + rng.normal(0, noise, pts.shape)  # tracking jitter
    samples = []
    for i, p in enumerate(pts):
        dirv = np.array([0.0, -1.0]) + rng.normal(0, 0.05, 2)
        dirv /= np.linalg.norm(dirv)
        scale = 0.22 * (1 + depth * rng.normal())
        samples.append(AimSample(tuple(p), tuple(dirv), scale))
    return samples


ok = True


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


def report(name, cols, rows, budget_px, **kw):
    """Fit on a grid, then measure error at 300 points the fit never saw."""
    tgt = grid(cols, rows)
    m = fit_pointer_map(simulate(tgt, **kw), tgt, SCREEN)
    test_tgt = np.column_stack([rng.uniform(0.05, 0.95, 300) * SCREEN[0],
                                rng.uniform(0.05, 0.95, 300) * SCREEN[1]])
    err = np.linalg.norm(m.predict_many(simulate(test_tgt, **kw)) - test_tgt, axis=1)
    rms, p95 = float(np.sqrt((err ** 2).mean())), float(np.percentile(err, 95))
    loocv = m.meta["loocv_rms_px"]
    print(f"{name:16s} n={len(tgt):3d}  model={m.meta['feature_set']:6s} "
          f"lam={m.meta['ridge_lambda']:7.3f}  "
          f"loocv={'   n/a' if loocv is None else format(loocv, '6.1f')}px  "
          f"held-out rms={rms:6.1f}px  p95={p95:6.1f}px")
    check(f"    {name}: p95 error under {budget_px} px", p95 < budget_px)
    return m


print("screen:", SCREEN, "-> px errors below\n")
report("2x2 (min)", 2, 2, 45)
report("3x3", 3, 3, 45)
report("4x3 (default)", 4, 3, 35)
report("5x4", 5, 4, 35)
report("4x3 no-distort", 4, 3, 30, distortion=0.0)
heavy = report("4x3 heavy-dist", 4, 3, 60, distortion=0.6)
report("4x3 noisy", 4, 3, 90, noise=0.006)
report("4x3 depth-vary", 4, 3, 40, depth=0.15)

print()
check("heavy distortion turns the correction stage on",
      heavy.meta["feature_set"] != "none")
check("...and it beats the plain homography out of sample",
      heavy.meta["loocv_rms_px"] < heavy.meta["homography_only_loocv_px"])
clean_model = report("4x3 clean recheck", 4, 3, 30, distortion=0.0)
check("a clean camera leaves the correction stage off",
      clean_model.meta["feature_set"] == "none")

# Degenerate inputs should raise, not silently produce garbage.
print()
try:
    fit_pointer_map(simulate(grid(3, 1))[:3], grid(3, 1)[:3], SCREEN)
    check("fewer than 4 calibration points is rejected", False)
except ValueError:
    check("fewer than 4 calibration points is rejected", True)

# Exact recovery when there is no distortion or noise at all.
tgt = grid(4, 3)
clean = simulate(tgt, distortion=0.0, noise=0.0)
H = fit_homography(np.array([s.tip for s in clean]), tgt)
err = np.linalg.norm(_apply(H, np.array([s.tip for s in clean])) - tgt, axis=1)
check(f"noiseless homography recovers the true mapping (max err {err.max():.1e} px)",
      err.max() < 1e-6)

# Round-trip through the calibration file.
import tempfile, pathlib as _pl
from gesture_control.mapping import PointerMap
m = fit_pointer_map(simulate(tgt, distortion=0.6), tgt, SCREEN)
f = _pl.Path(tempfile.mkdtemp()) / "calibration.json"
m.save(f)
reloaded = PointerMap.load(f)
check("save/load reproduces identical predictions",
      np.allclose(m.predict_many(clean), reloaded.predict_many(clean)))
check("the saved file is strictly valid JSON",
      "Infinity" not in f.read_text(encoding="utf-8"))

print()
print("RESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
