"""The ray model: a stick out of the palm, and where it lands on the screen.

The hand is simulated properly here -- placed at a real 3D position in front of
a camera, turned so that a ray out of its palm strikes a given point on a
screen of known size and position, then projected into the image the way a
camera would see it. So the fit is being asked to recover a geometry it was
never told, from measurements alone.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from gesture_control.mapping import (AimSample, fit_pointer_map, fit_ray_model,
                                     _ray_hits, _apply_affine, _fit_affine)

ok = True


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


# --- the truth the fit has to rediscover ---------------------------------
SCREEN = (1920, 1080)
SCREEN_W_M = 0.60                 # a 27-inch monitor, roughly
SCREEN_H_M = SCREEN_W_M * SCREEN[1] / SCREEN[0]
PLANE_Z = -0.05                   # screen plane sits just behind the camera
SCREEN_TOP_M = 0.02               # camera is mounted above the screen
FOCAL = 1.35                      # in image-height units
PALM_M = 0.092                    # true palm length
rng = np.random.default_rng(11)


def screen_to_world(px):
    """Screen pixel -> a point in camera coordinates, in metres."""
    fx = px[0] / SCREEN[0] - 0.5
    fy = px[1] / SCREEN[1]
    return np.array([fx * SCREEN_W_M, SCREEN_TOP_M + fy * SCREEN_H_M, PLANE_Z])


def sample_for(target_px, hand_xyz, noise=0.0, drop_z=False):
    """A hand at `hand_xyz` with its palm turned to point at `target_px`."""
    hand = np.asarray(hand_xyz, dtype=float)
    ray = screen_to_world(target_px) - hand
    normal = ray / np.linalg.norm(ray)
    if noise:
        normal = normal + rng.normal(0, noise, 3)
        normal = normal / np.linalg.norm(normal)

    # Project the palm centre into the image the way a pinhole camera would.
    u = 0.5 + FOCAL * hand[0] / hand[2]
    v = 0.5 + FOCAL * hand[1] / hand[2]
    apparent = FOCAL * PALM_M / hand[2]
    if noise:
        u += rng.normal(0, noise * 0.3)
        v += rng.normal(0, noise * 0.3)
        apparent *= 1.0 + rng.normal(0, noise * 0.3)
    if drop_z:
        normal = np.array([normal[0], normal[1], -1.0])
        normal = normal / np.linalg.norm(normal)
    return AimSample(tip=(u, v), direction=(normal[0], normal[1]),
                     scale=PALM_M, normal=tuple(normal), apparent=apparent)


def grid(cols=4, rows=3, margin=0.09):
    xs = np.linspace(margin, 1 - margin, cols) * SCREEN[0]
    ys = np.linspace(margin, 1 - margin, rows) * SCREEN[1]
    return np.array([[x, y] for y in ys for x in xs])


targets = grid()

print("A hand held in one place, turned to aim (a stick out of the palm):")
still = [sample_for(t, (0.0, 0.25, 0.55)) for t in targets]
focal, plane_z, affine = fit_ray_model(still, targets)
fitted = _apply_affine(affine, _ray_hits(still, focal, plane_z))
err = np.linalg.norm(fitted - targets, axis=1)
print(f"      recovered focal {focal:.2f} (true {FOCAL}), "
      f"screen depth {plane_z:+.3f}m (true {PLANE_Z:+.3f})")
print(f"      fit error {err.max():.2f}px max")
check("the ray geometry is recovered essentially exactly", err.max() < 2.0)

print("\nThe same hand moved around as well as turned:")
mixed, mixed_targets = [], []
for i, t in enumerate(targets):
    where = (0.10 * np.cos(i * 1.3), 0.22 + 0.05 * np.sin(i * 0.9),
             0.50 + 0.06 * np.cos(i * 0.7))
    mixed.append(sample_for(t, where))
    mixed_targets.append(t)
mixed_targets = np.array(mixed_targets)
model = fit_pointer_map(mixed, mixed_targets, SCREEN)
print(f"      chosen model: {model.meta['aims_by']}, "
      f"accuracy {model.meta['loocv_rms_px']:.1f}px")
check("moving and turning together is still fitted",
      model.meta["loocv_rms_px"] < SCREEN[1] * 0.02)
check("and it picks the ray", "ray" in model.meta["aims_by"])

# Held-out points, at hand positions never seen during calibration.
probe = np.column_stack([rng.uniform(0.08, 0.92, 250) * SCREEN[0],
                         rng.uniform(0.08, 0.92, 250) * SCREEN[1]])
held = [sample_for(t, (0.08 * np.cos(i), 0.24, 0.52 + 0.04 * np.sin(i)))
        for i, t in enumerate(probe)]
err = np.linalg.norm(model.predict_many(held) - probe, axis=1)
print(f"      unseen points, unseen hand positions: "
      f"rms {err.mean():.1f}px, p95 {np.percentile(err, 95):.1f}px")
check("it generalises to hand positions it never saw",
      np.percentile(err, 95) < SCREEN[1] * 0.03)

print("\nThe catch: a ray has a lever arm, and it multiplies angular error.")
print("      the hand sits ~0.55m from the screen, so a degree of error in")
print("      which way the palm faces becomes about a centimetre on screen.")
px_per_metre = SCREEN[0] / SCREEN_W_M
errors = []
for noise in (0.005, 0.01, 0.02, 0.04):
    noisy = [sample_for(t, (0.06 * np.cos(i), 0.23, 0.53), noise=noise)
             for i, t in enumerate(targets)]
    m = fit_pointer_map(noisy, targets, SCREEN)
    test = [sample_for(t, (0.06 * np.cos(i), 0.23, 0.53), noise=noise)
            for i, t in enumerate(probe)]
    e = float(np.linalg.norm(m.predict_many(test) - probe, axis=1).mean())
    errors.append(e)
    predicted = noise * 0.55 * px_per_metre       # angle x distance x scale
    print(f"      {np.degrees(noise):4.1f}deg of normal error -> {e:6.1f}px "
          f"(lever arm alone predicts {predicted:.0f}px)")
check("error grows in proportion to the angular noise, as the geometry says",
      1.4 < errors[-1] / errors[-2] < 3.0 and 1.4 < errors[1] / errors[0] < 3.0)

# Which model is better is therefore not a matter of taste: it depends on
# whether the hand's angle or its position is measured more precisely.
print("\n      So a moving hand beats a turning one whenever the angle is noisy:")
for noise in (0.005, 0.03):
    turning = [sample_for(t, (0.02 * np.cos(i), 0.23, 0.53), noise=noise)
               for i, t in enumerate(targets)]
    moving = [sample_for(t, tuple(screen_to_world(t) * 0.55
                                  + np.array([0, 0, 0.55])), noise=noise)
              for t in targets]
    a = fit_pointer_map(turning, targets, SCREEN).meta["loocv_rms_px"]
    b = fit_pointer_map(moving, targets, SCREEN).meta["loocv_rms_px"]
    print(f"      at {np.degrees(noise):4.1f}deg: turning in place {a:6.1f}px, "
          f"moving the hand {b:6.1f}px")

print("\nAgainst the flat models on the same data:")
flat = [AimSample(tip=s.tip, direction=s.direction, scale=s.scale) for s in still]
flat_model = fit_pointer_map(flat, targets, SCREEN)
ray_model = fit_pointer_map(still, targets, SCREEN)
print(f"      without the 3D measurements: {flat_model.meta['aims_by']:26s} "
      f"{flat_model.meta['loocv_rms_px']:6.1f}px")
print(f"      with them:                   {ray_model.meta['aims_by']:26s} "
      f"{ray_model.meta['loocv_rms_px']:6.1f}px")
check("the ray model beats the flat ones on ray-shaped data",
      ray_model.meta["loocv_rms_px"] < flat_model.meta["loocv_rms_px"])

print("\nSaving and reloading a ray calibration:")
import json, tempfile
from gesture_control.mapping import PointerMap
path = pathlib.Path(tempfile.mkdtemp()) / "c.json"
ray_model.save(path)
reloaded = PointerMap.load(path)
check("predictions survive the round trip",
      np.allclose(ray_model.predict_many(still), reloaded.predict_many(still)))
check("the file is valid JSON with the ray model in it",
      json.loads(path.read_text())["src"] == "ray")

print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
