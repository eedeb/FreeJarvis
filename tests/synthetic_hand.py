"""A fake hand for tests: landmark poses that point wherever you ask.

The synthetic camera projects a screen point through a known homography and
then bends it with barrel distortion and jitter, so anything fitted against it
has a ground truth to be measured against.
"""

import time

import numpy as np

from gesture_control import landmarks as lm
from gesture_control.mapping import _apply
from gesture_control.tracker import HandFrame

ASPECT = 1280 / 720

# Hand template in metric units, palm length = 1.0, y up-negative like image coords.
POINTING = {
    lm.WRIST: (0.0, 0.0), 1: (-0.35, -0.15), lm.THUMB_MCP: (-0.60, -0.40),
    lm.THUMB_IP: (-0.75, -0.65), lm.THUMB_TIP: (-0.85, -0.90),
    lm.INDEX_MCP: (-0.32, -0.95), lm.INDEX_PIP: (-0.36, -1.45),
    lm.INDEX_DIP: (-0.38, -1.75), lm.INDEX_TIP: (-0.40, -2.00),
    lm.MIDDLE_MCP: (0.0, -1.0), lm.MIDDLE_PIP: (0.02, -1.50), 11: (0.05, -1.20),
    lm.MIDDLE_TIP: (0.05, -0.90),
    lm.RING_MCP: (0.30, -0.93), lm.RING_PIP: (0.33, -1.35), 15: (0.32, -1.10),
    lm.RING_TIP: (0.30, -0.85),
    lm.PINKY_MCP: (0.58, -0.85), lm.PINKY_PIP: (0.60, -1.20), 19: (0.58, -1.00),
    lm.PINKY_TIP: (0.55, -0.80),
}
OPEN_PALM = dict(POINTING)
OPEN_PALM.update({lm.MIDDLE_TIP: (0.05, -2.05), 11: (0.04, -1.80),
                  lm.RING_TIP: (0.33, -1.95), 15: (0.32, -1.70),
                  lm.PINKY_TIP: (0.62, -1.70), 19: (0.60, -1.50),
                  lm.THUMB_TIP: (-1.10, -1.05)})
TWO_FINGERS = dict(POINTING)
TWO_FINGERS.update({lm.MIDDLE_TIP: (-0.15, -2.00), 11: (-0.10, -1.75),
                    lm.MIDDLE_PIP: (-0.05, -1.45)})

SCREEN = (2560, 1440)
H_SCREEN_TO_IMAGE = np.array([[-0.00026, 0.00002, 0.78],
                              [0.00001, 0.00030, 0.12],
                              [0.0, 0.0, 1.0]])
rng = np.random.default_rng(3)


def make_hand(target_px, pose=POINTING, pinch=None, scale=0.30, noise=0.0012,
              stamp=None, distortion=0.2, anchor="tip", tilt=(0.0, 0.0)):
    """A HandFrame whose index finger points at `target_px` on screen."""
    p = _apply(H_SCREEN_TO_IMAGE, np.asarray([target_px], dtype=float))[0]
    c = np.array([0.5, 0.5])
    d = p - c
    p = c + d * (1 + distortion * float(d @ d))
    p = p + rng.normal(0, noise, 2)

    tpl = np.zeros((21, 2))
    for i, xy in pose.items():
        tpl[i] = xy
    # Turning the hand: rotate the 3D pose, then re-project. Both the world
    # landmarks and the flattened image shape change together, the way they
    # would if you actually turned your hand towards a different part of the
    # screen.
    if tilt != (0.0, 0.0):
        rx, ry = np.radians(tilt[0]), np.radians(tilt[1])
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        rot = (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
               @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]))
    else:
        rot = np.eye(3)

    # Which part of the hand is placed on the target: the fingertip for finger
    # aiming, the palm centre for palm aiming.
    shape = np.zeros((21, 3))
    shape[:, :2] = tpl - tpl.mean(axis=0)
    flat = (shape @ rot.T)[:, :2] + tpl.mean(axis=0)      # turned, then seen flat
    ref = (flat[lm.INDEX_TIP] if anchor == "tip"
           else flat[list(lm.PALM)].mean(axis=0))
    metric = (flat - ref) * scale + p * np.array([ASPECT, 1.0])
    # Move the whole thumb, not just its tip. Teleporting the tip alone
    # changes the length of the thumb's bones, which a real hand cannot do --
    # and which the plausibility check rightly rejects as a lost track.
    _THUMB = [lm.THUMB_MCP, lm.THUMB_IP, lm.THUMB_TIP]
    if pinch in ("index", "middle"):
        target = metric[lm.INDEX_TIP if pinch == "index" else lm.MIDDLE_TIP]
        delta = target + np.array([0.05, 0.12]) * scale - metric[lm.THUMB_TIP]
        metric[_THUMB] += delta
    norm = metric / np.array([ASPECT, 1.0])

    # World landmarks: the same pose in metres about the hand's own centre,
    # with a palm roughly 9 cm long. Planar (z = 0) is enough here -- the pose
    # tests only ever measure angles and distances between joints.
    world = np.zeros((21, 3))
    world[:, :2] = (tpl - tpl.mean(axis=0)) * 0.09
    world = world @ rot.T
    if pinch in ("index", "middle"):
        anchor = world[lm.INDEX_TIP if pinch == "index" else lm.MIDDLE_TIP]
        shift = anchor + np.array([0.004, 0.010, 0.0]) - world[lm.THUMB_TIP]
        world[[lm.THUMB_MCP, lm.THUMB_IP, lm.THUMB_TIP]] += shift

    return HandFrame(stamp=stamp if stamp is not None else time.monotonic(),
                     norm=norm, metric=metric, handedness="Right",
                     confidence=0.95, world=world)


class FakeTracker:
    def __init__(self):
        self.hand = None
        self.fps = 30.0
        self.error = None
    def latest(self): return self.hand
    def latest_preview(self): return None
    def raise_if_failed(self): pass




# A closed fist: every finger folded back towards the palm.
FIST = dict(POINTING)
FIST.update({
    lm.INDEX_PIP: (-0.34, -1.40), 7: (-0.30, -1.10), lm.INDEX_TIP: (-0.26, -0.85),
    lm.MIDDLE_PIP: (0.02, -1.45), 11: (0.05, -1.15), lm.MIDDLE_TIP: (0.06, -0.88),
    lm.RING_PIP: (0.32, -1.36), 15: (0.34, -1.08), lm.RING_TIP: (0.33, -0.84),
    lm.PINKY_PIP: (0.60, -1.22), 19: (0.61, -0.99), lm.PINKY_TIP: (0.58, -0.79),
    lm.THUMB_TIP: (-0.30, -0.75),
    lm.INDEX_MCP: (-0.33, -0.91), lm.MIDDLE_MCP: (0.00, -0.96),
    lm.RING_MCP: (0.30, -0.90), lm.PINKY_MCP: (0.57, -0.82),
})


# Fingers spread wide, as when you hold a palm up to a webcam. The gaps between
# the digits are the point: armour that fills them in renders as a mitten, and
# nothing in the closer-fingered poses above catches that.
SPLAYED = dict(OPEN_PALM)
SPLAYED.update({
    lm.INDEX_MCP: (-0.42, -0.90), lm.INDEX_PIP: (-0.62, -1.42),
    7: (-0.72, -1.74), lm.INDEX_TIP: (-0.80, -2.02),
    lm.MIDDLE_MCP: (-0.05, -1.00), lm.MIDDLE_PIP: (-0.10, -1.55),
    11: (-0.12, -1.85), lm.MIDDLE_TIP: (-0.14, -2.15),
    lm.RING_MCP: (0.32, -0.95), lm.RING_PIP: (0.46, -1.45),
    15: (0.54, -1.74), lm.RING_TIP: (0.60, -2.00),
    lm.PINKY_MCP: (0.62, -0.85), lm.PINKY_PIP: (0.92, -1.22),
    19: (1.06, -1.46), lm.PINKY_TIP: (1.18, -1.68),
    lm.THUMB_MCP: (-0.78, -0.30), lm.THUMB_IP: (-1.10, -0.55),
    lm.THUMB_TIP: (-1.38, -0.78),
})
