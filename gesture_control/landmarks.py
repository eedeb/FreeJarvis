"""Landmark indices and the hand features the rest of the app reasons about.

MediaPipe returns landmarks two ways.

* **Image coordinates**, normalised to the frame: x in [0, 1] across the width,
  y in [0, 1] down the height.  These say *where the hand is*, so the pointing
  mapping is built on them.
* **World coordinates**, in metres, relative to the hand's own centre.  These
  say *what shape the hand is in*, independent of where the camera is.

Every pose question -- is this finger straight, are these two fingertips
touching -- is answered in world coordinates, and that matters more than it
sounds.  Pointing at a screen means pointing roughly at the camera, so the hand
is heavily foreshortened in the image: an outstretched finger projects to a
stub, and a thumb held well clear of the index finger projects to almost the
same spot.  Judged in 2D that reads as a curled finger and a pinch that is not
happening.  In 3D neither illusion survives.

Image coordinates are still anisotropic (a frame is wider than it is tall), so
the few places that do measure in 2D use "metric" coordinates, where x has been
multiplied by the frame aspect ratio.
"""

from __future__ import annotations

import numpy as np

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# (mcp, pip, tip) per finger: the three joints whose alignment says whether the
# finger is straight.
_FINGER_CHAIN = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_TIP),
}

# How straight counts as straight: cosine of the angle between the two
# segments.  A fully extended finger is near 1.0; a curled one goes negative as
# the tip folds back towards the palm.
_STRAIGHT_COS = {
    "index": 0.55,
    "middle": 0.55,
    "ring": 0.5,
    "pinky": 0.45,
    "thumb": 0.75,   # the thumb barely bends, so it needs a tighter test
}

CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def hand_scale(points: np.ndarray) -> float:
    """Palm length (wrist -> middle knuckle): the unit every ratio uses.

    In world coordinates this is a real, roughly constant length for a given
    hand, so ratios measured against it mean the same thing at any distance
    from the camera and at any angle to it.
    """
    return float(np.linalg.norm(points[MIDDLE_MCP] - points[WRIST]))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def finger_extended(points: np.ndarray, finger: str) -> bool:
    """True when a finger is straight.

    Measured as the alignment of the knuckle-to-middle-joint segment with the
    middle-joint-to-tip segment, which is a property of the finger itself: it
    does not care how the hand is turned, or how the camera sees it.  The
    older test compared distances from the wrist, and a finger aimed at the
    camera fails that one however straight it is.
    """
    mcp, pip, tip = _FINGER_CHAIN[finger]
    lower = _unit(points[pip] - points[mcp])
    upper = _unit(points[tip] - points[pip])
    if not lower.any() or not upper.any():
        return False
    return bool(float(lower @ upper) > _STRAIGHT_COS[finger])


def thumb_extended(points: np.ndarray) -> bool:
    return finger_extended(points, "thumb")


def bone_lengths(points: np.ndarray) -> np.ndarray:
    """Lengths of the bones the pinch depends on.

    A hand cannot change shape: these are fixed for a given person, so when
    the tracker reports them changing it has lost the hand rather than the
    hand having moved.  That makes them a way to spot an untrustworthy frame
    from the inside, without needing to know what the right answer was.
    """
    pairs = ((INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
             (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP))
    return np.array([float(np.linalg.norm(points[a] - points[b])) for a, b in pairs])


def pinch_ratio(points: np.ndarray, tip: int) -> float:
    """Thumb-to-fingertip gap as a fraction of palm length (0 = touching)."""
    scale = hand_scale(points)
    if scale < 1e-9:
        return float("inf")
    return float(np.linalg.norm(points[THUMB_TIP] - points[tip]) / scale)


def fingers_spread(points: np.ndarray) -> float:
    """Mean gap between neighbouring fingertips, in palm lengths.

    Distinguishes a deliberately splayed hand from one that merely has its
    fingers straight, which is what makes the open-palm gesture hard to trigger
    by accident.
    """
    scale = hand_scale(points)
    if scale < 1e-9:
        return 0.0
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    gaps = [np.linalg.norm(points[a] - points[b])
            for a, b in zip(tips, tips[1:])]
    return float(np.mean(gaps) / scale)


PALM = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
_CURL_FINGERS = ("index", "middle", "ring", "pinky")


def palm_center(points: np.ndarray) -> np.ndarray:
    """The wrist and four knuckles, averaged.

    The steadiest point on a hand, for two reasons.  Averaging five landmarks
    divides their independent jitter by roughly the square root of five, and
    unlike anything on a finger it stays put when the fingers move -- so
    closing the hand to click does not drag the aim along with it.
    """
    return points[list(PALM)].mean(axis=0)


def curl(points: np.ndarray) -> float:
    """How open the hand is: about 1 for a flat palm, negative for a fist.

    The mean straightness of the four fingers.  Four joints-worth of evidence
    agreeing makes this a far blunter, safer instrument than the distance
    between two fingertips, which is all a pinch ever was.
    """
    total = 0.0
    for finger in _CURL_FINGERS:
        mcp, pip, tip = _FINGER_CHAIN[finger]
        lower = _unit(points[pip] - points[mcp])
        upper = _unit(points[tip] - points[pip])
        total += float(lower @ upper) if lower.any() and upper.any() else 0.0
    return total / len(_CURL_FINGERS)


def palm_normal(world: np.ndarray) -> np.ndarray:
    """Unit vector out of the back of the hand.

    Which way the palm faces, from the plane through the wrist and the index
    and pinky knuckles.  Where the hand *is* does most of the work of aiming;
    this says which way it is turned, and the calibration fit uses it to tell
    those two apart.
    """
    a = world[INDEX_MCP] - world[WRIST]
    b = world[PINKY_MCP] - world[WRIST]
    return _unit(np.cross(a, b))


def point_direction(metric: np.ndarray) -> np.ndarray:
    """Unit vector along the index finger in the *image* plane.

    This one stays in 2D on purpose: it is a feature of the pointing mapping,
    which relates image positions to screen positions, so it has to live in the
    same space as the rest of that fit.
    """
    return _unit(metric[INDEX_TIP] - metric[INDEX_MCP])
