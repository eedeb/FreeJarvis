"""The nano gauntlet, drawn onto the preview from the hand landmarks.

Purely cosmetic. Nothing here is read by the control loop, nothing here feeds
back into it, and deleting the module would change no behaviour.

There is no mesh and no bitmap, on purpose. Flat art cannot follow a bending
finger: to use a real render it would have to be cut into one sprite per bone
and warped piece by piece, and a rigged model would need a GL context plus a
per-frame fit from 21 landmarks onto a skeleton it was never authored for.
Tracing the armour straight from the landmarks pins every plate to the joint it
armours, so the fingers bend correctly because they are never posed at all.

The two sides of the hand are drawn as different objects, which is most of what
makes it read as a worn glove rather than a decal. The back carries the stones,
the octagonal boss and steel knuckle housings; the palm carries grip pads, a
gold seam and no stones at all. Which one you get follows the hand you are
actually holding up.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import landmarks as lm

# Stark plate, in BGR. The crimson is deliberately dark: the reference armour
# is nearly maroon in shadow and only reads as red where the light catches it.
# A brighter base makes the whole hand look like plastic.
RED_DEEP = (30, 18, 96)
RED = (40, 26, 140)
RED_LIT = (68, 54, 196)
GOLD_DARK = (28, 96, 140)
GOLD = (58, 176, 232)
GOLD_LIT = (128, 220, 252)
STEEL_DARK = (74, 72, 80)
STEEL = (116, 116, 124)
STEEL_LIT = (162, 162, 170)
SEAM = (14, 12, 20)

# Across the knuckles index to pinky, then the thumb, then the boss on the back
# of the hand -- the order they sit in on the nano gauntlet.
STONES = (
    (232, 74, 208),     # power
    (250, 190, 70),     # space
    (60, 50, 240),      # reality
    (40, 140, 250),     # soul
    (110, 240, 80),     # time
    (40, 225, 255),     # mind
)

# Where the light comes from, in image space (y runs down). Everything that
# looks rounded rather than flat is one offset highlight along this direction.
LIGHT = np.array([-0.48, -0.88])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

# The four joints of each digit, and the half-width of the armour at each one
# as a fraction of palm length. Tapering is what stops a finger reading as a
# tube: real plate narrows towards the fingertip.
_DIGITS = (
    ((lm.THUMB_CMC, lm.THUMB_MCP, lm.THUMB_IP, lm.THUMB_TIP),
     (0.235, 0.205, 0.175, 0.150)),
    ((lm.INDEX_MCP, lm.INDEX_PIP, lm.INDEX_DIP, lm.INDEX_TIP),
     (0.235, 0.195, 0.163, 0.133)),
    ((lm.MIDDLE_MCP, lm.MIDDLE_PIP, 11, lm.MIDDLE_TIP),
     (0.245, 0.205, 0.170, 0.138)),
    ((lm.RING_MCP, lm.RING_PIP, 15, lm.RING_TIP),
     (0.225, 0.190, 0.157, 0.127)),
    ((lm.PINKY_MCP, lm.PINKY_PIP, 19, lm.PINKY_TIP),
     (0.205, 0.172, 0.142, 0.117)),
)

_PLATE_KEYS = [lm.WRIST, lm.THUMB_CMC, lm.INDEX_MCP, lm.MIDDLE_MCP,
               lm.RING_MCP, lm.PINKY_MCP]


# --------------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.array([1.0, 0.0])


def _perp(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])


def _mix(colour, other, f: float):
    """Blend towards another colour.

    Cheaper than a real shading model, and at the size a hand occupies in the
    preview it reads the same.
    """
    return tuple(float(c) * (1.0 - f) + float(o) * f for c, o in zip(colour, other))


def _fill(img, points, colour) -> None:
    cv2.fillConvexPoly(img, np.asarray(points, dtype=np.int32), colour, cv2.LINE_AA)


def _flat(img, points, colour) -> None:
    """A fill with no antialiasing, for interior plate that something else draws
    over. The smooth edge is only worth paying for on a visible one."""
    cv2.fillConvexPoly(img, np.asarray(points, dtype=np.int32), colour)


def _rim(img, points, colour, width: int = 1) -> None:
    cv2.polylines(img, [np.asarray(points, dtype=np.int32)], True, colour,
                  max(width, 1), cv2.LINE_AA)


def _stroke(img, a, b, colour, width: int = 1) -> None:
    cv2.line(img, (int(round(a[0])), int(round(a[1]))),
             (int(round(b[0])), int(round(b[1]))), colour, max(width, 1),
             cv2.LINE_AA)


def _quad(a, b, wa: float, wb: float) -> np.ndarray:
    """A tapered plate spanning two joints."""
    n = _perp(_unit(b - a))
    return np.array([a + n * wa, b + n * wb, b - n * wb, a - n * wa])


def _ngon(centre, r: float, sides: int, along) -> np.ndarray:
    """A regular polygon with one axis aligned to `along`.

    Hard edges are most of what separates the gauntlet from a cartoon glove, so
    the housings, caps and boss are polygons rather than circles.
    """
    base = float(np.arctan2(along[1], along[0]))
    ang = base + np.arange(sides) * (2.0 * np.pi / sides) + np.pi / sides
    return np.stack([centre[0] + r * np.cos(ang),
                     centre[1] + r * np.sin(ang)], axis=1)


def _shrink(poly, centre, f: float, offset=None):
    out = (np.asarray(poly) - centre) * f + centre
    return out if offset is None else out + offset


# --------------------------------------------------------------------------
# Fingers
# --------------------------------------------------------------------------

def _bone(img, a, b, wa: float, wb: float, back: bool) -> None:
    """One armoured phalanx.

    The dorsal side is ribbed the way the reference is, two raised bands per
    segment. The palmar side gets a grip panel instead, which is what the
    inside of a glove actually has and what keeps the two sides distinct even
    where the silhouette is identical.
    """
    d = _unit(b - a)
    n = _perp(d)
    _fill(img, _quad(a, b, wa * 1.14, wb * 1.14), SEAM)
    _flat(img, _quad(a, b, wa, wb), RED_DEEP)
    side = 1.0 if float(n @ LIGHT) > 0 else -1.0
    # A lit strip along the edge facing the light: the only thing making a flat
    # polygon read as a curved plate.
    _flat(img, _quad(a + n * side * wa * 0.46, b + n * side * wb * 0.46,
                     wa * 0.42, wb * 0.42), RED)
    _flat(img, _quad(a + n * side * wa * 0.74, b + n * side * wb * 0.74,
                     wa * 0.20, wb * 0.20), RED_LIT)

    if back:
        for f in (0.34, 0.66):
            p = a + (b - a) * f
            w = wa + (wb - wa) * f
            _stroke(img, p - n * w * 0.96, p + n * w * 0.96, SEAM,
                    max(int(w * 0.16), 1))
            _stroke(img, p + d * w * 0.13 - n * w * 0.92,
                    p + d * w * 0.13 + n * w * 0.92, RED_LIT,
                    max(int(w * 0.08), 1))
    else:
        mid = a + (b - a) * 0.5
        w = (wa + wb) * 0.5
        _fill(img, _quad(a + (b - a) * 0.24, a + (b - a) * 0.76,
                         w * 0.58, w * 0.58), SEAM)
        _flat(img, _quad(a + (b - a) * 0.28, a + (b - a) * 0.72,
                         w * 0.46, w * 0.46), STEEL_DARK)
        _stroke(img, mid - n * w * 0.32, mid + n * w * 0.32, STEEL_DARK, 1)


def _band(img, before, joint, after, w: float, back: bool) -> None:
    """The articulation across a joint: a band, not a bead.

    A circle centred on a joint always reads as a bead threaded on a string, at
    any size. Plate armour hinges across the digit, so the band follows the
    joint's own axis -- the bisector of the two bones meeting there.
    """
    d = _unit(_unit(after - joint) - _unit(before - joint))
    if not d.any():
        d = _perp(_unit(after - before))
    n = _perp(d)
    quad = np.array([joint + n * w * 0.98 + d * w * 0.20,
                     joint - n * w * 0.98 + d * w * 0.20,
                     joint - n * w * 0.98 - d * w * 0.20,
                     joint + n * w * 0.98 - d * w * 0.20])
    _fill(img, quad, SEAM)
    if back:
        _flat(img, _shrink(quad, joint, 0.84), RED_DEEP)
        _flat(img, _shrink(quad, joint, 0.52, LIGHT * w * 0.10), GOLD_DARK)
    else:
        _flat(img, _shrink(quad, joint, 0.84), GOLD_DARK)
        _flat(img, _shrink(quad, joint, 0.52, LIGHT * w * 0.10), GOLD)


def _cap(img, p, along, r: float, back: bool) -> None:
    """The fingertip. No joint past it to hinge against, so it gets a cap."""
    shell = _ngon(p, r * 0.98, 6, along)
    _fill(img, shell, SEAM)
    body = (RED_DEEP, RED, RED_LIT) if back else (GOLD_DARK, GOLD, GOLD_LIT)
    _flat(img, _shrink(shell, p, 0.86), body[0])
    _flat(img, _shrink(shell, p, 0.66, LIGHT * r * 0.18), body[1])
    _flat(img, _shrink(shell, p, 0.30, LIGHT * r * 0.28), body[2])


# --------------------------------------------------------------------------
# The two faces of the hand
# --------------------------------------------------------------------------

def _hull(pts, unit: float):
    hand = pts[_PLATE_KEYS]
    centre = hand.mean(axis=0)
    # Push each landmark away from the centre so the plate covers the hand
    # rather than joining its joints, then let the hull round off the shape.
    outer = np.array([p + _unit(p - centre) * unit * 0.30 for p in hand])
    return cv2.convexHull(outer.astype(np.float32)).reshape(-1, 2), centre


def _stone(img, p, r: float, colour, charge: float, pulse: float,
           sides: int, along) -> None:
    """A stone in its housing.

    The housing is what sells it as set into metal rather than painted on, so
    it gets as many layers as the stone does.
    """
    seat = _ngon(p, r * 1.62, sides, along)
    _fill(img, seat, SEAM)
    _flat(img, _shrink(seat, p, 0.90), GOLD_DARK)
    _flat(img, _shrink(seat, p, 0.74, LIGHT * r * 0.14), GOLD)
    _flat(img, _shrink(seat, p, 0.60), GOLD_DARK)
    lit = _mix(colour, (255, 255, 255), 0.30 * charge)
    c = (int(round(p[0])), int(round(p[1])))
    cv2.circle(img, c, max(int(r), 2), _mix(lit, (0, 0, 0), 0.35), -1, cv2.LINE_AA)
    cv2.circle(img, c, max(int(r * 0.86), 1),
               _mix(lit, (0, 0, 0), 0.22 * (1.0 - pulse)), -1, cv2.LINE_AA)
    off = (int(round(p[0] + LIGHT[0] * r * 0.30)),
           int(round(p[1] + LIGHT[1] * r * 0.30)))
    cv2.circle(img, off, max(int(r * 0.40), 1), _mix(lit, (255, 255, 255), 0.70),
               -1, cv2.LINE_AA)


def _dorsal(img, pts, unit: float, hull, centre):
    """The back of the hand: red plate, steel accents, the mind-stone boss."""
    up = _unit(pts[lm.MIDDLE_MCP] - pts[lm.WRIST])
    n = _perp(up)
    _fill(img, hull, SEAM)
    _flat(img, _shrink(hull, centre, 0.96), GOLD_DARK)
    _flat(img, _shrink(hull, centre, 0.90), RED_DEEP)
    _flat(img, _shrink(hull, centre, 0.84, LIGHT * unit * 0.05), RED)
    _flat(img, _shrink(hull, centre, 0.62, LIGHT * unit * 0.13), RED_LIT)
    _flat(img, _shrink(hull, centre, 0.50, LIGHT * unit * 0.10), RED)

    # Steel accent panels down either side, as on the reference bracer.
    for s in (-1.0, 1.0):
        edge = centre + n * s * unit * 0.40
        _fill(img, [edge + up * unit * 0.28, edge - up * unit * 0.32,
                    edge - up * unit * 0.26 - n * s * unit * 0.11,
                    edge + up * unit * 0.24 - n * s * unit * 0.11], STEEL_DARK)

    return centre - up * unit * 0.13


def _boss(img, p, unit: float, up, charge: float, pulse: float) -> None:
    """The octagonal housing on the back of the hand, and the mind stone."""
    shell = _ngon(p, unit * 0.26, 8, up)
    _fill(img, shell, SEAM)
    _flat(img, _shrink(shell, p, 0.92), GOLD_DARK)
    _flat(img, _shrink(shell, p, 0.80, LIGHT * unit * 0.02), GOLD)
    _flat(img, _shrink(shell, p, 0.56), GOLD_DARK)
    _rim(img, _shrink(shell, p, 0.68), GOLD_LIT, 1)
    _stone(img, p, unit * 0.092, STONES[5], charge, pulse, 6, up)


def _knuckle_bar(img, pts, unit: float, centre):
    """The housing the four knuckle stones are set into.

    On the reference they share one bar across the knuckles rather than sitting
    in four separate bezels, and that single span is most of what makes the
    silhouette recognisable. It still tracks the hand: the bar is defined by
    the index and pinky knuckles, so it turns and shortens as they do.
    """
    seats = [pts[k] + _unit(centre - pts[k]) * unit * 0.10
             for k in (lm.INDEX_MCP, lm.MIDDLE_MCP, lm.RING_MCP, lm.PINKY_MCP)]
    a, b = seats[0], seats[-1]
    d = _unit(b - a)
    n = _perp(d)
    w = unit * 0.19
    a, b = a - d * unit * 0.15, b + d * unit * 0.15
    quad = np.array([a + n * w, b + n * w, b - n * w, a - n * w])
    _fill(img, quad, SEAM)
    _flat(img, _shrink(quad, (a + b) / 2.0, 0.92), STEEL_DARK)
    _flat(img, _shrink(quad, (a + b) / 2.0, 0.74, LIGHT * unit * 0.03), STEEL)
    _rim(img, quad, GOLD_DARK, 1)
    return seats, n


def _palmar(img, pts, unit: float, hull, centre) -> None:
    """The palm: plainer plate, a gold seam, grip pads, no stones.

    Different enough that turning your hand over is unmistakable, which is the
    whole reason the two sides are drawn separately.
    """
    up = _unit(pts[lm.MIDDLE_MCP] - pts[lm.WRIST])
    n = _perp(up)
    _fill(img, hull, SEAM)
    _flat(img, _shrink(hull, centre, 0.96), GOLD_DARK)
    _flat(img, _shrink(hull, centre, 0.90), RED_DEEP)
    _flat(img, _shrink(hull, centre, 0.82, LIGHT * unit * 0.04), RED)

    # A gold seam the length of the palm, splitting it into the two panels the
    # glove is built from.
    _fill(img, [centre + up * unit * 0.50 + n * unit * 0.05,
                centre - up * unit * 0.50 + n * unit * 0.05,
                centre - up * unit * 0.50 - n * unit * 0.05,
                centre + up * unit * 0.50 - n * unit * 0.05], GOLD_DARK)

    # Grip pads: the heel of the hand, and one under each knuckle.
    heel = centre - up * unit * 0.32
    pad = _ngon(heel, unit * 0.30, 6, n)
    _fill(img, pad, STEEL_DARK)
    _flat(img, _shrink(pad, heel, 0.72, LIGHT * unit * 0.04), STEEL)
    for key in (lm.INDEX_MCP, lm.MIDDLE_MCP, lm.RING_MCP, lm.PINKY_MCP):
        p = pts[key] + _unit(centre - pts[key]) * unit * 0.17
        small = _ngon(p, unit * 0.115, 6, up)
        _fill(img, small, STEEL_DARK)
        _flat(img, _shrink(small, p, 0.68, LIGHT * unit * 0.02), STEEL)


def _cuff(img, pts, unit: float, centre, back: bool) -> None:
    """The forearm collar."""
    wrist = pts[lm.WRIST]
    d = _unit(wrist - centre)
    n = _perp(d)
    near, far = wrist - d * unit * 0.06, wrist + d * unit * 0.66
    mid = wrist + d * unit * 0.30
    shell = [near + n * unit * 0.50, far + n * unit * 0.62,
             far - n * unit * 0.62, near - n * unit * 0.50]
    _fill(img, shell, SEAM)
    _flat(img, [near + n * unit * 0.46, mid + n * unit * 0.53,
                mid - n * unit * 0.53, near - n * unit * 0.46],
          RED_DEEP if back else RED)
    _flat(img, [far + n * unit * 0.58, far - n * unit * 0.58,
                mid - n * unit * 0.53, mid + n * unit * 0.53], GOLD_DARK)
    _flat(img, [far + n * unit * 0.52, far - n * unit * 0.52,
                mid - n * unit * 0.46, mid + n * unit * 0.46], GOLD)
    # A steel band splitting the collar, as on the reference bracer.
    _fill(img, [mid + n * unit * 0.53 + d * unit * 0.04,
                mid - n * unit * 0.53 + d * unit * 0.04,
                mid - n * unit * 0.53 - d * unit * 0.04,
                mid + n * unit * 0.53 - d * unit * 0.04], STEEL_DARK)
    _rim(img, shell, GOLD_LIT, 1)


def _facing_back(pts, handedness: str) -> bool:
    """True when the camera is looking at the back of the hand.

    The preview is mirrored, and the tracker is fed the unmirrored frame, so
    its handedness label is the anatomical hand. In mirrored image coordinates
    a right hand with its palm to the camera turns wrist -> index knuckle ->
    pinky knuckle one way; seeing the back of that same hand flips the sign.
    """
    a = pts[lm.INDEX_MCP] - pts[lm.WRIST]
    b = pts[lm.PINKY_MCP] - pts[lm.WRIST]
    cross = float(a[0] * b[1] - a[1] * b[0])
    side = 1.0 if handedness.lower().startswith("r") else -1.0
    return side * cross < 0.0


# --------------------------------------------------------------------------

def draw(frame: np.ndarray, pts, handedness: str = "Right",
         charge: float = 0.0, now: float | None = None) -> None:
    """Render the gauntlet over `frame`, in place.

    `pts` is the (21, 2) landmark array already in this frame's pixel
    coordinates. `charge` in [0, 1] brightens the stones -- the app feeds it
    the pinch, so they light up as you close your hand.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (21, 2) or not np.isfinite(pts).all():
        return
    unit = float(np.linalg.norm(pts[lm.MIDDLE_MCP] - pts[lm.WRIST]))
    # Below this the plates are a few pixels across and it reads as a smear;
    # the bare landmarks are more use than that.
    if unit < 18.0:
        return
    now = time.monotonic() if now is None else now
    charge = min(max(float(charge), 0.0), 1.0)
    back = _facing_back(pts, handedness)

    hull, centre = _hull(pts, unit)
    up = _unit(pts[lm.MIDDLE_MCP] - pts[lm.WRIST])
    _cuff(frame, pts, unit, centre, back)
    boss = _dorsal(frame, pts, unit, hull, centre) if back else None
    if not back:
        _palmar(frame, pts, unit, hull, centre)

    for joints, widths in _DIGITS:
        w = [unit * f for f in widths]
        for i in range(3):
            _bone(frame, pts[joints[i]], pts[joints[i + 1]], w[i], w[i + 1], back)
        for i in range(1, 3):
            _band(frame, pts[joints[i - 1]], pts[joints[i]], pts[joints[i + 1]],
                  w[i], back)
        _cap(frame, pts[joints[3]], _unit(pts[joints[3]] - pts[joints[2]]),
             w[3], back)

    if not back:
        return

    # The knuckle hardware goes on after the fingers, so a curled hand cannot
    # paint over the thing the whole silhouette is recognised by.
    def pulse_at(i):
        return 0.86 + 0.14 * float(np.sin(now * 2.1 + i * 1.05))

    seats, _ = _knuckle_bar(frame, pts, unit, centre)
    glow = []
    r = unit * 0.088
    for i, (p, colour) in enumerate(zip(seats, STONES[:4])):
        _stone(frame, p, r, colour, charge, pulse_at(i), 6, up)
        glow.append((p, _mix(colour, (255, 255, 255), 0.30 * charge),
                     r * (1.0 + 0.35 * charge) * pulse_at(i)))

    thumb = pts[lm.THUMB_MCP] + _unit(centre - pts[lm.THUMB_MCP]) * unit * 0.10
    _stone(frame, thumb, r, STONES[4], charge, pulse_at(4), 6, up)
    glow.append((thumb, _mix(STONES[4], (255, 255, 255), 0.30 * charge), r))

    _boss(frame, boss, unit, up, charge, pulse_at(5))
    glow.append((boss, _mix(STONES[5], (255, 255, 255), 0.30 * charge),
                 unit * 0.092))

    _bloom(frame, unit, glow)


def _bloom(frame, unit: float, stones) -> None:
    """Soft light thrown off the stones.

    Three things keep this cheap, and it needs all three -- done naively it
    cost longer than the camera takes to produce the next frame.

    The region is the stones' own bounding box, not the hand's: they sit in one
    corner of a splayed hand, and every pixel outside that box is resized and
    blended for nothing. The blur runs on a downscaled copy. And the composite
    is confined to the same box.
    """
    h, w = frame.shape[:2]
    seats = np.array([p for p, _, _ in stones], dtype=float)
    # Far enough out to hold the widest stone plus the blur's own reach.
    pad = max(r * 2.1 for _, _, r in stones) + unit * 0.70
    x0 = max(int(seats[:, 0].min() - pad), 0)
    y0 = max(int(seats[:, 1].min() - pad), 0)
    x1 = min(int(seats[:, 0].max() + pad), w)
    y1 = min(int(seats[:, 1].max() + pad), h)
    bw, bh = x1 - x0, y1 - y0
    if bw < 8 or bh < 8:
        return
    # Pick the downscale from the blur radius, not from the box. Sizing it to
    # the box leaves the kernel just as wide in the smaller image -- which is
    # the whole cost -- and a 69-tap Gaussian on a thumbnail is no cheaper than
    # it sounds. Scaling until sigma lands near four pixels is what makes it
    # nearly free, and a bloom has no detail that survives the round trip.
    sigma = unit * 0.22
    step = max(1, min(int(sigma / 4.0), 24))
    sw, sh = max(bw // step, 4), max(bh // step, 4)
    layer = np.zeros((sh, sw, 3), np.uint8)
    for p, colour, r in stones:
        cv2.circle(layer, (int((p[0] - x0) / step), int((p[1] - y0) / step)),
                   max(int(r * 2.1 / step), 1), colour, -1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), max(sigma / step, 1.5))
    layer = cv2.resize(layer, (bw, bh), interpolation=cv2.INTER_LINEAR)
    frame[y0:y1, x0:x1] = cv2.addWeighted(frame[y0:y1, x0:x1], 1.0,
                                          layer, 0.55, 0.0)
