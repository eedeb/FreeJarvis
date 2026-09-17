"""Generate the gauntlet: an armoured glove built around the tracked skeleton.

    python tools/blender_run.py tools/bl_gauntlet.py models/gauntlet.glb

Every version of this before now started from someone else's mesh and tried to
work out where its joints were. That is the wrong way round, and it is where
essentially all of the difficulty came from -- a downloaded model does not know
which of its vertices is a knuckle, so the answer had to be guessed from
principal axes, connectivity and vertex clustering, and each of those guesses
was wrong for at least one digit.

Here the skeleton comes first. The bones are laid out to match MediaPipe's 21
landmarks exactly, one bone per tracked joint pair, and then the armour is
extruded along those bones. So a vertex's weights are not inferred, they are
recorded at the moment the vertex is created; there is no segmentation step, no
end-finding, no mirror search, and no seam to fill, because the pieces are
built overlapping in the first place.

It is also original work, which the thing it replaces was not.

Output is a rigged .glb. Pass --preview to also render turntable PNGs beside it.
"""

import math
import pathlib
import sys

import bmesh
import bpy
from mathutils import Vector

# ---------------------------------------------------------------- the skeleton

# The rest pose, in knuckle spans: x from the thumb side towards the pinky, y
# from the wrist towards the fingertips, z out through the back of the hand.
# A right hand, flat, fingers straight and slightly fanned -- the pose that
# skins most cleanly, since no joint starts out bent.
#
# The palm layout is measured from captures/hand_114838 and hand_202433, which
# agree with each other to about 5%. hand_205047 is left out: MediaPipe reports
# its knuckle span as 23mm against 54 and 64mm for the other two, so its world
# landmarks are not describing the same hand. The finger bone lengths are
# standard anthropometry rather than measured, because all three captures have
# the fingers curled and the app calibrates them per-user at runtime anyway.
REST = {
    "WRIST": (0.00, -1.45, 0.00),
    "THUMB_CMC": (-0.42, -1.05, -0.12),
    "THUMB_MCP": (-0.70, -0.50, -0.26),
    "THUMB_IP": (-0.88, -0.14, -0.36),
    "THUMB_TIP": (-1.00, 0.20, -0.44),
    "INDEX_MCP": (-0.50, 0.06, 0.02),
    "INDEX_PIP": (-0.56, 0.62, 0.00),
    "INDEX_DIP": (-0.59, 0.94, -0.02),
    "INDEX_TIP": (-0.61, 1.18, -0.05),
    "MIDDLE_MCP": (-0.17, 0.10, 0.00),
    "MIDDLE_PIP": (-0.18, 0.71, 0.00),
    "MIDDLE_DIP": (-0.19, 1.09, -0.02),
    "MIDDLE_TIP": (-0.20, 1.34, -0.05),
    "RING_MCP": (0.15, 0.06, -0.01),
    "RING_PIP": (0.19, 0.62, -0.02),
    "RING_DIP": (0.22, 0.98, -0.04),
    "RING_TIP": (0.24, 1.22, -0.07),
    "PINKY_MCP": (0.50, -0.12, -0.02),
    "PINKY_PIP": (0.60, 0.32, -0.04),
    "PINKY_DIP": (0.66, 0.57, -0.06),
    "PINKY_TIP": (0.70, 0.79, -0.09),
}

DIGITS = ("thumb", "index", "middle", "ring", "pinky")
SEGMENTS = ("meta", "prox", "mid", "dist")
# Which landmarks each digit runs through. Both the thumb and the fingers have
# four bones, so the naming is uniform even though the anatomy is not: for a
# finger `meta` is the metacarpal inside the palm, and for the thumb it is the
# short bone from the wrist to the base of the thenar.
JOINTS = {
    "thumb": ("WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP"),
    "index": ("WRIST", "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP"),
    "middle": ("WRIST", "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP"),
    "ring": ("WRIST", "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP"),
    "pinky": ("WRIST", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"),
}
# Metacarpals live under the palm plate, so no tube is built on them -- the
# palm's own outline is drawn through the knuckles and the base of the thumb,
# so it already covers everything those bones run under.
BARE = {(d, "meta") for d in DIGITS}
# The thumb keeps the bone from its base to its knuckle. Dropping it, on the
# grounds that the thenar is part of the palm, left the thumb floating clear of
# the hand with daylight under it -- the palm's underside and the thumb's first
# knuckle are a tenth of a span apart, and something has to span that.

# Life size, so the file means something when opened on its own.
SPAN = 0.085

# ------------------------------------------------------------------- the shape

# Cross-section of a plate, as a superellipse. Two is an ellipse and reads as a
# sausage; higher is a rounded rectangle, which is what a plate looks like.
SQUARENESS = 3.0
RING_POINTS = 18
# Half-width and half-depth of each digit, in knuckle spans, at the knuckle and
# at the tip. A glove is bigger than the hand in it, and these already include
# that: a finger is about 0.11 spans across the bone, and these start at 0.15.
# Grown 12% over the first set, which left a rim of bare finger showing round
# every digit: the landmarks sit on the bone, a finger has flesh round them,
# and a glove has more again.
GIRTH = {
    "thumb": ((0.188, 0.159), (0.132, 0.112)),
    "index": ((0.179, 0.150), (0.123, 0.105)),
    "middle": ((0.185, 0.155), (0.127, 0.109)),
    "ring": ((0.177, 0.147), (0.121, 0.103)),
    "pinky": ((0.159, 0.132), (0.109, 0.094)),
}
# Segments that are not the width their digit's taper would give them. The
# thumb's first bone runs through the thenar -- the mound at the base of the
# thumb -- which is about twice the width of the thumb above it, and modelling
# it as a thumb-width tube left the first web space bare: on a captured hand
# with the thumb spread, a fifth of the palm the tracker reports had no armour
# over it at all. Widening the mound covers the web *and* follows the thumb,
# which growing the palm plate sideways would not have done.
# Slimmer through the hand than it was. At 1.10 the thenar tube reached 33.8mm
# out from the palm's mid-plane against the plate's own 25.4mm, so it crossed
# the palm-side surface and the two solids met in a hard ellipse -- the thumb
# looked like it was phasing through the hand. It is still wide, because the
# mound is wide; it is just no longer deeper than the plate it sits in.
BULGE = {("thumb", "prox"): (1.55, 0.72)}
# Segments painted the palm's gold instead of the digits' red.
#
# The thenar is part of the palm on a real gauntlet, and here it also has to
# be. Plates are built overlapping and interpenetrating on purpose -- that is
# what makes the glove one object with no seams -- but an interpenetration is
# only invisible while both solids are the same colour. Red thenar into gold
# palm drew the intersection curve in full: a red wedge lying across the palm,
# which is what "the thumb is phasing through the hand" looks like.
GOLD_SEGMENTS = {("thumb", "prox")}

# How far each segment runs past its own bone at each end. The overlap is the
# whole reason there is no seam to close later: neighbouring plates interpene-
# trate rather than meeting, which is also what articulated armour does.
LAP_BACK, LAP_FWD = 0.10, 0.10
# Where the raised knuckle band sits along a segment, how wide it is, and how
# far it stands proud.
# Narrower than it was. At 0.16 and 0.10 the gold collar took a fifth of every
# segment, and with a gold tip and a gold guard on top of that the fingers came
# out more gold than red; this glove is red with gold at the joints.
BAND_AT, BAND_WIDE, BAND_RAISE = 0.13, 0.055, 1.14

# The palm plate, drawn as an outline in the plane of the hand: round the
# knuckles from the pinky to the index, down past the thumb, across the wrist
# and back up the far side. In knuckle spans, on the same axes as REST.
#
# Written out rather than derived. The first version took the convex hull of
# the seven palm landmarks, grew it about its centre and smoothed it, which is
# a tidy three lines and produces a blob: seven points cannot describe a hand,
# growing about the centre lengthens the palm as much as it widens it, and the
# wrist -- one landmark -- comes to a point. A hand outline is a shape, so it
# is worth ten lines of shape.
PALM_OUTLINE = (
    (0.74, -0.05), (0.25, 0.21), (-0.17, 0.25), (-0.69, 0.16),   # knuckles
    (-0.86, -0.22), (-0.94, -0.64), (-0.76, -1.06),              # thumb side
    (-0.39, -1.42), (0.26, -1.42),                               # wrist
    (0.59, -1.06), (0.76, -0.51),                                # pinky side
)
# Half the thickness through the hand, and how the edge rolls over: each level
# is (height as a fraction of that, how far the outline is inset there). Inset
# rather than scaled -- scaling towards the centre pinches the wrist end into a
# cone, which is exactly what the first version did.
# Flatter and thinner than it started. A hand is a slab with a rolled edge and
# the first version rolled it over most of its own thickness, which read as a
# cushion rather than as a plate.
PALM_HALF = 0.140
PALM_ROLL = ((-1.00, 0.105), (-0.88, 0.030), (0.0, 0.0),
             (0.88, 0.030), (1.00, 0.105))
# How far the back of the hand domes up over the knuckles.
PALM_DOME = 0.03
# Where the wrist cuff starts, and how wide the groove above it is -- both
# measured along the hand in knuckle spans, like the outline.
CUFF_AT, GROOVE_WIDE = -1.10, 0.10
# The flat of the hand, as rings working inwards: how far each is inset from
# the outline, and how far it stands proud of the one before.
#
# The rise is the point. Panel lines drawn as a change of material were tried
# first and came out as a zigzag: a groove is a line in x and y, the rings step
# inwards in whole increments, and the faces that fall within the line's width
# are a staircase rather than a line. A terrace is made of the same rings, has
# an edge that catches the light on its own, and cannot be jagged because
# nothing is being approximated -- the step is where the geometry says it is.
# The flat of the hand, as (how far in towards the middle, how far proud of
# the ring before it).
#
# Scaled towards the centre rather than inset by a distance. Insetting is the
# right way to roll the rim -- it keeps the shape, where scaling would pinch
# the narrow wrist end into a cone -- but it folds a closed outline through
# itself wherever the outline is concave, and the palm's is, slightly, after
# smoothing. Even at a tenth of a span that left 82 needle-thin faces and a
# dark gash across the back of the hand. Over the flat of the plate a scale is
# indistinguishable from an inset and cannot fold.
DECK = ((0.93, 0.000), (0.83, 0.028), (0.71, 0.040))
# The knuckle guards: how much of each follows its finger rather than the palm,
# the rings it is built from as (radius, height above the knuckle), and how
# wide and long each one is in knuckle spans.
GUARD_SHARE = 0.55
# Raised when the palm plate was thickened: at the old heights the guards were
# 0.045 spans proud of the plate, and growing the plate to cover more finger
# swallowed them -- the index and pinky guards vanished entirely and the other
# three became gold blobs half-buried at the finger roots.
GUARD_SHELL = ((1.00, 0.04), (0.92, 0.16), (0.68, 0.23))
GUARD_SIZE = {"thumb": (0.17, 0.15), "index": (0.21, 0.19),
              "middle": (0.22, 0.20), "ring": (0.21, 0.19),
              "pinky": (0.19, 0.17)}
# The thumb's lower guard covers the base of the thenar, which is broader than
# its knuckle, so it is not the same size as the one above it.
GUARD_BASE = (0.26, 0.24)
# Which joints of each digit get a guard: the knuckle, where the digit's plate
# meets the palm's.
#
# The thumb was given a second one at the base of the thenar, on the reasoning
# that it has two such joins. It is invisible -- that joint sits inside the
# palm plate, and a boss raised towards the back of the hand from a point below
# the plate's own surface never comes out of it. Fifty-six vertices for a
# pixel-identical render, so it is not there.
GUARD_JOINTS = {"thumb": (2,), "index": (1,), "middle": (1,),
                "ring": (1,), "pinky": (1,)}
PALM_POINTS = 64
# How far from the thumb's knuckle the palm plate starts following the thumb,
# where it follows it fully, and how much of it ever does. Kept under a half so
# the plate stays a palm plate: past that the whole corner swings away with the
# thumb and opens a gap where it used to be.
WEB_REACH, WEB_CORE, WEB_SHARE = 0.95, 0.30, 0.45
# Steps the skin weights are quantised into before they are written.
WEIGHT_STEPS = 32

RED, GOLD, STEEL, DARK, GLOW = range(5)
MATERIALS = (
    ("plate_red", (0.62, 0.055, 0.045, 1.0), 0.85, 0.30),
    ("plate_gold", (0.83, 0.55, 0.13, 1.0), 0.95, 0.22),
    ("plate_steel", (0.115, 0.120, 0.135, 1.0), 1.00, 0.34),
    ("joint_dark", (0.045, 0.045, 0.05, 1.0), 0.55, 0.52),
    ("repulsor", (0.72, 0.86, 1.00, 1.0), 0.10, 0.10),
)


# The pads on the palm side, as (where, how wide and long, how proud, colour):
# the heel of the hand, the mound under the thumb, and the ridge the fingers
# fold onto. Positions in knuckle spans, on the same axes as the outline.
PAD_SHELL = ((1.00, 0.05), (0.90, 0.70), (0.62, 1.00))
# The repulsor's disc, and how much bare plate is kept around it.
REPULSOR_R, PAD_CLEAR = 0.30, 0.07
# The mound at the base of the thumb is now the palm's, not the thumb's. It
# has to be, to be one surface: a mound carried by the thumb's own plate is a
# separate solid crossing the palm, and a depth buffer draws that honestly.
PADS = ((( -0.05, -1.06), (0.44, 0.20), 0.055, GOLD),
        (( -0.52, -0.62), (0.34, 0.48), 0.090, GOLD),
        ((  0.04,  0.02), (0.62, 0.16), 0.045, GOLD))


def rest(name) -> Vector:
    return Vector(REST[name]) * SPAN


def frame_at(direction: Vector) -> tuple[Vector, Vector]:
    """Two axes across a bone: one out through the back, one across the hand.

    Rolling a plate consistently needs a reference that is not the bone itself.
    The back of the hand is that reference, and it is well conditioned for
    every digit here -- including the thumb, whose bones run diagonally but
    never straight up out of the palm.
    """
    out = Vector((0.0, 0.0, 1.0))
    up = out - direction * direction.dot(out)
    if up.length < 1e-5:
        up = Vector((0.0, 1.0, 0.0)) - direction * direction.dot(
            Vector((0.0, 1.0, 0.0)))
    up.normalize()
    return up, direction.cross(up).normalized()


def superellipse(n=RING_POINTS, power=SQUARENESS):
    """Unit cross-section, once, since every ring on the model shares it."""
    out = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        c, s = math.cos(a), math.sin(a)
        out.append((math.copysign(abs(c) ** (2.0 / power), c),
                    math.copysign(abs(s) ** (2.0 / power), s)))
    return out


SECTION = superellipse()
# Which way each face of a tube points, taken at the middle of the quad: above
# zero is the back of the hand, below is the palm. A glove is not one colour
# all the way round -- red over the back and gold underneath is most of what
# makes the shape read as this glove rather than as a red tube -- and the only
# thing needed to say which is which is the sign of this.
_SIDE = [0.5 * (SECTION[i][1] + SECTION[(i + 1) % RING_POINTS][1])
         for i in range(RING_POINTS)]
BACK_FACE = [y > 0.0 for y in _SIDE]
# ...and the faces on each flank, where the back meets the palm, are the seam
# between the two plates.
#
# Narrow. At 0.30 the seam took four of the eighteen faces round a digit, which
# is a fifth of it, and a digit seen exactly side-on -- the thumb usually, and
# the outer fingers often -- turns its whole flank to the camera. The seam then
# reads as a black stripe down the finger rather than as a line between plates.
EDGE_FACE = [abs(y) < 0.16 for y in _SIDE]


def shell_colours(back, front, seam=None):
    """One material per face around a tube: back, palm, and the seam between."""
    return [seam if seam is not None and EDGE_FACE[i]
            else (back if BACK_FACE[i] else front)
            for i in range(RING_POINTS)]


def bridge(bm, lower, upper, material):
    """Join two equal loops of vertices with a band of quads.

    `material` is either one index for the whole band or one per face.
    """
    n = len(lower)
    per = material if isinstance(material, (list, tuple)) else None
    for i in range(n):
        face = bm.faces.new((lower[i], lower[(i + 1) % n],
                             upper[(i + 1) % n], upper[i]))
        face.material_index = per[i % len(per)] if per is not None else material


def cap(bm, loop, apex, material):
    """Close a loop with a fan to a single point."""
    n = len(loop)
    for i in range(n):
        face = bm.faces.new((loop[i], loop[(i + 1) % n], apex))
        face.material_index = material


# Along a plate: where each ring sits on its bone, and how much the section is
# scaled there. The first and last run past the bone's own ends -- that is the
# overlap -- and the three in the middle raise the knuckle band.
PROFILE = ((-LAP_BACK, 0.86), (0.015, 0.99), (0.045, 1.00),
           (BAND_AT - BAND_WIDE, 1.00), (BAND_AT, BAND_RAISE),
           (BAND_AT + BAND_WIDE, 1.00),
           (0.62, 0.99), (1.0 + LAP_FWD, 0.90))
# One entry per band, and each is either a single material or a colour per face
# round the tube: a dark flexing joint at the knuckle, a gold collar over it,
# then red plate with a dark seam down each flank.
#
# The digits are red the whole way round rather than red over the back and gold
# underneath. Gold undersides are closer to the reference and worse on a
# webcam: half the time a hand is held palm towards the lens, and the glove
# then reads as a gold glove. Red from every angle with gold only at the joints
# is what makes it recognisable in a preview window.
BANDS = (DARK, DARK,
         shell_colours(GOLD, GOLD, STEEL),
         shell_colours(GOLD, GOLD, STEEL),
         shell_colours(GOLD, GOLD),
         shell_colours(RED, RED, DARK),
         shell_colours(RED, RED, DARK))
# Rings added past the end of the last plate on a digit, to round the fingertip
# off rather than leaving it a cut tube. Gold, like the real glove's tips --
# they were steel-grey, which read as fingernails poking out of the armour.
DOME = ((0.05, 0.82), (0.09, 0.58))


def plate(bm, weights, digit, index, bone):
    """One armour plate, extruded along one bone."""
    names = JOINTS[digit]
    head, tail = rest(names[index]), rest(names[index + 1])
    span = tail - head
    length = span.length
    direction = span / length
    up, side = frame_at(direction)
    (knuckle_w, knuckle_d), (tip_w, tip_d) = GIRTH[digit]

    wider, deeper = BULGE.get((digit, SEGMENTS[index]), (1.0, 1.0))

    def loop(t, scale):
        # Girth tapers down the whole digit, not just down this bone, so a
        # plate meets its neighbour at the same width it left off at.
        u = min(max((index - 1 + t) / 3.0, 0.0), 1.0)
        # ...and a bulge fades out towards the far end, so the mound at the
        # base of the thumb meets the thumb at the thumb's own width.
        swell = 1.0 + (wider - 1.0) * max(0.0, 1.0 - t)
        sink = 1.0 + (deeper - 1.0) * max(0.0, 1.0 - t)
        half_w = (knuckle_w + (tip_w - knuckle_w) * u) * SPAN * scale * swell
        half_d = (knuckle_d + (tip_d - knuckle_d) * u) * SPAN * scale * sink
        centre = head + direction * (length * t)
        made = [bm.verts.new(centre + side * (x * half_w) + up * (y * half_d))
                for x, y in SECTION]
        for vert in made:
            weights[vert] = {bone: 1.0}
        return made

    def paint(material):
        """This segment's colours, with red swapped for gold where it belongs."""
        if (digit, SEGMENTS[index]) not in GOLD_SEGMENTS:
            return material
        if isinstance(material, list):
            return [GOLD if m == RED else m for m in material]
        return GOLD if material == RED else material

    rings = [loop(t, scale) for t, scale in PROFILE]
    for i, material in enumerate(BANDS):
        bridge(bm, rings[i], rings[i + 1], paint(material))

    # The open end at the knuckle is capped flat and dark, so that where two
    # plates overlap you see a shadowed gap rather than straight into the tube.
    back = bm.verts.new(head - direction * (length * LAP_BACK * 1.25))
    weights[back] = {bone: 1.0}
    cap(bm, rings[0], back, DARK)

    if index < 3:
        front = bm.verts.new(tail + direction * (length * LAP_FWD * 1.25))
        weights[front] = {bone: 1.0}
        cap(bm, list(reversed(rings[-1])), front, paint(DARK))
        return

    # The last plate on a digit is a fingertip, so it gets domed instead.
    previous = rings[-1]
    for extra, scale in DOME:
        nxt = loop(1.0 + LAP_FWD + extra, scale)
        bridge(bm, previous, nxt, shell_colours(RED, RED))
        previous = nxt
    apex = bm.verts.new(tail + direction * (length * (LAP_FWD + 0.10)))
    weights[apex] = {bone: 1.0}
    cap(bm, list(reversed(previous)), apex, GOLD)


def guard(bm, weights, digit, at):
    """A raised plate over one knuckle, bridging the palm and the finger.

    The thing that makes a glove look like one object rather than a palm with
    tubes lying next to it. Each finger's own plate stops at its knuckle and
    the palm plate stops before it, so however well they overlap there is a
    line across the render where one ends and the other begins; a guard sits
    across that line, proud of both, and the eye reads the join as a knuckle
    instead of as a seam.

    Weighted mostly to the finger, so it rides the knuckle as the finger bends
    -- which is what a real one does -- but not entirely, or it would swing
    clear of the palm and open the very gap it is covering.
    """
    knuckle = rest(JOINTS[digit][at])
    forward = (rest(JOINTS[digit][at + 1]) - knuckle).normalized()
    up, side = frame_at(forward)
    share = {"root": 1.0 - GUARD_SHARE,
             f"{digit}_{SEGMENTS[at]}": GUARD_SHARE}
    wide, long_ = GUARD_BASE if (digit == "thumb" and at == 1)         else GUARD_SIZE[digit]

    rings = []
    for radius, lift in GUARD_SHELL:
        made = []
        for x, y in SECTION:
            vert = bm.verts.new(knuckle
                                + side * (x * radius * wide * SPAN)
                                + forward * (y * radius * long_ * SPAN)
                                + up * (lift * SPAN))
            weights[vert] = dict(share)
            made.append(vert)
        rings.append(made)
    for i in range(len(rings) - 1):
        bridge(bm, rings[i], rings[i + 1], GOLD)
    crown = bm.verts.new(knuckle + up * ((GUARD_SHELL[-1][1] + 0.02) * SPAN))
    root = bm.verts.new(knuckle + up * ((GUARD_SHELL[0][1] - 0.02) * SPAN))
    weights[crown] = weights[root] = dict(share)
    cap(bm, rings[-1], crown, GOLD)
    # The underside is inside the palm and never seen, but it closes the solid.
    cap(bm, list(reversed(rings[0])), root, DARK)


def inset(loop, distance):
    """Pull a closed outline inwards by a fixed distance, everywhere.

    The outline is wound clockwise, so the outward normal of an edge running
    (dx, dy) is (dy, -dx); a corner uses the mean of the two edges meeting
    there. Insetting by a distance keeps the shape and rounds the rim, where
    scaling towards a centre would have made the narrow end vanish first.
    """
    n = len(loop)
    out = []
    for i in range(n):
        (px, py), (cx, cy), (nx, ny) = loop[i - 1], loop[i], loop[(i + 1) % n]
        away = Vector((0.0, 0.0))
        for ax, ay, bx, by in ((px, py, cx, cy), (cx, cy, nx, ny)):
            edge = Vector((by - ay, -(bx - ax)))
            if edge.length > 1e-9:
                away += edge.normalized()
        if away.length < 1e-9:
            out.append((cx, cy))
            continue
        away.normalize()
        out.append((cx - away.x * distance, cy - away.y * distance))
    return out


def smooth_loop(points, count):
    """Resample a closed polygon into a smooth loop of `count` points.

    Catmull-Rom, so the palm's outline is a curve through the landmarks rather
    than the seven-sided plate the hull on its own would give.
    """
    n = len(points)
    out = []
    for i in range(count):
        u = i * n / count
        k = int(u)
        f = u - k
        p0, p1 = points[(k - 1) % n], points[k % n]
        p2, p3 = points[(k + 1) % n], points[(k + 2) % n]
        out.append(tuple(
            0.5 * ((2 * p1[a]) + (-p0[a] + p2[a]) * f
                   + (2 * p0[a] - 5 * p1[a] + 4 * p2[a] - p3[a]) * f * f
                   + (-p0[a] + 3 * p1[a] - 3 * p2[a] + p3[a]) * f * f * f)
            for a in (0, 1)))
    return out


PALM_MARKS = ("WRIST", "THUMB_CMC", "THUMB_MCP", "INDEX_MCP",
              "MIDDLE_MCP", "RING_MCP", "PINKY_MCP")


def palm(bm, weights, bone):
    """The plate over the back and palm of the hand.

    Its outline is the hull of the landmarks that bound the palm, pushed out to
    allow for flesh and for the glove being bigger than the hand, and then
    smoothed. Deriving it from the same landmarks the tracker reports is what
    makes it land on the hand instead of near it -- the shape and the fit come
    from one source rather than two that have to be reconciled.
    """
    knuckle = REST["THUMB_MCP"][:2]

    def web(x, y, bone):
        """How much of this palm vertex follows the thumb rather than the palm.

        The one place on the glove where a vertex is not rigid to a single
        bone, and it earns it. The web between the thumb and the index finger
        opens and closes as the thumb spreads; a palm plate rigid to the wrist
        cannot follow that, so on a captured hand with the thumb abducted a
        fifth of the palm the tracker reports had no armour over it. Blending
        the corner of the plate onto the thumb's own bone closes the web by
        moving with it, which is what skinning is for.
        """
        far = math.hypot(x - knuckle[0], y - knuckle[1])
        near = (WEB_REACH - far) / (WEB_REACH - WEB_CORE)
        share = WEB_SHARE * min(max(near, 0.0), 1.0)
        if share <= 0.0:
            return {bone: 1.0}
        return {bone: 1.0 - share, "thumb_prox": share}

    outline = smooth_loop(PALM_OUTLINE, PALM_POINTS)
    cx = sum(p[0] for p in outline) / len(outline)
    cy = sum(p[1] for p in outline) / len(outline)
    mid_z = sum(rest(n).z for n in PALM_MARKS) / len(PALM_MARKS)
    half = PALM_HALF * SPAN

    def dome(x, y, height):
        """The back of a hand is not flat -- it rises over the knuckles."""
        if height <= 0.0:
            return 0.0
        near = min(max((y + 1.5) / 1.7, 0.0), 1.0)
        return PALM_DOME * SPAN * height * near * near

    rings = []
    for height, pull in PALM_ROLL:
        made = []
        for x, y in inset(outline, pull):
            vert = bm.verts.new((x * SPAN, y * SPAN,
                                 mid_z + height * half + dome(x, y, height)))
            weights[vert] = web(x, y, bone)
            made.append(vert)
        rings.append(made)

    # Gold on the palm, red across the back -- and on the back, a gold cuff at
    # the wrist with a dark groove above it. The groove is what stops the back
    # of the hand reading as one smooth pillow: real armour is panels, and a
    # panel edge is the cheapest detail there is, being nothing but a change of
    # material along a line the geometry already has.
    # The wrist end of the outline, as a contiguous run of its own samples.
    # The cuff is drawn on the rolled rim only, and by index rather than by
    # position: on the rim an index range is an arc, so its two ends are the
    # only places the colour changes and the edge is clean. Asking instead
    # whether each face's centre lay past the wrist gave a torn edge, because
    # the flat of the hand is built from three concentric rings and a band
    # across it can only ever follow a ring boundary -- it cannot cut across
    # one, so it came out as a staircase two steps deep.
    hem = {i for i, (_, y) in enumerate(
        [(0.5 * (outline[i][0] + outline[(i + 1) % len(outline)][0]),
          0.5 * (outline[i][1] + outline[(i + 1) % len(outline)][1]))
         for i in range(len(outline))]) if y < CUFF_AT}

    def face_of(level, i, deck=False):
        """Which material a face carries: gold palm, red back, gold cuff."""
        if not deck and i in hem:
            return GOLD
        return GOLD if level < 2 else RED

    for i in range(len(rings) - 1):
        bridge(bm, rings[i], rings[i + 1],
               [face_of(i, k) for k in range(len(outline))])

    # The two flat faces, built as concentric decks rather than as one fan to a
    # single point. A fan is three lines of code and gives a surface with no
    # interior edges, so there is nowhere for a panel line to be: the grooves
    # came out on the rim of the palm and nowhere across it. Insetting the
    # outline a few times gives faces at known positions, and the same rule
    # that colours the rim then draws straight lines over the whole plate.
    def deck(edge, height, sign):
        loop, flat = rings[edge], outline
        base = inset(outline, PALM_ROLL[edge][1])
        for shrink, rise in DECK:
            pulled = [(cx + (x - cx) * shrink, cy + (y - cy) * shrink)
                      for x, y in base]
            made = []
            for x, y in pulled:
                vert = bm.verts.new((x * SPAN, y * SPAN,
                                     mid_z + height * (half + rise * SPAN)
                                     + dome(x, y, height)))
                weights[vert] = web(x, y, bone)
                made.append(vert)
            band = [face_of(edge, k, deck=True) for k in range(len(outline))]
            bridge(bm, loop if sign > 0 else list(reversed(loop)),
                   made if sign > 0 else list(reversed(made)), band)
            loop, flat = made, pulled
        peak = bm.verts.new((cx * SPAN, cy * SPAN,
                             mid_z + height * (half + DECK[-1][1] * SPAN)
                             + dome(cx, cy, height)))
        weights[peak] = {bone: 1.0}
        cap(bm, loop if sign > 0 else list(reversed(loop)), peak,
            face_of(edge, 0, deck=True))

    def pad(at, size, tall, material):
        """A raised boss on the palm side of the hand.

        The palm plate is a big smooth face with one repulsor on it, and seen
        square to the camera -- which is half the time, since that is how a
        hand is held -- it reads as a paddle. These are the pads a hand
        actually has: the heel, the mound under the thumb, and the ridge under
        the fingers. Geometry rather than a change of material, for the same
        reason the terrace is: a raised edge shades itself and cannot come out
        as a staircase.
        """
        x0, y0 = at
        wide, long_ = size
        below = mid_z - (half + DECK[-1][1] * SPAN)

        def clear(x, y):
            """Push a point out of the repulsor's seat if it landed inside.

            The pads are placed by hand and the repulsor sits wherever the
            palm's outline puts its centre, so the two are one edited constant
            away from colliding -- and did: the mound at the base of the thumb
            grew over the repulsor by 0.23 spans, and the other two cleared it
            by 0.04. Enforcing the gap here rather than trusting the numbers
            means moving a pad, resizing one, or reshaping the palm cannot put
            it back. The pads end up hugging the disc, which is what a plate
            around a fitting looks like anyway.
            """
            dx, dy = x - cx, y - cy
            far = math.hypot(dx, dy)
            keep = REPULSOR_R + PAD_CLEAR
            if far >= keep:
                return x, y
            if far < 1e-9:
                return cx + keep, cy
            return cx + dx / far * keep, cy + dy / far * keep

        rings_ = []
        for radius, rise in PAD_SHELL:
            made = []
            for x, y in SECTION:
                px, py = clear(x0 + x * radius * wide, y0 + y * radius * long_)
                vert = bm.verts.new((px * SPAN, py * SPAN,
                                     below - rise * tall * SPAN))
                weights[vert] = web(px, py, bone)
                made.append(vert)
            rings_.append(made)
        for k in range(len(rings_) - 1):
            bridge(bm, list(reversed(rings_[k])), list(reversed(rings_[k + 1])),
                   material)
        px, py = clear(x0, y0)
        crown = bm.verts.new((px * SPAN, py * SPAN, below - tall * SPAN * 1.15))
        seat = bm.verts.new((px * SPAN, py * SPAN, below + 0.02 * SPAN))
        weights[crown] = weights[seat] = web(px, py, bone)
        cap(bm, list(reversed(rings_[-1])), crown, material)
        cap(bm, rings_[0], seat, DARK)

    deck(len(rings) - 1, PALM_ROLL[-1][0], 1)
    deck(0, PALM_ROLL[0][0], -1)
    for at, size, tall, material in PADS:
        pad(at, size, tall, material)
    # The outermost surface, not the mid-plane's half-thickness: the terrace
    # stands proud of it, and the repulsor sank inside the palm when it was
    # seated against the old number.
    return Vector((cx * SPAN, cy * SPAN, mid_z)), half + DECK[-1][1] * SPAN


def repulsor(bm, weights, bone, centre, thick, radius=REPULSOR_R, rings=18):
    """The disc in the middle of the palm."""
    base, lip = [], []
    for i in range(rings):
        a = 2.0 * math.pi * i / rings
        x = centre.x + math.cos(a) * radius * SPAN
        y = centre.y + math.sin(a) * radius * SPAN
        base.append(bm.verts.new((x, y, centre.z - thick * 1.02)))
        lip.append(bm.verts.new((centre.x + (x - centre.x) * 0.74,
                                 centre.y + (y - centre.y) * 0.74,
                                 centre.z - thick * 1.14)))
    eye = bm.verts.new((centre.x, centre.y, centre.z - thick * 1.16))
    # The back of the disc, so it is a solid rather than an open bowl. Without
    # it the outer ring is eighteen edges with one face each, which is every
    # non-manifold edge in the model.
    seat = bm.verts.new((centre.x, centre.y, centre.z - thick * 1.02))
    for vert in base + lip + [eye, seat]:
        weights[vert] = {bone: 1.0}
    bridge(bm, list(reversed(base)), list(reversed(lip)), STEEL)
    cap(bm, list(reversed(lip)), eye, GLOW)
    cap(bm, base, seat, STEEL)


# ------------------------------------------------------------------ assembly

def build_armature():
    """The skeleton: a root at the wrist and four bones down every digit."""
    data = bpy.data.armatures.new("hand")
    rig = bpy.data.objects.new("gauntlet_rig", data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")

    knuckles = [rest(JOINTS[d][1]) for d in DIGITS[1:]]
    root = data.edit_bones.new("root")
    root.head = rest("WRIST")
    root.tail = sum(knuckles, Vector()) / len(knuckles)
    root.align_roll(Vector((0.0, 0.0, 1.0)))

    for digit in DIGITS:
        names = JOINTS[digit]
        parent = root
        for index, segment in enumerate(SEGMENTS):
            bone = data.edit_bones.new(f"{digit}_{segment}")
            bone.head = rest(names[index])
            bone.tail = rest(names[index + 1])
            bone.parent = parent
            # Connected everywhere but at the metacarpal, whose head is the
            # wrist rather than the root's tail.
            bone.use_connect = index > 0
            bone.align_roll(Vector((0.0, 0.0, 1.0)))
            parent = bone
        # A marker at the fingertip, carrying no weight.
        #
        # glTF stores a joint as a point, not as a bone with two ends, so the
        # last bone of a chain exports its knuckle and loses its tip. Without
        # this the loader can recover the rest direction of every bone except
        # the one at the end of each digit, and would have to get those five
        # from somewhere else. One empty bone per digit is cheaper than a
        # special case, and it means the file describes all 21 landmarks.
        tip = data.edit_bones.new(f"{digit}_tip")
        tip.head = rest(names[4])
        tip.tail = rest(names[4]) + (rest(names[4]) - rest(names[3])) * 0.25
        tip.parent = parent
        tip.use_connect = True
        tip.align_roll(Vector((0.0, 0.0, 1.0)))
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def build_mesh(rig):
    """The armour, and the weights recorded as it is made."""
    bm = bmesh.new()
    weights = {}
    for digit in DIGITS:
        for index, segment in enumerate(SEGMENTS):
            if (digit, segment) in BARE:
                continue
            plate(bm, weights, digit, index, f"{digit}_{segment}")
    centre, thick = palm(bm, weights, "root")
    repulsor(bm, weights, "root", centre, thick)
    for digit in DIGITS:
        for at in GUARD_JOINTS[digit]:
            guard(bm, weights, digit, at)

    # Every plate is built ring by ring and capped at both ends, and the two
    # caps have to wind opposite ways for the solid to be closed. Reasoning
    # that out per cap is the kind of thing that is right until it is not --
    # the first version left 1,426 edges whose two faces disagreed, which is
    # invisible in a preview render and turns into holes the moment anything
    # culls back faces. Let the geometry decide instead.
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.verts.index_update()
    # Read the weights off before to_mesh, which invalidates the bmesh verts.
    # Grouped by bone *and* by weight, because Blender's vertex_groups.add
    # takes one weight for a whole list of vertices. Weights are quantised into
    # a few dozen steps first, which turns one call per vertex into one call
    # per step and is finer than a skin needs.
    by_bone = {}
    for vert, share in weights.items():
        for bone, value in share.items():
            if value <= 1e-4:
                continue
            step = round(value * WEIGHT_STEPS) / WEIGHT_STEPS
            by_bone.setdefault((bone, step), []).append(vert.index)

    me = bpy.data.meshes.new("gauntlet")
    bm.to_mesh(me)
    bm.free()
    for polygon in me.polygons:
        polygon.use_smooth = True

    obj = bpy.data.objects.new("gauntlet", me)
    bpy.context.collection.objects.link(obj)
    for name, base, metallic, roughness in MATERIALS:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        shader = mat.node_tree.nodes["Principled BSDF"]
        shader.inputs["Base Color"].default_value = base
        shader.inputs["Metallic"].default_value = metallic
        shader.inputs["Roughness"].default_value = roughness
        if name == "repulsor":
            for slot in ("Emission Color", "Emission"):
                if slot in shader.inputs:
                    shader.inputs[slot].default_value = base
                    break
            if "Emission Strength" in shader.inputs:
                shader.inputs["Emission Strength"].default_value = 6.0
        me.materials.append(mat)

    # Weights, not heat-map guesses. Each plate is rigid on its own bone --
    # which is what plate armour is -- and the plates overlap, so there is
    # nothing to blend across and nothing to tear open when a finger bends.
    groups = {}
    for (bone, value), indices in by_bone.items():
        if bone not in groups:
            groups[bone] = obj.vertex_groups.new(name=bone)
        groups[bone].add(indices, value, "REPLACE")
    obj.parent = rig
    obj.modifiers.new("Armature", "ARMATURE").object = rig
    return obj


def look_at(camera, target, angle, elevation, distance):
    from math import cos, radians, sin
    a, e = radians(angle), radians(elevation)
    camera.location = target + Vector((cos(a) * cos(e), sin(a) * cos(e),
                                       sin(e))) * distance
    camera.rotation_euler = (target - camera.location).to_track_quat(
        "-Z", "Y").to_euler()


def preview(obj, out: pathlib.Path):
    """Render the model from four sides, so it can be judged without a GUI."""
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.render.resolution_x = scene.render.resolution_y = 560
    scene.render.film_transparent = False
    scene.world = scene.world or bpy.data.worlds.new("world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = \
        (0.05, 0.05, 0.06, 1.0)

    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 5.0
    sun.rotation_euler = (0.9, 0.1, 0.8)
    bpy.context.collection.objects.link(sun)
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    target = sum((Vector(c[:]) for c in obj.bound_box), Vector()) / 8.0
    reach = max(obj.dimensions) * 1.9
    shots = (("back", 90, 62), ("palm", 90, -62),
             ("thumb", 190, 12), ("pinky", 5, 22))
    made = []
    for name, angle, elevation in shots:
        look_at(camera, target, angle, elevation, reach)
        scene.render.filepath = str(out.with_name(f"{out.stem}_{name}.png"))
        bpy.ops.render.render(write_still=True)
        made.append(scene.render.filepath)
    return made


def main() -> int:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    want_preview = "--preview" in args
    args = [a for a in args if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    out = pathlib.Path(args[0]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    # 5.2 leaves an object behind even when asked for an empty scene.
    for stray in list(bpy.data.objects):
        bpy.data.objects.remove(stray, do_unlink=True)
    rig = build_armature()
    obj = build_mesh(rig)
    tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    print(f"  {len(obj.data.vertices):,} verts, {tris:,} tris, "
          f"{len(rig.data.bones)} bones, "
          f"{len(obj.vertex_groups)} vertex groups")

    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB",
                              export_apply=False)
    print(f"  wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    if want_preview:
        for shot in preview(obj, out):
            print(f"  wrote {shot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
