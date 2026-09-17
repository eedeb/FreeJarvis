"""The generated gauntlet, skinned onto the tracked hand.

This replaces an earlier attempt that fitted an unrigged downloaded mesh onto
the hand. The difference is not in the maths, which is ordinary linear blend
skinning, but in what has to be known beforehand: nothing. The model carries a
skeleton whose bones are named for the landmarks they span, so which vertices
belong to a knuckle, which way a digit points and where its ends are all come
out of the file. The old fit had to infer every one of those from the shape of
the mesh -- principal axes, vertex clustering, connectivity -- and each
inference was wrong for at least one digit at some point. The model is now
generated around this skeleton by tools/bl_gauntlet.py, which is why the
weights are a record rather than a guess.

One transform is built per bone:

    T = translate(tracked head) . stretch . scale . rotate . translate(-rest head)

Its head is pinned to the joint the tracker reports, it points along the bone
the tracker reports, and it is stretched to that bone's length. Nothing
accumulates down a chain, so a bad landmark disturbs two plates rather than
every plate beyond it.
"""

from __future__ import annotations

import pathlib

import numpy as np

from . import landmarks as lm
from .gltf import load_skinned

# Landmarks each digit runs through. The same table the model was generated
# from -- it is the contract between the two, and the bone names encode it, so
# a mismatch shows up as a missing bone rather than as a subtly wrong pose.
CHAIN = {
    "thumb": (lm.WRIST, lm.THUMB_CMC, lm.THUMB_MCP, lm.THUMB_IP, lm.THUMB_TIP),
    "index": (lm.WRIST, lm.INDEX_MCP, lm.INDEX_PIP, lm.INDEX_DIP, lm.INDEX_TIP),
    "middle": (lm.WRIST, lm.MIDDLE_MCP, lm.MIDDLE_PIP, 11, lm.MIDDLE_TIP),
    "ring": (lm.WRIST, lm.RING_MCP, lm.RING_PIP, 15, lm.RING_TIP),
    "pinky": (lm.WRIST, lm.PINKY_MCP, lm.PINKY_PIP, 19, lm.PINKY_TIP),
}
SEGMENTS = ("meta", "prox", "mid", "dist")
# Metacarpals, whose length and direction are fixed by the palm rather than by
# a joint the tracker can see bend.
PALM_MARKS = (lm.INDEX_MCP, lm.MIDDLE_MCP, lm.RING_MCP, lm.PINKY_MCP)
# The thumb, which is what tells one hand from the other.
THUMB_MARKS = (lm.THUMB_CMC, lm.THUMB_MCP, lm.THUMB_IP, lm.THUMB_TIP)
# There is deliberately no "the preview is mirrored, so expect the opposite
# sign" constant here, though there was one and it was wrong.
#
# The preview *is* mirrored, and a right hand does therefore arrive measuring
# as a left one. The answer to that is to reflect the model, which is what the
# comparison below already asks for -- not to redefine which signs count as
# agreeing. Redefining it made every hand match without reflecting anything,
# and the two frames were then left disagreeing about which way the palm
# normal points: the model was drawn back to front, so the gold palm and the
# repulsor appeared on the back of the hand.
#
# It survived because the fit was unaffected. A glove drawn inside out lands
# in exactly the same place, covers exactly as much, and passes every check
# that measures where the armour is rather than which face of it you see.

# How far a digit may be stretched or squashed along its own bone to reach the
# joint the tracker reports. A hand whose proportions differ from the model's
# is normal; a landmark that has jumped is not.
STRETCH = (0.55, 1.75)
# How far the hand may reach in depth, as a multiple of its palm length.
DEPTH_LIMIT = 1.6


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def _ortho(across: np.ndarray, along: np.ndarray) -> np.ndarray | None:
    """An orthonormal frame from two axes that are only roughly perpendicular.

    Columns: across the knuckles, along the hand, out through the back. The
    two inputs meet at whatever angle a particular hand's knuckles make -- 74
    degrees on one measured here -- so stacking them raw gives a matrix that is
    not a rotation, and the rig multiplies two of those together.
    """
    e2 = _unit(along)
    e1 = _unit(across - e2 * float(e2 @ across))
    if not e1.any() or not e2.any():
        return None
    return np.stack([e1, e2, np.cross(e2, e1)], axis=1)


def _chirality(points: np.ndarray, frame: np.ndarray) -> float:
    """Which hand this is: +1 or -1, and only the difference matters.

    Measured as how far the thumb sits off the plane of the palm, which is a
    pseudo-scalar -- it changes sign under reflection, and a left hand is a
    reflected right hand.

    The obvious measurement is not. Asking which side of the knuckle line the
    thumb is on -- its component along the across-the-hand axis -- is a dot
    product, and a dot product is preserved by every orthogonal map including a
    reflection. Measured on three captures and their mirror images it gave the
    same number to the last decimal for both, so the check could never fire and
    a left hand would have been given a right glove. That is the same class of
    bug that made the old renderer draw a mangled starfish, and it survived
    here only because every capture to hand is of one chirality.

    The whole thumb is summed rather than one landmark of it. On the model the
    four thumb landmarks stand 0.13 to 0.47 knuckle spans off the palm and on
    the captures -0.14 to -1.04, so any one of them would do; four is free and
    does not care which way the thumb happens to be curled.
    """
    return float(np.sign(((points[THUMB_MARKS, :] - points[lm.WRIST])
                          @ frame[:, 2]).sum()))


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector `a` onto unit vector `b`."""
    v = np.cross(a, b)
    c = float(a @ b)
    sin2 = float(v @ v)
    if sin2 < 1e-18:
        if c > 0.0:
            return np.eye(3)
        u = np.eye(3)[int(np.argmin(np.abs(a)))]
        u = _unit(u - a * float(a @ u))
        return 2.0 * np.outer(u, u) - np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / sin2)


def tracked_pose(hand, pts_px: np.ndarray, heads=None, tails=None,
                 rest_len=None):
    """The hand in a 3D space measured in screen pixels.

    Returns the pose and the scale in pixels per model unit.

    The image landmarks say where the hand is and how big, and they are the
    reliable half. The world landmarks carry the shape. Turning one into the
    other needs a scale, and the obvious way to get it -- match the spread of
    the world x and y, which are camera-aligned, against the spread on screen
    -- is right for x and y and wrong for z.

    It is wrong because MediaPipe's world z is not in the same units as its x
    and y. Measured on three captures of one hand it reports the hand as 111 to
    123mm deep against 95 to 146mm wide, and a hand is not that thick. Scaling
    z by the number that fits x and y therefore made every hand about twice as
    deep as it is, which tilted the palm frame by tens of degrees and, on the
    one capture where the world landmarks are worst, spread the palm plate into
    a disc across the fingers.

    That is real, and it is not yet fixable here. Bone lengths do not change,
    so a gain can in principle be solved for by asking which one makes the 3D
    bone lengths match the model's proportions. Swept over three captures, the
    best gain came out as 0.00, 1.50 and 0.95 times the xy scale, and the
    residual barely moved across the whole range -- 20.0% at the optimum
    against 21.1% at a gain of one. The objective is flat because that 20% is
    not depth at all: it is the model's standard-anthropometry proportions
    disagreeing with a particular hand, and it is an order of magnitude larger
    than the thing being solved for.

    So depth waits for per-user bone-length calibration, which needs many
    frames rather than one, and until then the xy scale stands. Measurement
    says it is within a couple of percent of the best any single frame allows.
    The bone table is still passed in, because calibration is what will use it.
    """
    world = np.asarray(getattr(hand, "world", None), dtype=float)
    if world.shape != (21, 3) or not np.isfinite(world).all():
        return None, None
    centred = world - world.mean(axis=0)
    seen = float(((pts_px - pts_px.mean(axis=0)) ** 2).sum())
    flat = float((centred[:, 0] ** 2 + centred[:, 1] ** 2).sum())
    if flat < 1e-12 or seen <= 0.0:
        return None, None

    gain = float(np.sqrt(seen / flat))
    # How big the hand is on screen, in pixels per model unit. Taken from the
    # bones rather than from the overall spread, and by median rather than by
    # mean, so one landmark that has jumped moves it by nothing.
    scale = None
    if heads is not None and len(heads):
        want = np.asarray(rest_len, dtype=float)
        deep = (centred[tails, 2] - centred[heads, 2]) * gain
        got = np.hypot(np.linalg.norm(pts_px[tails] - pts_px[heads], axis=1),
                       deep)
        usable = want > 1e-9
        if usable.sum() >= 3:
            scale = float(np.median(got[usable] / want[usable]))

    out = np.empty((21, 3))
    out[:, :2] = pts_px
    # A hand held straight at the camera is the one pose this cannot pin down
    # -- every bone is foreshortened at once and the solve has nothing left to
    # separate depth from size -- so the answer is still bounded.
    limit = DEPTH_LIMIT * float(np.linalg.norm(
        pts_px[lm.MIDDLE_MCP] - pts_px[lm.WRIST]))
    out[:, 2] = np.clip(centred[:, 2] * gain, -limit, limit)
    return out, scale


class Glove:
    """The rigged model. Load once, pose per frame."""

    def __init__(self, path: pathlib.Path | str | None = None):
        self.reason: str | None = None
        self.model: dict | None = None
        if path is None:
            path = (pathlib.Path(__file__).resolve().parent.parent
                    / "models" / "gauntlet.glb")
        self.path = pathlib.Path(path)
        try:
            self._load()
        except Exception as exc:            # noqa: BLE001 - any failure disables it
            self.reason = f"{type(exc).__name__}: {exc}"
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        model = load_skinned(self.path)
        slot = {name: i for i, name in enumerate(model["joint_names"])}

        # Every landmark's rest position, read off the skeleton. A bone's name
        # says which landmark its head is, and the inverse bind matrix says
        # where that head sits, so 21 positions fall out of 26 bone names with
        # nothing hard-coded and nothing measured.
        rest_lm = np.full((21, 3), np.nan)
        rest_lm[lm.WRIST] = model["rest"][slot["root"]][:3, 3]
        # The root runs from the wrist to the middle of the knuckles, which is
        # not a landmark pair, so it is written with `None` for its far end and
        # resolved against the palm's centre. Leaving it out of this list is
        # the obvious mistake and not an obvious symptom: every joint still
        # lands correctly, because the digits are pinned to their own
        # landmarks, while the palm -- the one thing the root drives -- stays
        # at its rest position near the origin and drags the glove's bounding
        # box across the whole frame.
        bones = [(slot["root"], lm.WRIST, None)]
        for digit, chain in CHAIN.items():
            for k, segment in enumerate(SEGMENTS + ("tip",)):
                name = f"{digit}_{segment}"
                if name not in slot:
                    raise ValueError(f"{self.path.name} has no bone {name!r}")
                if k:                          # meta's head is the wrist
                    rest_lm[chain[k]] = model["rest"][slot[name]][:3, 3]
                if segment != "tip":
                    bones.append((slot[name], chain[k], chain[k + 1]))
        if not np.isfinite(rest_lm).all():
            raise ValueError("the skeleton does not cover all 21 landmarks")

        frame = _ortho(rest_lm[lm.PINKY_MCP] - rest_lm[lm.INDEX_MCP],
                       rest_lm[PALM_MARKS, :].mean(axis=0) - rest_lm[lm.WRIST])
        if frame is None:
            raise ValueError("the skeleton's palm is degenerate")

        model["rest_lm"] = rest_lm
        model["frame"] = frame
        model["bones"] = bones
        # Every real bone as (head, tail, rest length), for the depth solve.
        # The root is left out: it runs to the middle of the knuckles, which is
        # not a landmark, so it has no 2D length to measure against.
        pairs = [(h, t) for _, h, t in bones if t is not None]
        model["heads"] = np.array([h for h, _ in pairs])
        model["tails"] = np.array([t for _, t in pairs])
        model["bone_len"] = np.linalg.norm(
            rest_lm[model["tails"]] - rest_lm[model["heads"]], axis=1)
        # Which hand the model is, so one of the other kind can be spotted
        # and the model reflected for it.
        model["hand_sign"] = _chirality(rest_lm, frame)
        # Metacarpals do not bend, so their lengths are the steadiest thing on
        # a hand to take an overall scale from.
        model["palm_rest"] = np.linalg.norm(
            rest_lm[PALM_MARKS, :] - rest_lm[lm.WRIST], axis=1)
        self.model = model
        self._flipped: dict | None = None

    def _for(self, mirror: bool) -> dict:
        """The model, reflected across the palm if this is the other hand.

        Reflecting the rest pose is the whole of it: a left glove is a right
        glove seen in a mirror, and every transform below is built from the
        rest pose, so nothing downstream needs to know which hand it is.
        """
        if not mirror:
            return self.model
        if self._flipped is None:
            model = dict(self.model)
            frame = model["frame"]
            # Reflect through the plane of the palm's long axis and normal,
            # which is to say negate the across-the-knuckles component.
            axis = frame[:, 0]
            reflect = np.eye(3) - 2.0 * np.outer(axis, axis)
            model["verts"] = self.model["verts"] @ reflect.T
            model["normals"] = self.model["normals"] @ reflect.T
            model["rest_lm"] = self.model["rest_lm"] @ reflect.T
            model["rest"] = self.model["rest"].copy()
            model["rest"][:, :3, 3] = self.model["rest"][:, :3, 3] @ reflect.T
            model["frame"] = _ortho(
                model["rest_lm"][lm.PINKY_MCP] - model["rest_lm"][lm.INDEX_MCP],
                model["rest_lm"][PALM_MARKS, :].mean(axis=0)
                - model["rest_lm"][lm.WRIST])
            # A reflection reverses winding, so the faces are turned over too
            # rather than leaving the renderer to cull the wrong side.
            model["faces"] = self.model["faces"][:, ::-1].copy()
            self._flipped = model
        return self._flipped

    def pose(self, hand, pts_px) -> dict | None:
        """Skin the model onto one tracked hand.

        Returns the posed vertices and normals in the same pixel space the
        landmarks are in, plus the faces, their materials and which bone drove
        each one.
        """
        if self.model is None:
            return None
        pts_px = np.asarray(pts_px, dtype=float)
        if pts_px.shape != (21, 2) or not np.isfinite(pts_px).all():
            return None
        pose, solved = tracked_pose(hand, pts_px, self.model["heads"],
                                    self.model["tails"],
                                    self.model["bone_len"])
        if pose is None or not np.isfinite(pose).all():
            return None

        live = _ortho(pose[lm.PINKY_MCP] - pose[lm.INDEX_MCP],
                      pose[PALM_MARKS, :].mean(axis=0) - pose[lm.WRIST])
        if live is None:
            return None
        # Reflect the model when the hand in front of the camera is not the
        # one the model is. Mirroring the preview makes every hand read as the
        # other kind, so in practice this fires on every frame -- which is
        # correct, and is what puts the palm on the palm.
        sign = _chirality(pose, live)
        model = self._for(sign != 0.0 and sign != self.model["hand_sign"])

        rest_lm = model["rest_lm"]
        turn = live @ model["frame"].T

        # One scale for the whole glove, from the metacarpals: they cannot
        # bend, so their length changes only with how big the hand is.
        #
        # Measured in the image plane, against where the model's own
        # metacarpals would project. Comparing their lengths in three
        # dimensions instead is the obvious version and it double-counts the
        # depth: a hand turned away has its metacarpals foreshortened on screen
        # already, and `turn` has that rotation in it. On the capture whose
        # world landmarks are wrong -- 23mm of knuckle span where the same hand
        # measures 54 and 64mm elsewhere -- that inflated the palm into a disc
        # covering the fingers. Only x and y are used here, and those come
        # straight from the image landmarks, which are the reliable half.
        # The depth solve already answered how big the hand is on screen, so
        # take its answer. The metacarpals are the fallback: they cannot bend,
        # which makes them the steadiest thing on a hand to measure.
        scale = solved
        if scale is None or not np.isfinite(scale) or scale <= 0.0:
            aimed = (rest_lm[PALM_MARKS, :] - rest_lm[lm.WRIST]) @ turn.T
            flat = np.linalg.norm(aimed[:, :2], axis=1)
            seen = np.linalg.norm(pose[PALM_MARKS, :2] - pose[lm.WRIST, :2],
                                  axis=1)
            usable = flat > 1e-9
            if usable.sum() < 2:
                return None
            scale = float(np.median(seen[usable] / flat[usable]))
        if not np.isfinite(scale) or scale <= 0.0:
            return None

        joints = len(model["joint_names"])
        rot = np.tile(np.eye(3), (joints, 1, 1))
        off = np.zeros((joints, 3))
        knuckles = pose[PALM_MARKS, :].mean(axis=0)
        rest_knuckles = rest_lm[PALM_MARKS, :].mean(axis=0)
        for index, head, tail in model["bones"]:
            far = knuckles if tail is None else pose[tail]
            rest_far = rest_knuckles if tail is None else rest_lm[tail]
            want = _unit(far - pose[head])
            rest_span = rest_far - rest_lm[head]
            length = float(np.linalg.norm(rest_span))
            if not want.any() or length < 1e-9:
                continue
            aimed = turn @ _unit(rest_span)
            spin = _align(aimed, want) @ turn
            # Stretched along its own bone to the length the tracker reports,
            # and scaled evenly across it. Both matter and they are not the
            # same number: driving the girth from the per-bone ratio would give
            # neighbouring plates different thicknesses.
            reach = float(np.linalg.norm(far - pose[head]))
            stretch = np.clip(reach / (scale * length), *STRETCH)
            spin = (np.eye(3) + (stretch - 1.0) * np.outer(want, want)) @ spin

            if tail is None:
                # The palm is the one part with a second rigid measurement, and
                # it needs it. Every other bone is a line: one length, one
                # stretch, done. The palm is a plate, and fitting a plate to a
                # hand by its length alone leaves its width to the global
                # scale -- which is a 3D scale, so on a hand turned away from
                # the camera it came out too wide. Measured across five
                # captures, the length landed within 4% every time while the
                # knuckle span was drawn up to 35% too wide, and always on the
                # two captures where the hand was most turned.
                #
                # The knuckle line is rigid, the tracker reports it, and it is
                # perpendicular to the wrist run. So it gets its own stretch,
                # along an axis square to the first, and the two do not fight.
                span = pose[lm.PINKY_MCP] - pose[lm.INDEX_MCP]
                side = span - want * float(want @ span)
                # The same measurement on the model, taken square to the
                # model's own wrist run so the two ratios describe the same
                # thing.
                rest_span = rest_lm[lm.PINKY_MCP] - rest_lm[lm.INDEX_MCP]
                rest_dir = _unit(rest_far - rest_lm[head])
                rest_side = rest_span - rest_dir * float(rest_dir @ rest_span)
                wide = float(np.linalg.norm(side))
                rest_wide = float(np.linalg.norm(rest_side))
                if wide > 1e-9 and rest_wide > 1e-9:
                    fat = np.clip(wide / (scale * rest_wide), *STRETCH)
                    axis = side / wide
                    spin = (np.eye(3)
                            + (fat - 1.0) * np.outer(axis, axis)) @ spin
            rot[index] = spin * scale
            off[index] = pose[head] - rot[index] @ rest_lm[head]

        verts, normals = model["verts"], model["normals"]
        out = np.zeros_like(verts)
        nout = np.zeros_like(normals)
        for column in range(model["joints"].shape[1]):
            which = model["joints"][:, column]
            share = model["weights"][:, column]
            hot = share > 1e-6
            if not hot.any():
                continue
            take = which[hot]
            moved = np.einsum("nij,nj->ni", rot[take], verts[hot]) + off[take]
            out[hot] += moved * share[hot, None]
            nout[hot] += np.einsum("nij,nj->ni", rot[take],
                                   normals[hot]) * share[hot, None]
        nout /= np.maximum(np.linalg.norm(nout, axis=1, keepdims=True), 1e-12)

        faces = model["faces"]
        return {"verts": out, "normals": nout, "faces": faces,
                "material": model["material"], "vcolour": model["vcolour"],
                # Which bone drove each triangle, for draw ordering: a plate is
                # rigid, so its first vertex names it.
                "part": model["joints"][faces[:, 0], 0],
                "colours": model["colours"], "mirrored": model is not self.model}
