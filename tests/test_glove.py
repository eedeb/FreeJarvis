"""The rigged glove: the model contract, the skinning, and both renderers.

Measured on real captures rather than on a generated hand. The synthetic hand
in synthetic_hand.py is still worth having -- it poses and tilts on demand --
but it has been wrong about this overlay five separate times, so the checks
that decide whether the glove is on the hand are asked of frames the tracker
actually saw.

Press D while the app is running to add one.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cv2
import numpy as np

from gesture_control import landmarks as lm
from gesture_control.config import Settings
from gesture_control.glove import (CHAIN, Glove, PALM_MARKS, SEGMENTS,
                                   _chirality, _ortho, tracked_pose)
from gesture_control.render import Renderer, draw_flat
from gesture_control.tracker import HandFrame

ok = True
GREEN = (0, 255, 0)
# How thick each digit is at the knuckle and at the tip, in knuckle spans, for
# rebuilding the hand's own silhouette from the landmarks. Hand anthropometry,
# not measured off the model -- the point is to check the glove against a hand
# rather than against itself.
FLESH = {"thumb": (0.30, 0.22), "index": (0.24, 0.17), "middle": (0.25, 0.17),
         "ring": (0.23, 0.16), "pinky": (0.21, 0.15)}


# Material indices, in the order tools/bl_gauntlet.py names them.
RED, GOLD = 0, 1


def hand_side(hand, screen):
    """Which side of the hand faces the camera, from the landmarks alone.

    The palm frame's normal leaves the palm on a right hand and the back on a
    left one -- that is the whole basis of the chirality test -- so given the
    chirality, the sign of the normal's depth says which side is towards the
    lens. Nothing about the model enters this.
    """
    pose, _ = tracked_pose(hand, screen, glove.model["heads"],
                           glove.model["tails"], glove.model["bone_len"])
    frame = _ortho(pose[lm.PINKY_MCP] - pose[lm.INDEX_MCP],
                   pose[PALM_MARKS, :].mean(axis=0) - pose[lm.WRIST])
    towards = frame[2, 2] < 0.0
    if _chirality(pose, frame) > 0:
        towards = not towards
    return "back" if towards else "palm"


def glove_side(posed):
    """Which side of the glove the camera can see, by counting its plate."""
    tri = posed["verts"][posed["faces"]]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    plate = glove.model["joints"][:, 0][posed["faces"][:, 0]] == 0
    seen = plate & (normal[:, 2] < 0.0)
    gold = int((seen & (posed["material"] == GOLD)).sum())
    red = int((seen & (posed["material"] == RED)).sum())
    return "palm" if gold > red else "back"


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


ROOT = pathlib.Path(__file__).resolve().parent.parent
CAPTURES = sorted((ROOT / "captures").glob("*.npz"))


def load(path):
    data = np.load(path)
    norm = data["norm"].astype(float)
    hand = HandFrame(stamp=0.0, norm=norm, metric=norm,
                     handedness=str(data["handedness"]), confidence=1.0,
                     world=data["world"].astype(float))
    return hand, data["screen"].astype(float), data["frame"]


print("The model is a rig, and says so:")
glove = Glove()
check(f"it loads ({glove.reason or 'no error'})", glove.available)
if not glove.available:
    print("\nRESULT: SOME CHECKS FAILED")
    raise SystemExit(1)

model = glove.model
names = set(model["joint_names"])
# The bone names are the contract between tools/bl_gauntlet.py and this module.
# A model that does not honour it should fail loudly at load rather than pose
# something subtly wrong, which is the failure mode this whole rewrite exists
# to remove.
wanted = {f"{digit}_{segment}"
          for digit in CHAIN for segment in SEGMENTS + ("tip",)} | {"root"}
check(f"every bone the rig needs is present ({len(wanted)})",
      wanted <= names)
check("all 21 landmarks have a rest position",
      np.isfinite(model["rest_lm"]).all())
check("every vertex is weighted",
      bool(np.all(np.abs(model["weights"].sum(axis=1) - 1.0) < 1e-6)))
check("no vertex follows a bone that does not exist",
      int(model["joints"].max()) < len(model["joint_names"]))
# A hand is longer than it is wide and much wider than it is thick. This is not
# a fussy check: it catches a model exported in the wrong units or with a stray
# object welded into it, which is otherwise only visible as "the glove is huge".
extent = np.ptp(model["verts"], axis=0)
check(f"the model is hand-shaped ({np.round(extent * 1000).astype(int)} mm)",
      0.06 < extent.max() < 0.40 and extent.min() < 0.4 * extent.max())

print("\nIt lands on the hand:")
if not CAPTURES:
    print("      no captures/*.npz -- press D while the app is running")
posed_all = {}
for path in CAPTURES:
    hand, screen, frame = load(path)
    posed = glove.pose(hand, screen)
    check(f"{path.stem}: it poses", posed is not None)
    if posed is None:
        continue
    posed_all[path.stem] = (posed, screen, frame)
    surface = posed["verts"][:, :2]
    reach = float(np.linalg.norm(screen[lm.MIDDLE_MCP] - screen[lm.WRIST]))

    # Every joint the glove claims to cover should have armour within a few
    # pixels of it. This is the check that caught the root bone having no
    # transform at all, and the rig shearing its digits off their own bones.
    miss = [float(np.linalg.norm(surface - screen[joint], axis=1).min())
            for chain in CHAIN.values() for joint in chain[1:]]
    worst = max(miss)
    print(f"      {path.stem}: joints sit {np.median(miss):.1f}px from the "
          f"armour, worst {worst:.1f}px ({worst / reach:.2f} spans)")
    check(f"{path.stem}: the armour reaches every joint ({worst:.0f}px)",
          worst < 0.12 * reach)

    # ...and the glove has to be about the size of the hand. Too small stops
    # covering it; too large was the old renderer's failure, where the palm
    # grew into a disc wider than the frame.
    size = np.ptp(surface, axis=0) / reach
    print(f"      {path.stem}: the glove measures {np.round(size, 2)} spans")
    check(f"{path.stem}: it is hand-sized ({np.round(size, 1)})",
          1.4 < size[0] < 3.4 and 1.2 < size[1] < 3.4)

print("\nBoth renderers draw it:")
gpu = Renderer()
print(f"      GPU: {'available' if gpu.available else gpu.reason}")
for name, (posed, screen, frame) in posed_all.items():
    covered = {}
    for tag, drawer in (("cpu", draw_flat),
                        ("gpu", gpu.draw if gpu.available else None)):
        if drawer is None:
            continue
        canvas = np.full(frame.shape, GREEN, np.uint8)
        drew = drawer(canvas, posed)
        check(f"{name}: the {tag} renderer draws", drew)
        if drew:
            covered[tag] = np.any(canvas != GREEN, axis=2)

    # The palm the tracker reports, and how much of it ends up under armour.
    ring = screen[[lm.WRIST, lm.THUMB_CMC, lm.THUMB_MCP, lm.INDEX_MCP,
                   lm.MIDDLE_MCP, lm.RING_MCP, lm.PINKY_MCP]]
    ring = ring[np.argsort(np.arctan2(*(ring - ring.mean(axis=0)).T[::-1]))]
    quad = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(quad, [ring.astype(np.int32)], 1)
    for tag, on in covered.items():
        share = float((on & (quad > 0)).sum() / max(quad.sum(), 1))
        print(f"      {name}: {tag} covers {share:5.1%} of the palm")
        check(f"{name}: the {tag} renderer covers the palm ({share:.0%})",
              share > 0.90)

    # ...and the whole hand, not just the palm. The silhouette is rebuilt from
    # the landmarks: the palm as a filled polygon and every bone as a capsule
    # of about the right thickness, using hand anthropometry -- a finger is
    # roughly 0.22 knuckle spans across at the knuckle and 0.16 at the tip.
    #
    # The palm polygon on its own is not enough. It passed at 100% on a capture
    # where the first web space was bare, because the web is outside it.
    skin = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(skin, [ring.astype(np.int32)], 1)
    reach = float(np.linalg.norm(screen[lm.PINKY_MCP] - screen[lm.INDEX_MCP]))
    for digit, chain in CHAIN.items():
        wide, thin = FLESH[digit]
        for k in range(1, len(chain) - 1):
            step = (k - 1) / max(len(chain) - 3, 1)
            fat = int(max(reach * (wide + (thin - wide) * step) * 0.5, 2))
            cv2.line(skin, tuple(screen[chain[k]].astype(int)),
                     tuple(screen[chain[k + 1]].astype(int)), 1, fat * 2)
            cv2.circle(skin, tuple(screen[chain[k]].astype(int)), fat, 1, -1)
        cv2.circle(skin, tuple(screen[chain[-1]].astype(int)),
                   int(max(reach * thin * 0.5, 2)), 1, -1)
    for tag, on in covered.items():
        share = float((on & (skin > 0)).sum() / max(skin.sum(), 1))
        print(f"      {name}: {tag} covers {share:5.1%} of the whole hand")
        check(f"{name}: the {tag} renderer covers the hand ({share:.0%})",
              share > 0.95)

    # The side of the glove facing the camera has to be the side of the hand
    # facing the camera: gold palm and repulsor when you can see your palm, red
    # plate when you can see your knuckles.
    #
    # Nothing else here can catch this. A glove drawn inside out lands in the
    # same place, covers the same hand and leaves the same holes, so every
    # check above passed for as long as the palm was rendering on the back.
    #
    # Which way the hand faces is worked out from the landmarks alone. The
    # frame's normal leaves the palm on a right hand and the back on a left
    # one -- that is what makes the thumb's offset a chirality test -- so given
    # the chirality, the sign of the normal's depth says which side is towards
    # the lens.
    hand, _, _ = load(next(p for p in CAPTURES if p.stem == name))
    wants = hand_side(hand, screen)
    shows = glove_side(posed)
    print(f"      {name}: the hand shows its {wants}, the glove its {shows}")
    check(f"{name}: the glove shows the {wants} of the hand", shows == wants)

    # Holes you can see through: background that cannot be reached from
    # outside without crossing armour. A flood fill has no kernel to be the
    # wrong size, which a morphological close does -- on a curled hand the
    # notches between the knuckles are about the size of one.
    for tag, on in covered.items():
        escape = (~on).astype(np.uint8)
        cv2.floodFill(escape, np.zeros((frame.shape[0] + 2,
                                        frame.shape[1] + 2), np.uint8),
                      (0, 0), 0)
        holes = float(escape.sum()) / max(int(on.sum()), 1)
        print(f"      {name}: {tag} leaves {holes:5.2%} holes")
        check(f"{name}: the {tag} render is solid ({holes:.2%})", holes < 0.02)

    # The two paths draw the same glove. They shade it differently -- one per
    # pixel, one per triangle -- so only the silhouette is comparable, and the
    # depth buffer legitimately hides a little that the sorted fills do not.
    if len(covered) == 2:
        agree = float((covered["cpu"] & covered["gpu"]).sum()
                      / max((covered["cpu"] | covered["gpu"]).sum(), 1))
        print(f"      {name}: the two renderers agree on {agree:.0%} of it")
        check(f"{name}: GPU and CPU draw the same glove ({agree:.0%})",
              agree > 0.85)

print("\nProjected light, not painted metal:")
# The silhouette check above deliberately ignores colour, so nothing there
# would notice the glove being rendered completely differently. These are the
# properties that make it a hologram rather than a cyan-painted glove.
from gesture_control.render import HOLO_COVER, Renderer as _R

if posed_all and Renderer is not None:
    name, (posed, screen, frame) = next(iter(posed_all.items()))
    gpu = _R()
    if gpu.available:
        hand_only = frame.copy()
        floor = np.zeros(frame.shape[:2], np.uint8)

        gpu.set_style(True)
        holo = frame.copy()
        gpu.draw(holo, posed, solid=floor)
        gpu.set_style(False)
        metal = frame.copy()
        gpu.draw(metal, posed)

        touched = np.abs(holo.astype(int) - hand_only.astype(int)).max(axis=2) > 8
        inside = touched.copy()
        # Erode hard, so "inside" is the middle of a plate rather than its rim.
        inside = cv2.erode(inside.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        rim = touched & ~inside
        check(f"it draws something ({int(touched.sum())} px)", touched.sum() > 2000)

        # A hologram is bright along its edges and faint across its faces --
        # that single property is the whole look, and it is what a flat cyan
        # repaint would fail.
        lit_rim = holo[rim].astype(float).mean()
        lit_face = holo[inside].astype(float).mean()
        check(f"the edges are brighter than the faces "
              f"({lit_face:.0f} -> {lit_rim:.0f})", lit_rim > lit_face * 1.15)

        # The hand has to remain visible through it, or it is not a projection.
        before = hand_only[inside].astype(float).mean()
        after = holo[inside].astype(float).mean()
        check(f"the hand still shows through the faces "
              f"(was {before:.0f}, now {after:.0f})", after > before * 0.55)

        # One colour of light. Blue-dominant in BGR.
        blue, green, red = (holo[touched][:, i].astype(float).mean() for i in range(3))
        check(f"it is projected in one colour, not painted "
              f"(B{blue:.0f} G{green:.0f} R{red:.0f})", blue > red * 1.4)
        mb, mg, mr = (metal[touched].astype(float)[:, i].mean() for i in range(3))
        check(f"where the metal look is warm (B{mb:.0f} G{mg:.0f} R{mr:.0f})",
              mr > mb)

        # The bug worth a regression test: the overlay decides opacity from how
        # much a pixel differs from the bare camera, so a *translucent* glove
        # was faded a second time and nearly vanished. It now claims its own
        # coverage as an alpha floor.
        check(f"it claims its coverage from the overlay ({int(floor.max())})",
              floor.max() == 255)
        claimed = (floor > 0)
        check("and claims it everywhere it drew",
              float((touched & claimed).sum()) / max(touched.sum(), 1) > 0.9)
        check("but nowhere it did not", float((claimed & ~touched).sum())
              / max(claimed.sum(), 1) < 0.35)

check("the holo look is the default", Settings().gauntlet_style == "holo")
check("and the metal one is still reachable",
      Settings(gauntlet_style="solid").gauntlet_style == "solid")

print("\nA left hand gets a left glove:")
if posed_all:
    name, (posed, screen, frame) = next(iter(posed_all.items()))
    hand, screen, _ = load(next(p for p in CAPTURES if p.stem == name))
    # Reflect the hand across the middle of the frame. Mirroring the landmarks
    # is exactly what a camera does to the other hand, so the glove has to come
    # back reflected too -- and the model is a right hand, so this is the path
    # that reflects it.
    flipped = screen.copy()
    flipped[:, 0] = frame.shape[1] - flipped[:, 0]
    other = HandFrame(stamp=0.0, norm=hand.norm, metric=hand.metric,
                      handedness=hand.handedness, confidence=1.0,
                      world=hand.world * np.array([-1.0, 1.0, 1.0]))
    got = glove.pose(other, flipped)
    check("the other hand poses", got is not None)
    if got is not None:
        check("...and gets the mirrored model", got["mirrored"] != posed["mirrored"])
        reach = float(np.linalg.norm(flipped[lm.MIDDLE_MCP] - flipped[lm.WRIST]))
        miss = max(float(np.linalg.norm(got["verts"][:, :2] - flipped[j],
                                        axis=1).min())
                   for chain in CHAIN.values() for j in chain[1:])
        print(f"      mirrored: joints sit within {miss:.1f}px of the armour")
        check(f"...and it still reaches every joint ({miss:.0f}px)",
              miss < 0.12 * reach)

print("\nIt keeps up with the camera:")
# Measured against this machine as it is right now, not against a number
# written down on an idle one.
#
# The history is worth keeping, because the obvious fix was tried and did not
# work. These were absolute millisecond bounds, and they failed on code that
# was in fact *faster* than what it replaced -- so they were changed to take
# the minimum of 25 runs instead of 6, on the theory that the minimum would
# find a quiet slot even on a busy machine. It does not: with the app itself
# running, or a build, or anything else with the GPU, all 25 samples are slow
# and the minimum is slow with them. Measured here, the CPU path came out at
# 7ms, 23ms and 32ms on three consecutive runs of the same code.
#
# So the bound is a multiple of a reference workload timed the same way in the
# same run -- a blur over the same frame, which is the same order of cost and
# the same kind of work, and which slows down when the machine is busy exactly
# as the renderers do. What is being asserted is the thing actually worth
# asserting: that drawing the glove costs about what it has always cost
# relative to everything else, and has not doubled.
REFERENCE_SIGMA = 6.0
# The GPU path is checked against the reference, where the ratio is steady:
# 0.7-0.9 across runs whose absolute times vary by three times. The CPU path
# is not, because nothing here tracks it -- OpenCV's blur is threaded and the
# flat renderer is not, so the two feel a busy machine differently, and a
# single-threaded numpy reference measured 5ms, 16ms and 14ms on three
# consecutive runs of identical code. Three separate references were tried;
# none of them held.
#
# So the CPU path gets a ceiling rather than a budget, and the comment says
# what that means: it is a smoke test, not a performance assertion. 60ms is
# twice the camera's frame period, so it still catches the renderer becoming
# something that cannot possibly keep up, which is the regression worth
# having a test for. Asserting anything tighter than that on a machine
# somebody is using produces a red suite and no information, which is what
# this test did for most of a day.
GPU_BUDGET = 3.0                       # times the reference; idle is ~0.8
CPU_CEILING = 60.0                     # ms, a smoke bound and nothing more


def fastest(fn, frame, rounds=25):
    """The quickest of `rounds` goes, in milliseconds."""
    best = float("inf")
    for _ in range(rounds):
        canvas = frame.copy()
        start = time.perf_counter()
        fn(canvas)
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best


if posed_all:
    posed, screen, frame = next(iter(posed_all.values()))
    for _ in range(3):
        cv2.GaussianBlur(frame, (0, 0), REFERENCE_SIGMA)
    reference = fastest(lambda c: cv2.GaussianBlur(c, (0, 0), REFERENCE_SIGMA),
                        frame)
    print(f"      reference: a blur of the same frame costs {reference:.1f} ms "
          f"on this machine right now")
    for tag, drawer in (("cpu", draw_flat),
                        ("gpu", gpu.draw if gpu.available else None)):
        if drawer is None:
            continue
        for _ in range(3):
            drawer(frame.copy(), posed)
        best = fastest(lambda c: drawer(c, posed), frame)
        ratio = best / max(reference, 1e-6)
        print(f"      {tag}: {best:.1f} ms a frame at "
              f"{frame.shape[1]}x{frame.shape[0]} ({ratio:.1f}x the reference)")
        if tag == "gpu":
            check(f"the gpu renderer keeps up ({best:.1f}ms, {ratio:.1f}x the "
                  f"reference, budget {GPU_BUDGET:.0f}x)", ratio < GPU_BUDGET)
        else:
            check(f"the cpu renderer is not hopeless ({best:.1f}ms, under "
                  f"{CPU_CEILING:.0f})", best < CPU_CEILING)

print("\nIt stays out of the control path:")
check("the rig is the default", Settings().gauntlet_model == "rig")
check("and can be forced back to the drawn one",
      Settings(gauntlet_model="drawn").gauntlet_model == "drawn")
for module in ("glove", "render", "gltf"):
    src = (ROOT / "gesture_control" / f"{module}.py").read_text(encoding="utf-8")
    check(f"{module}.py imports nothing that could move the mouse",
          "mouse" not in src and "Mouse" not in src)

print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
