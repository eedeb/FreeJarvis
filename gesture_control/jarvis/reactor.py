"""The arc reactor in the corner: a real 3D orb, projected every frame.

Drawn rather than loaded -- a PNG would have to be licensed, could not
animate, and could not answer the one question this graphic exists to answer,
which is what Jarvis is doing right now.

**Why it is genuinely 3D and not a drawing of a sphere.**  The earlier version
was flat: concentric `cv2.ellipse` arcs that spun in the plane of the screen.
Nothing about that reads as depth, because nothing ever passes in front of
anything else.  This holds a point cloud on the surface of a unit sphere,
rotates it with a real matrix, divides by depth for perspective, and shades
each point by how near it ended up.  Turning it then does what turning a solid
object does: the far side dims and slows, the near side brightens and sweeps
across, structure crosses over itself, and the silhouette stays put while the
detail moves through it.  That parallax is the whole effect, and it is not
something a 2D drawing can fake.

**Why points rather than polygons.**  Thousands of shaded points are one
numpy matrix multiply, one divide and one scatter-add -- no rasteriser, no GL
context, no depth buffer, about a millisecond at the size this runs at.  A
mesh would need all of those to look worse, because the thing being drawn is
a cloud of filaments and sparks rather than a surface.

Everything is additive.  The overlay decides which pixels are solid by
comparing the drawn frame against the bare camera frame, so brightening the
camera is what makes the orb opaque -- there is no separate mask to maintain,
and the glow feathers into the wash on its own.

Four states, each with its own motion, readable from across the room:

    idle        slow drift, dim              nothing is happening
    listening   bright, opened up            the microphone is open
    thinking    fast spin                    FreeClaw has the question
    speaking    pulses with the waveform     a reply is being read out
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# One colour of light, the same cyan the glove is projected in, so the whole
# interface reads as coming from one machine. The core runs hotter and whiter
# than the rim, which is what makes it glow rather than merely be blue.
# BGR, because everything downstream is OpenCV.
DEEP = np.array((92, 42, 8), np.float32)       # far side, in shadow
WARM = np.array((246, 170, 44), np.float32)    # the body of the orb
HOT = np.array((255, 240, 198), np.float32)    # near side and core

# How much of the background to take away behind the orb, at the centre.
# Subtracted from the frame before the light is added, and it has to be this
# strong: the overlay is see-through, so a bright wallpaper shows through and
# fine amber filaments over pale blue sky simply disappear. Taking the
# backdrop down to near-black first is what gives the orb something to sit on.
# It also earns its own opacity for free -- the overlay makes a pixel solid
# when it differs from the bare camera, and darkening is a difference.
SHADOW = np.array((96, 104, 112), np.float32)
# Where the darkening starts fading out, and where it reaches nothing, as
# fractions of the canvas half-width. The outer figure is past the orb's own
# edge so the disc is never visible as a rim.
SHADOW_SOLID, SHADOW_EDGE = 0.62, 1.02

# How far in from the top-right corner, as a fraction of the orb's size.
INSET = 0.34

# Perspective. The camera sits FOCAL sphere-radii away, so a point on the near
# pole is magnified by FOCAL/(FOCAL-1) and one on the far pole shrunk by
# FOCAL/(FOCAL+1). Low numbers exaggerate the depth: 3.4 is enough to see the
# near side bulge without the far side collapsing to nothing.
FOCAL = 3.4

# The sphere fills this fraction of the canvas, before perspective.
FILL = 0.34

# How bright the far side of the sphere is relative to the near side. Not zero:
# a shell you cannot see through stops looking like a shell and starts looking
# like a disc.
BACKSIDE = 0.22

STATES = ("idle", "listening", "thinking", "speaking")

# Per state: overall brightness, how fast the orb turns, and how hard it
# breathes. "speaking" breathes hardest because its level is the real playback
# waveform, so the orb genuinely moves with the words.
MOOD = {
    "idle":      (0.70, 0.16, 0.05),
    "listening": (1.05, 0.28, 0.16),
    "thinking":  (0.95, 0.95, 0.09),
    "speaking":  (1.05, 0.22, 0.34),
}

# The wake-word flare: the brightness it adds at full, what fraction survives
# each 1/60s, and how far it pushes the shell outward. 0.86 gives a flash of
# about a third of a second -- long enough to catch out of the corner of an
# eye, short enough not to be mistaken for a state.
FLARE_LIFT, FLARE_DECAY, FLARE_SWELL = 1.4, 0.86, 0.10

# The cloud. Filaments are short curved traces over the surface -- the thing
# that makes it look built rather than sprayed; parallels and meridians are the
# structure they hang on; sparks are loose points that catch the light.
FILAMENTS, FILAMENT_STEPS = 132, 26
PARALLELS = (-0.72, -0.44, -0.15, 0.15, 0.44, 0.72)
MERIDIANS = 9
RING_STEPS = 132
SPARKS = 1500

# A second, smaller shell inside the first, turning the other way. Nested
# shells are what stop the orb reading as a hollow bauble: something visibly
# behind something else, both moving, is the strongest depth cue available
# and it costs one more matrix multiply.
INNER_RADIUS, INNER_SPARKS, INNER_FILAMENTS = 0.54, 700, 46

# The tilt of the axis it spins about. Straight up looks like a globe on a
# stand; leaning it slightly is what makes the rotation read as an object
# turning in space rather than a wheel.
TILT = math.radians(17.0)


def _rotation(yaw: float, tilt: float) -> np.ndarray:
    """Spin about the vertical, then lean the whole thing towards the viewer."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(tilt), math.sin(tilt)
    spin = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), np.float32)
    lean = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)), np.float32)
    return lean @ spin


def _build_cloud() -> tuple[np.ndarray, np.ndarray]:
    """Every point on the sphere, once. Returns (positions, brightness).

    Seeded, so the orb is the same object every run rather than a different
    scatter each launch -- it should look like a thing the app owns, not like
    static.
    """
    rng = np.random.default_rng(20140801)
    points, weights = [], []

    # Parallels and meridians: the armature. Drawn as broken arcs rather than
    # closed circles, because an unbroken ring reads as a drawn outline while
    # a broken one reads as structure seen through other structure.
    for lat in PARALLELS:
        radius = math.sqrt(max(0.0, 1.0 - lat * lat))
        angle = np.linspace(0.0, 2.0 * math.pi, RING_STEPS, endpoint=False)
        keep = rng.random(RING_STEPS) > 0.22
        points.append(np.stack((np.cos(angle) * radius,
                                np.full(RING_STEPS, lat, np.float32),
                                np.sin(angle) * radius), axis=1)[keep])
        weights.append(np.full(int(keep.sum()), 0.85, np.float32))

    for i in range(MERIDIANS):
        lon = 2.0 * math.pi * i / MERIDIANS
        t = np.linspace(-math.pi / 2, math.pi / 2, RING_STEPS)
        keep = rng.random(RING_STEPS) > 0.34
        points.append(np.stack((np.cos(t) * math.cos(lon), np.sin(t),
                                np.cos(t) * math.sin(lon)), axis=1)[keep])
        weights.append(np.full(int(keep.sum()), 0.55, np.float32))

    # Filaments: from a random point, walk a short way along a great circle,
    # curving as it goes. Great circles because a path that stays on the
    # surface is what makes the sphere legible when it turns -- a straight
    # line in 3D would cut through it and break the illusion.
    for _ in range(FILAMENTS):
        start = rng.normal(size=3).astype(np.float32)
        start /= np.linalg.norm(start)
        side = np.cross(start, rng.normal(size=3).astype(np.float32))
        side /= max(np.linalg.norm(side), 1e-6)
        span = rng.uniform(0.25, 1.15)
        curve = rng.uniform(-0.55, 0.55)
        t = np.linspace(0.0, span, FILAMENT_STEPS, dtype=np.float32)
        drift = t + curve * t * t
        arc = (np.outer(np.cos(drift), start) + np.outer(np.sin(drift), side))
        points.append(arc.astype(np.float32))
        # Bright at the head, fading along the tail: a filament with a
        # direction reads as drawn by something, not as a scratch.
        weights.append(np.linspace(1.0, 0.25, FILAMENT_STEPS, dtype=np.float32)
                       * rng.uniform(0.6, 1.0))

    # Sparks, thickened towards the equator so the orb has a waist.
    spark = rng.normal(size=(SPARKS, 3)).astype(np.float32)
    spark /= np.linalg.norm(spark, axis=1, keepdims=True)
    spark[:, 1] *= 0.82
    spark /= np.linalg.norm(spark, axis=1, keepdims=True)
    points.append(spark)
    weights.append(rng.uniform(0.12, 0.75, SPARKS).astype(np.float32))

    return (np.concatenate(points).astype(np.float32),
            np.concatenate(weights).astype(np.float32))


def _build_inner() -> tuple[np.ndarray, np.ndarray]:
    """The inner shell: sparser, and mostly filaments so it reads as a cage."""
    rng = np.random.default_rng(1979)
    points, weights = [], []
    for _ in range(INNER_FILAMENTS):
        start = rng.normal(size=3).astype(np.float32)
        start /= np.linalg.norm(start)
        side = np.cross(start, rng.normal(size=3).astype(np.float32))
        side /= max(np.linalg.norm(side), 1e-6)
        t = np.linspace(0.0, rng.uniform(0.4, 1.6), FILAMENT_STEPS, dtype=np.float32)
        arc = np.outer(np.cos(t), start) + np.outer(np.sin(t), side)
        points.append(arc.astype(np.float32))
        weights.append(np.linspace(0.9, 0.2, FILAMENT_STEPS, dtype=np.float32))
    spark = rng.normal(size=(INNER_SPARKS, 3)).astype(np.float32)
    spark /= np.linalg.norm(spark, axis=1, keepdims=True)
    points.append(spark)
    weights.append(rng.uniform(0.10, 0.55, INNER_SPARKS).astype(np.float32))
    return (np.concatenate(points).astype(np.float32) * INNER_RADIUS,
            np.concatenate(weights).astype(np.float32))


_CLOUD, _WEIGHT = _build_cloud()
_INNER, _INNER_WEIGHT = _build_inner()

# How far the wide bloom is shrunk before it is blurred. A Gaussian of sigma s
# on an image reduced by k is a Gaussian of s*k on the original, at 1/k^2 the
# pixels with a 1/k kernel -- so this is the same glow for a sixteenth of the
# arithmetic. Measured here: 3.51ms exact against 0.30ms at a quarter, with the
# worst pixel differing by 0.8 of 255. On a glow that is not a difference
# anyone can see, and it was the single most expensive thing the overlay did.
BLOOM_SHRINK = 4

# Depth shading, as a table rather than as arithmetic. The colour of a point
# is a function of one number -- how near it is -- so the whole ramp is 256
# answers, and looking them up costs a fifth of computing eight thousand of
# them. Built once at import.
_SHADE_STEPS = 256
_ramp = np.linspace(0.0, 1.0, _SHADE_STEPS, dtype=np.float32)[:, None]
_hot_ramp = np.clip((_ramp - 0.86) / 0.14, 0.0, 1.0)
_SHADE = ((DEEP[None, :] * (1.0 - _ramp) + WARM[None, :] * _ramp)
          * (1.0 - _hot_ramp) + HOT[None, :] * _hot_ramp).astype(np.float32)


class Reactor:
    """One orb, holding its own rotation and animation phase between frames."""

    def __init__(self, size: int) -> None:
        self.size = max(48, int(size))
        self.state = "idle"
        # 0..1, how loud the user or the reply currently is. Decays on its own
        # so a dropped update settles rather than freezing at full.
        self.level = 0.0
        self._flare = 0.0
        self._phase = 0.0                        # breathing
        self._yaw = 0.0                          # the spin, in radians
        self._build()

    def _build(self) -> None:
        size = self.size
        self._canvas = np.zeros((size, size, 3), np.float32)
        self._out = np.zeros((size, size, 3), np.uint8)
        # Every intermediate the render needs, allocated once. At 280px these
        # are a quarter of a megabyte apiece and numpy was making four of them
        # per frame -- which cost more than the blur they were feeding.
        self._art = np.empty((size, size, 3), np.float32)
        self._tight = np.empty((size, size, 3), np.float32)
        self._bloom = np.empty((size, size, 3), np.float32)
        small = max(8, size // BLOOM_SHRINK)
        self._small = np.empty((small, small, 3), np.float32)
        self._small_size = (small, small)

        # The backdrop and the rim never animate, so they are drawn once.
        # Blurring them per frame cost more than everything else together.
        half = np.abs(np.arange(size, dtype=np.float32) - (size - 1) / 2.0)
        reach = np.sqrt(half[None, :] ** 2 + half[:, None] ** 2) / (size / 2.0)
        # Smoothstep from solid to nothing, so the disc has no visible edge.
        fade = np.clip((SHADOW_EDGE - reach) / (SHADOW_EDGE - SHADOW_SOLID),
                       0.0, 1.0)
        fade = fade * fade * (3.0 - 2.0 * fade)
        self._shadow = (fade[:, :, None] * SHADOW[None, None, :]).astype(np.uint8)
        # The same falloff as an alpha floor. The orb's backdrop is opaque
        # because it says so, not because darkening the camera happened to
        # change the pixels enough for the overlay to notice.
        self._solid = np.clip(fade * 305.0, 0.0, 255.0).astype(np.uint8)

        # A faint limb around the silhouette. Real spheres are brightest at
        # the edge, where the line of sight passes through the most of them,
        # and that single cue does more for "this is a ball" than any amount
        # of interior detail.
        grid = (np.arange(size, dtype=np.float32) - size / 2.0) / (size * FILL)
        rr = np.sqrt(grid[None, :] ** 2 + grid[:, None] ** 2)
        # Weak on purpose. Points spread uniformly over a sphere pile up at
        # the silhouette when projected -- the density goes as 1/sqrt(1-r^2)
        # -- so most of the limb is already there for free, and painting a
        # full-strength ring on top of it just looks like a drawn circle.
        limb = np.clip(1.0 - np.abs(rr - 0.97) / 0.13, 0.0, 1.0) ** 2
        limb *= (rr < 1.10)
        self._limb = limb[:, :, None] * WARM[None, None, :] * 0.13

        # The core, once: a tight bright centre with a wide soft falloff.
        k = max(9, int(size * 0.42) | 1)
        g = (np.arange(k, dtype=np.float32) - k // 2) / (k / 2.0)
        d = np.sqrt(g[None, :] ** 2 + g[:, None] ** 2)
        glow = np.exp(-(d / 0.16) ** 2) * 1.15 + np.exp(-(d / 0.46) ** 2) * 0.30
        self._core = np.clip(glow, 0.0, 1.6)[:, :, None] * HOT[None, None, :]

        # Corner brackets, static: they frame the moving part and give the
        # whole thing somewhere to sit rather than floating.
        chrome = np.zeros((size, size, 3), np.uint8)
        arm, pad = int(size * 0.13), int(size * 0.03)
        edge = tuple(int(c * 0.75) for c in WARM)
        for cx, cy, dx, dy in ((pad, pad, 1, 1), (size - pad, pad, -1, 1),
                               (pad, size - pad, 1, -1),
                               (size - pad, size - pad, -1, -1)):
            cv2.line(chrome, (cx, cy), (cx + dx * arm, cy), edge, 1, cv2.LINE_AA)
            cv2.line(chrome, (cx, cy), (cx, cy + dy * arm), edge, 1, cv2.LINE_AA)
        self._chrome = chrome.astype(np.float32)

    def set_state(self, state: str, level: float | None = None) -> None:
        if state in STATES:
            self.state = state
        if level is not None:
            self.level = float(min(max(level, 0.0), 1.0))

    def flare(self) -> None:
        """Heard. Punch the brightness up so it is obvious across a room."""
        self._flare = 1.0

    # -- the render --------------------------------------------------------

    def _render(self, dt: float) -> np.ndarray:
        bright, spin, breath = MOOD.get(self.state, MOOD["idle"])
        self._phase += dt
        self._yaw += dt * spin * 2.0 * math.pi * 0.16
        self.level *= 0.90
        # Frame-rate independent decay: the preview redraws at whatever the
        # camera manages, so a fixed per-frame factor would make the flare
        # last twice as long on a slow machine.
        self._flare *= FLARE_DECAY ** (dt * 60.0)
        if self._flare < 0.01:
            self._flare = 0.0

        size = self.size
        mid = size / 2.0
        pulse = 1.0 + breath * math.sin(self._phase * 2.2) + 0.45 * self.level * breath
        gain = bright * min(pulse, 1.35) + FLARE_LIFT * self._flare
        swell = 1.0 + FLARE_SWELL * self._flare + 0.05 * self.level

        canvas = self._canvas
        canvas[:] = 0.0
        radius = size * FILL * swell
        # The two shells turn opposite ways, so one is always visibly moving
        # across the other. Same shading for both -- depth is depth.
        self._shell(canvas, _CLOUD, _WEIGHT, self._yaw, mid, radius, gain)
        self._shell(canvas, _INNER, _INNER_WEIGHT, -self._yaw * 1.6, mid,
                    radius, gain * 0.85)

        # The core, as a precomputed falloff rather than a filled circle: a
        # hard-edged disc reads as a sticker on the front, where something
        # that fades into the shell around it reads as light inside it.
        k = self._core.shape[0]
        lo = int(mid) - k // 2
        canvas[lo:lo + k, lo:lo + k] += self._core * (
            gain * (1.0 + 0.5 * self.level + 0.9 * self._flare))

        # scaleAdd rather than `canvas += x * k`: the numpy form builds a
        # whole float image for the product and throws it away, twice a frame.
        cv2.scaleAdd(self._limb, gain, canvas, dst=canvas)
        cv2.scaleAdd(self._chrome, bright, canvas, dst=canvas)

        # Bloom, in two passes. A tight one thickens every filament into
        # something with a body; a wide, weak one is the haze that makes it
        # read as light rather than as paint. The wide one is done on a
        # quarter-size copy -- see BLOOM_SHRINK.
        art = self._art
        np.clip(canvas, 0.0, 255.0, out=art)
        cv2.GaussianBlur(art, (0, 0), size * 0.011, dst=self._tight)
        cv2.resize(art, self._small_size, dst=self._small,
                   interpolation=cv2.INTER_AREA)
        cv2.GaussianBlur(self._small, (0, 0), size * 0.055 / BLOOM_SHRINK,
                         dst=self._small)
        cv2.resize(self._small, (size, size), dst=self._bloom,
                   interpolation=cv2.INTER_LINEAR)

        cv2.scaleAdd(self._tight, 0.85, art, dst=art)
        cv2.scaleAdd(self._bloom, 0.55, art, dst=art)
        np.clip(art, 0.0, 255.0, out=art)
        return art.astype(np.uint8, copy=False)

    def _shell(self, canvas, cloud, weight, yaw, mid, radius, gain) -> None:
        """Rotate one shell, project it, shade it by depth and splat it."""
        turned = cloud @ _rotation(yaw, TILT).T
        depth = turned[:, 2]
        scale = FOCAL / (FOCAL - depth)
        xs = mid + turned[:, 0] * radius * scale
        ys = mid - turned[:, 1] * radius * scale

        # Near points are brighter and hotter; far ones sink towards the
        # shadow colour. This is the shading, and it is the other half of what
        # makes the thing look solid. Taken from `_SHADE` rather than computed:
        # it is a function of one number, so 256 answers cover every point on
        # both shells, and the table is a gather where the arithmetic was four
        # allocations the width of the cloud.
        near = (depth + 1.0) * 0.5
        lit = weight * (BACKSIDE + (1.0 - BACKSIDE) * near ** 1.4) * gain
        shade = np.clip(near * (_SHADE_STEPS - 1), 0, _SHADE_STEPS - 1)
        colour = _SHADE[shade.astype(np.int32)]
        self._splat(canvas, xs, ys, colour * lit[:, None])

    @staticmethod
    def _splat(canvas: np.ndarray, xs: np.ndarray, ys: np.ndarray,
               colour: np.ndarray) -> None:
        """Add every point into the canvas, spread over the four pixels it
        falls between.

        Bilinear rather than rounded: at this size a point moves less than a
        pixel per frame, and rounding makes the whole cloud jitter between
        pixels as it turns -- which looks like noise, not like rotation.

        Scattered with `bincount` rather than the obvious `np.add.at`.  They
        compute the same thing, but `add.at` is the unbuffered ufunc path and
        took 7.2ms of a 9.3ms frame here -- more than everything else in the
        renderer put together.  `bincount` over flattened indices is the same
        accumulation done in one pass per channel, and costs about a tenth of
        that.
        """
        height, width = canvas.shape[:2]
        x0 = np.floor(xs).astype(np.int32)
        y0 = np.floor(ys).astype(np.int32)
        fx, fy = xs - x0, ys - y0

        flat, parts = [], []
        for dx, dy, weight in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                               (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            px, py = x0 + dx, y0 + dy
            good = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            flat.append(py[good] * width + px[good])
            parts.append(colour[good] * weight[good, None])
        if not flat:
            return
        index = np.concatenate(flat)
        contribution = np.concatenate(parts)
        cells = height * width
        rows = canvas.reshape(cells, 3)
        for channel in range(3):
            rows[:, channel] += np.bincount(
                index, weights=contribution[:, channel], minlength=cells)

    def draw(self, frame: np.ndarray, dt: float, top: int | None = None,
             solid: np.ndarray | None = None) -> tuple[int, int, int, int]:
        """Draw into the top-right of `frame`. Returns the box it used.

        `top` is where the orb may start, so it can be pushed below whatever
        else is claiming the corner. `solid` is the overlay's alpha floor,
        which the orb's backdrop stamps itself into.
        """
        height, width = frame.shape[:2]
        size = min(self.size, width, height)
        if size != self.size:
            self.size = size                     # screen smaller than expected
            self._build()
        inset = int(size * INSET)
        x0 = max(0, width - size - inset // 2)
        y0 = min(inset // 2 if top is None else top, max(0, height - size))
        x1, y1 = min(width, x0 + size), min(height, y0 + size)

        art = self._render(dt)[: y1 - y0, : x1 - x0]
        roi = frame[y0:y1, x0:x1]
        # Darken under the orb, then add the light. Two steps because a single
        # add over a bright camera frame washes the amber out to white.
        cv2.subtract(roi, self._shadow[: y1 - y0, : x1 - x0], dst=roi)
        cv2.add(roi, art, dst=roi)
        if solid is not None:
            patch = solid[y0:y1, x0:x1]
            cv2.max(patch, self._solid[: y1 - y0, : x1 - x0], dst=patch)
        return x0, y0, x1, y1
