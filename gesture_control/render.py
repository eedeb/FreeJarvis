"""Draw the posed glove with a depth buffer, on the GPU.

What this replaces is a painter's algorithm: sort the triangles back to front,
paint them in that order, and hope nothing overlaps in a way the sort cannot
express. That hope is not available here. Measured on a captured hand, the palm
plate spans 316 pixels of depth and the finger plates 299, and the two ranges
sit almost exactly on top of each other -- so the palm is simultaneously in
front of some finger triangles and behind others, and *no* ordering of whole
parts is correct. The glove rendered with its palm pasted over its own fingers.

A depth buffer answers the question per pixel and the whole class of problem
goes away, along with the machinery built to work around it: the depth slabs,
the part ordering, the fill batching, the inflation of every triangle to hide
the cracks between neighbours, the morphological crack-and-seam filler and the
median blur that hid the faceting. Shading is per pixel too, so the quantised
palette goes as well.

Only the hand's own rectangle is drawn and read back. Reading a whole 1080p
frame off the GPU costs 3.7ms; a hand is a tenth of that area.
"""

from __future__ import annotations

import numpy as np

try:
    import moderngl
except Exception:                       # noqa: BLE001 - optional dependency
    moderngl = None

import cv2

# Light direction and the specular exponent, matching the look the flat
# renderer arrived at -- wrap lighting rather than plain Lambert, so a normal
# that is a little wrong reads as slightly darker plate instead of a black
# speck, plus a tight highlight, which is most of what separates painted metal
# from a flat coloured shape.
LIGHT = np.array([-0.42, -0.62, -0.66])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
SHINE, SPECULAR = 50.0, 0.30

# The hologram. BGR, and deliberately not the armour's red and gold: a
# projection is one colour of light, and the plates have to read as brightness
# within it rather than as paint. Cyan because that is what a Stark workbench
# projects, and because it is the furthest thing from skin -- an amber glove
# over an amber hand is one shape, a cyan one is two.
HOLO_TINT = (215.0, 168.0, 40.0)        # the body
HOLO_EDGE = (255.0, 232.0, 150.0)       # the rim, running towards white
# How solid a surface facing straight at you is. Low enough to see the hand
# through it, high enough that a flat plate is not a hole.
HOLO_FLOOR = 0.34
# Scanline pitch in pixels, and how far the red and blue channels are pulled
# apart in the composite. Both small: the moment either is obvious it stops
# looking like a projection and starts looking like a fault.
HOLO_SCAN = 3.0
HOLO_SPLIT = 1
# How hard the glove's own coverage is pushed into the overlay's alpha floor.
# Without this the hologram is dimmed twice: once by its own translucency, and
# again by the overlay, which decides opacity from how much a pixel differs
# from the bare camera -- and a faint hologram barely differs from it. Anything
# the glove touched should be as solid as the overlay can make it; what shows
# there is already the hand seen through the projection.
HOLO_COVER = 5.0
# That gain as a lookup. The claim runs on every pixel of the hand's rectangle
# every frame, and doing it in float -- multiply, clip, cast -- was two
# full-size temporaries for what is a function of one byte.
_COVER_LUT = np.clip(np.arange(256, dtype=np.float32) * HOLO_COVER,
                     0, 255).astype(np.uint8)
# How much of what is behind the hologram it hides. Zero would be pure added
# light, which loses the plates against a bright hand; one is ordinary alpha,
# which *darkens* the hand wherever the projection is dim -- and a projection
# that makes things darker is a shadow, not a hologram. Half keeps the hand
# readable underneath while the cyan still adds.
HOLO_OCCLUDE = 0.45
# Where the ramp stops brightening the paint and starts adding white, so a
# highlight can be brighter than the paint it sits on.
HILIGHT, HILIGHT_MIX = 0.89, 0.40
# Samples per pixel. The silhouette is the whole point: at one sample a
# generated mesh's clean edges are thrown away by the rasteriser.
SAMPLES = 4
# Padding round the hand, as a fraction of the region, so a plate that reaches
# past the landmarks is not clipped.
PAD = 0.10

_VERTEX = """
#version 330
in vec3 in_pos;
in vec3 in_nrm;
in vec3 in_col;
uniform vec4 bounds;      // x0, y0, width, height of the region, in pixels
uniform vec2 depth;       // near, far
out vec3 v_nrm;
out vec3 v_col;
void main() {
    vec2 p = (in_pos.xy - bounds.xy) / bounds.zw;
    // Screen y runs down the image and clip y runs up it.
    float z = (in_pos.z - depth.x) / max(depth.y - depth.x, 1e-6);
    gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, z * 2.0 - 1.0, 1.0);
    v_nrm = in_nrm;
    v_col = in_col;
}
"""

_FRAGMENT = """
#version 330
in vec3 v_nrm;
in vec3 v_col;
out vec4 f_col;
uniform vec3 light;
uniform float shine;
uniform float specular;
uniform float hilight;
uniform float hilight_mix;
uniform float holo;        // 0 = painted metal, 1 = projected light
uniform vec3 tint;         // the hologram's body colour
uniform vec3 edge;         // and its rim, which runs hotter
uniform float floor_a;     // how solid a face-on surface is
uniform float scan;        // scanline pitch, in pixels
void main() {
    vec3 n = normalize(v_nrm);
    // Two-sided. The armour is built from overlapping closed plates, and where
    // one plate cuts into another the far side of the cut is what shows.
    if (n.z > 0.0) n = -n;

    if (holo < 0.5) {
        float wrapped = 0.5 + 0.5 * dot(n, light);
        float lit = pow(max(wrapped, 0.0), 1.5) * 0.74 + 0.20;
        vec3 half_way = normalize(light + vec3(0.0, 0.0, -1.0));
        lit += pow(max(dot(n, half_way), 0.0), shine) * specular;
        lit = clamp(lit, 0.0, 1.0);
        vec3 body = v_col * min(lit / hilight, 1.0);
        float hot = max(0.0, (lit - hilight) / (1.0 - hilight));
        f_col = vec4(body + (vec3(1.0) - body) * hot * hilight_mix, 1.0);
        return;
    }

    // A hologram is bright where you are looking along its surface and faint
    // where you are looking straight at it -- you see the edges of a projected
    // solid, not its faces. That is the whole effect, and it comes out of one
    // number: how far this surface has turned away from the camera.
    float facing = abs(n.z);
    float rim = pow(1.0 - facing, 2.6);

    // The model's paint becomes brightness rather than colour, so the plates
    // still read as separate parts without the armour's red and gold fighting
    // the projection's own colour.
    float lum = dot(v_col, vec3(0.114, 0.587, 0.299));
    vec3 body = tint * (0.45 + 0.85 * lum);

    // A weak specular keeps the plates from looking like flat cutouts; it is
    // the one cue that survives from the solid version.
    vec3 half_way = normalize(light + vec3(0.0, 0.0, -1.0));
    float gloss = pow(max(dot(n, half_way), 0.0), shine) * specular;

    // Scanlines, by screen row. Fine and shallow: a strong ripple reads as a
    // broken display rather than as a projection.
    float lines = 0.88 + 0.12 * sin(gl_FragCoord.y * 6.2831853 / max(scan, 1.0));

    vec3 colour = (body * (0.62 + 0.85 * rim) + edge * rim * rim * 1.15
                   + gloss) * lines;
    float alpha = clamp(floor_a + (1.0 - floor_a) * rim, 0.0, 1.0) * lines;
    f_col = vec4(colour, alpha);
}
"""


class Renderer:
    """A GL context and one program, reused for the life of the app."""

    def __init__(self):
        self.reason: str | None = None
        self.ctx = None
        self._size = (0, 0)
        self._msaa = self._flat = None
        self._vbo = None
        self._vao: dict = {}
        if moderngl is None:
            self.reason = "moderngl is not installed"
            return
        try:
            self.ctx = moderngl.create_standalone_context()
            self.program = self.ctx.program(vertex_shader=_VERTEX,
                                            fragment_shader=_FRAGMENT)
            self.program["light"].value = tuple(LIGHT)
            self.program["shine"].value = SHINE
            self.program["specular"].value = SPECULAR
            self.program["hilight"].value = HILIGHT
            self.program["hilight_mix"].value = HILIGHT_MIX
            self.program["tint"].value = tuple(c / 255.0 for c in HOLO_TINT)
            self.program["edge"].value = tuple(c / 255.0 for c in HOLO_EDGE)
            self.program["floor_a"].value = HOLO_FLOOR
            self.program["scan"].value = HOLO_SCAN
            self.program["holo"].value = 1.0
            self.ctx.enable(moderngl.DEPTH_TEST)
        except Exception as exc:        # noqa: BLE001 - no GPU is a valid state
            self.reason = f"{type(exc).__name__}: {exc}"
            self.ctx = None

    @property
    def available(self) -> bool:
        return self.ctx is not None

    def _target(self, width: int, height: int):
        """Framebuffers at least this big, grown in steps and then kept.

        Allocating exactly the hand's rectangle would reallocate on almost
        every frame, since the hand is never quite the same size twice.
        """
        # Rounded up to a multiple of 64 rather than to a power of two: a
        # 560-wide hand would otherwise take a 1024-wide buffer and read back
        # nearly twice the pixels it needs.
        want = (max(64, -(-width // 64) * 64), max(64, -(-height // 64) * 64))
        if want != self._size:
            for old in (self._msaa, self._flat):
                if old is not None:
                    old.release()
            self._msaa = self.ctx.framebuffer(
                self.ctx.renderbuffer(want, 4, samples=SAMPLES),
                self.ctx.depth_renderbuffer(want, samples=SAMPLES))
            self._flat = self.ctx.simple_framebuffer(want, components=4)
            self._size = want
        return self._msaa, self._flat

    def _array(self, posed, verts, faces):
        """The vertex array, built once and rewritten each frame.

        Allocating a vertex buffer, an index buffer and a vertex array per
        frame and releasing them again cost about 6ms of a 10ms draw -- most of
        the time was spent telling the driver about geometry rather than
        drawing it. The vertices change every frame and are written in place;
        the indices never change at all, so they are kept per chirality, which
        is the only thing that reorders them.
        """
        data = np.empty((len(verts), 9), np.float32)
        data[:, 0:3] = verts
        data[:, 3:6] = posed["normals"]
        data[:, 6:9] = posed["vcolour"] / 255.0
        raw = data.tobytes()
        if self._vbo is None or self._vbo.size < len(raw):
            if self._vbo is not None:
                self._vbo.release()
                for index, array in self._vao.values():
                    array.release()
                    index.release()
                self._vao.clear()
            self._vbo = self.ctx.buffer(raw, dynamic=True)
        else:
            self._vbo.write(raw)

        key = bool(posed.get("mirrored", False))
        if key not in self._vao:
            index = self.ctx.buffer(faces.astype(np.int32).tobytes())
            self._vao[key] = (index, self.ctx.vertex_array(
                self.program,
                [(self._vbo, "3f 3f 3f", "in_pos", "in_nrm", "in_col")],
                index_buffer=index, index_element_size=4))
        return self._vao[key][1]

    def set_style(self, holo: bool) -> None:
        """Projected light, or painted metal."""
        if self.available:
            self.program["holo"].value = 1.0 if holo else 0.0
            self.holo = bool(holo)

    def draw(self, frame: np.ndarray, posed: dict,
             solid: np.ndarray | None = None) -> bool:
        """Composite the posed glove onto `frame`. Returns False if it could not."""
        if self.ctx is None or posed is None:
            return False
        verts = posed["verts"]
        faces = posed["faces"]
        if not len(faces) or not np.isfinite(verts).all():
            return False

        height, width = frame.shape[:2]
        low = verts[:, :2].min(axis=0)
        high = verts[:, :2].max(axis=0)
        pad = (high - low) * PAD + 2.0
        # Whole pixels from here on. The projection, the framebuffer and the
        # crop all have to describe the same rectangle, and rounding each from
        # the float box separately does not achieve that: ceil(hi) - floor(lo)
        # can exceed ceil(hi - lo) by one, which sized the framebuffer a pixel
        # narrower than the crop that read it and failed every frame.
        ix0, iy0 = int(np.floor(low[0] - pad[0])), int(np.floor(low[1] - pad[1]))
        ix1, iy1 = int(np.ceil(high[0] + pad[0])), int(np.ceil(high[1] + pad[1]))
        want_w, want_h = ix1 - ix0, iy1 - iy0
        # The projection covers the whole rectangle even where it runs off the
        # frame; clipping it instead would shear the glove at the screen edge.
        region = (float(ix0), float(iy0), float(want_w), float(want_h))

        x0, y0 = max(ix0, 0), max(iy0, 0)
        x1, y1 = min(ix1, width), min(iy1, height)
        if x1 - x0 < 4 or y1 - y0 < 4 or want_w < 4 or want_h < 4:
            return False

        near = float(verts[:, 2].min()) - 1.0
        far = float(verts[:, 2].max()) + 1.0
        msaa, flat = self._target(want_w, want_h)

        array = self._array(posed, verts, faces)
        self.program["bounds"].value = region
        self.program["depth"].value = (near, far)
        msaa.use()
        self.ctx.viewport = (0, 0, want_w, want_h)
        msaa.clear(0.0, 0.0, 0.0, 0.0, viewport=(0, 0, want_w, want_h))
        array.render()
        self.ctx.copy_framebuffer(flat, msaa)
        raw = flat.read(viewport=(0, 0, want_w, want_h), components=4,
                        alignment=1)

        # GL hands back its rows bottom to top.
        shot = np.frombuffer(raw, np.uint8).reshape(want_h, want_w, 4)[::-1]
        # The rectangle the projection covered may hang off the frame; take the
        # part of it that landed on screen.
        ox, oy = x0 - ix0, y0 - iy0
        patch = shot[oy:oy + (y1 - y0), ox:ox + (x1 - x0)]
        if patch.shape[0] < y1 - y0 or patch.shape[1] < x1 - x0:
            return False
        # Alpha is coverage, and with multisampling it is partial at the
        # silhouette, so the edge is blended rather than stamped.
        #
        # Every step is an OpenCV call on purpose. Written as numpy -- promote
        # to uint16, multiply, add, divide by 255, cast back -- this one blend
        # cost 8ms of a 9.5ms draw, against 1.5ms for all of the actual
        # rendering. Four full-size temporaries and an integer division are
        # more work than the GPU was doing.
        target = frame[y0:y1, x0:x1]
        if getattr(self, "holo", True) and HOLO_SPLIT:
            # Pull the red and blue channels a pixel apart. Real projected
            # light does this at its edges, and it is the cheapest thing that
            # separates "a hologram of a glove" from "a glove drawn in cyan":
            # one shifted channel on an edge is a colour fringe, and a colour
            # fringe is what the eye reads as light that has been through
            # something.
            patch = np.ascontiguousarray(patch)
            blue, green, red, coverage = cv2.split(patch)
            offset = HOLO_SPLIT
            blue = np.roll(blue, -offset, axis=1)
            red = np.roll(red, offset, axis=1)
            patch = cv2.merge((blue, green, red, coverage))
        coverage = np.ascontiguousarray(patch[:, :, 3])
        if solid is not None:
            # See HOLO_COVER: tell the overlay this region is the app's, so it
            # is not faded a second time for being subtle.
            claim = solid[y0:y1, x0:x1]
            cv2.max(claim, cv2.LUT(coverage, _COVER_LUT), dst=claim)
        alpha = coverage.astype(np.float32)
        alpha *= 1.0 / 255.0
        if getattr(self, "holo", True):
            # Light adds. Scaling the hole it punches in the frame rather than
            # replacing what is there keeps the hand visible through the
            # projection -- see HOLO_OCCLUDE.
            alpha_hole = alpha * HOLO_OCCLUDE
            hole = cv2.merge([alpha_hole, alpha_hole, alpha_hole])
            weight = cv2.merge([alpha, alpha, alpha])
            fore = cv2.multiply(np.ascontiguousarray(patch[:, :, :3]), weight,
                                dtype=cv2.CV_8U)
            back = cv2.multiply(target, 1.0 - hole, dtype=cv2.CV_8U)
            cv2.add(back, fore, dst=target)
            return True
        weight = cv2.merge([alpha, alpha, alpha])
        fore = cv2.multiply(np.ascontiguousarray(patch[:, :, :3]), weight,
                            dtype=cv2.CV_8U)
        back = cv2.multiply(target, 1.0 - weight, dtype=cv2.CV_8U)
        cv2.add(fore, back, dst=target)
        return True


# Depth slabs the fallback cuts the glove into before batching fills by
# colour, how finely it quantises the light, and how much each triangle is
# inflated to hide the seam with its neighbour.
#
# Ten slabs rather than the twenty-eight this started at. Slabs multiply the
# number of fill batches -- twenty-eight of them over a hundred colour keys
# gave 1,149 batches for 3,852 triangles, which is a batched fill that batches
# nothing -- and down to five the silhouette is pixel-for-pixel identical.
# Timing does not choose between them: measured across the whole range the
# draw varies more between repeats of one setting (8.3 to 21.2ms) than between
# settings, so this is picked on there being less work to do rather than on a
# stopwatch that cannot see it.
FLAT_SLABS, FLAT_LEVELS, FLAT_GROW = 10, 20, 1.06


def _shade(posed: dict):
    """Per-triangle colour, and a key that groups triangles sharing one.

    The key is the material and the light level, not the colour. Quantising
    the three channels separately instead gives up to a level per channel
    cubed, and on a five-material glove that came to 1,395 groups for 3,852
    triangles -- a batched fill that barely batched. Two materials never share
    a colour, so material times level says the same thing in a hundred groups.
    """
    faces = posed["faces"]
    normals = posed["normals"][faces].mean(axis=1)
    normals = normals / np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    # Two-sided, as the shader is: where one plate cuts into another, the far
    # side of the cut is what shows.
    normals = normals * np.where(normals[:, 2:3] > 0.0, -1.0, 1.0)
    lit = np.clip(0.5 + 0.5 * (normals @ LIGHT), 0.0, None) ** 1.5 * 0.74 + 0.20
    half = LIGHT + np.array([0.0, 0.0, -1.0])
    half = half / np.linalg.norm(half)
    lit = np.clip(lit + np.clip(normals @ half, 0.0, 1.0) ** SHINE * SPECULAR,
                  0.0, 1.0)

    level = np.clip((lit * FLAT_LEVELS).astype(np.int64), 0, FLAT_LEVELS - 1)
    # Shade from the quantised level, so every triangle in a group really is
    # the colour the group is filled with.
    step = ((level + 0.5) / FLAT_LEVELS)[:, None]
    base = posed["vcolour"][faces[:, 0]]
    body = base * np.minimum(step / HILIGHT, 1.0)
    hot = np.maximum(0.0, (step - HILIGHT) / (1.0 - HILIGHT))
    colour = np.clip(body + (255.0 - body) * hot * HILIGHT_MIX, 0, 255)
    return colour, posed["material"].astype(np.int64) * FLAT_LEVELS + level


def draw_flat(frame: np.ndarray, posed: dict) -> bool:
    """Draw the glove without a GPU: sorted triangles, flat fills.

    A painter's algorithm, and it has the ordering flaw that motivated the GPU
    path -- but sorting per triangle rather than per part makes that flaw small
    instead of structural, because two triangles rarely straddle each other the
    way two whole plates do.

    Fills are batched. OpenCV has no call that takes a colour per polygon, but
    it has one taking many polygons in a single colour, so the triangles are
    ordered by depth, cut into slabs, and grouped inside a slab by quantised
    colour. Order within a slab is arbitrary and a slab is thin, so little that
    overlaps shares one. Triangles are inflated slightly as well: without a
    depth buffer the seams between neighbours show as hairlines.
    """
    verts, faces = posed["verts"], posed["faces"]
    if not len(faces) or not np.isfinite(verts).all():
        return False
    tri = verts[faces]
    xy = tri[:, :, :2]
    height, width = frame.shape[:2]
    if (xy[:, :, 0].max() < 0 or xy[:, :, 1].max() < 0
            or xy[:, :, 0].min() > width or xy[:, :, 1].min() > height):
        return False

    colour, key = _shade(posed)
    depth = tri[:, :, 2].mean(axis=1)
    slab = np.empty(len(depth), np.int64)
    slab[np.argsort(-depth)] = np.arange(len(depth)) * FLAT_SLABS // len(depth)
    code = slab * (key.max() + 1) + key

    order = np.argsort(code, kind="stable")
    code = code[order]
    middle = xy.mean(axis=1, keepdims=True)
    grown = middle + (xy - middle) * FLAT_GROW
    shapes = np.ascontiguousarray(grown[order].astype(np.int32))
    paint = colour[order]

    cuts = np.flatnonzero(np.diff(code)) + 1
    for lo, hi in zip(np.concatenate([[0], cuts]),
                      np.concatenate([cuts, [len(order)]])):
        # Aliased on purpose. Anti-aliasing 1,400 fills costs 2.7ms of a 10.5ms
        # draw, and this is the path taken only when there is no GPU to do it
        # properly -- spending a quarter of the budget softening the edges of a
        # render that is already faceted is the wrong trade.
        cv2.fillPoly(frame, shapes[lo:hi], tuple(paint[lo]))
    return True

