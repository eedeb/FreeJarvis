"""Turning a hand pose into a screen coordinate.

The mapping is learned during calibration and has two stages.

1.  A **homography** from the index fingertip position in the camera image to a
    pixel on the screen.  A homography (8 degrees of freedom) is the exact
    projective relationship between two planes, so it absorbs the camera
    position, roll, and off-axis mounting for free -- which is why four dots are
    already enough to get something usable.

2.  A **ridge-regularised residual correction** on top of it.  A pure
    homography assumes the fingertip moves in a flat plane, which it does not:
    lens distortion bends the edges, where the finger *aims* matters as well as
    where it *is*, and leaning towards or away from the camera shifts
    everything.  So the second stage regresses whatever error the homography
    leaves over features describing exactly those effects -- position, pointing
    direction, and a depth term taken from the apparent size of the palm.

Stage 2 has enough parameters to overfit a dozen calibration dots, so the
feature set *and* the ridge strength are both chosen by leave-one-out
cross-validation over the whole pipeline (the homography is refit inside every
fold).  If no candidate beats the plain homography on held-out dots, the
correction is dropped entirely.  The LOOCV error doubles as the accuracy
reported back to the user, so that number is an honest out-of-sample estimate
rather than a fit residual.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .landmarks import INDEX_TIP, hand_scale, point_direction

# Candidate residual feature sets, smallest first.  Names index into the block
# of columns built by `_columns`.
_COLUMN_NAMES = ("u", "v", "dx", "dy", "e", "uu", "uv", "vv", "eu", "ev")
_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "linear": ("u", "v", "dx", "dy", "e"),
    "full": _COLUMN_NAMES,
}
_LAMBDAS = np.logspace(-3.0, 3.0, 13)


@dataclass(frozen=True)
class AimSample:
    """Everything the mapping needs from one video frame."""

    tip: tuple[float, float]        # aim point, normalised image coords
    direction: tuple[float, float]  # facing, projected into the image plane
    scale: float                    # palm length in metres (world landmarks)
    # The two below are what the ray model needs and the flat models ignore:
    # the full 3D facing, and how big the palm looks in the image. Together
    # with `scale` they are enough to place the hand in space.
    normal: tuple[float, float, float] = (0.0, 0.0, -1.0)
    apparent: float = 0.0           # palm length as seen, in image widths

    @classmethod
    def from_landmarks(cls, norm: np.ndarray, metric: np.ndarray) -> "AimSample":
        d = point_direction(metric)
        return cls(
            tip=(float(norm[INDEX_TIP, 0]), float(norm[INDEX_TIP, 1])),
            direction=(float(d[0]), float(d[1])),
            scale=hand_scale(metric),
        )

    @classmethod
    def average(cls, samples: Sequence["AimSample"]) -> "AimSample":
        """Robust per-dot average: the median beats the mean when tracking blips."""
        tips, dirs, scales = _stack(samples)
        d = np.median(dirs, axis=0)
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            d = d / n
        med = np.median(tips, axis=0)
        normals = np.median(np.array([s.normal for s in samples]), axis=0)
        length = float(np.linalg.norm(normals))
        if length > 1e-9:
            normals = normals / length
        return cls(tip=(float(med[0]), float(med[1])),
                   direction=(float(d[0]), float(d[1])),
                   scale=float(np.median(scales)),
                   normal=(float(normals[0]), float(normals[1]), float(normals[2])),
                   apparent=float(np.median([s.apparent for s in samples])))


def _stack(samples: Sequence[AimSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tips = np.array([s.tip for s in samples], dtype=float)
    dirs = np.array([s.direction for s in samples], dtype=float)
    scales = np.array([s.scale for s in samples], dtype=float)
    return tips, dirs, scales


# --------------------------------------------------------------------------
# Stage 1: homography
# --------------------------------------------------------------------------

def _conditioning_transform(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: centre the points, scale them to mean |p| = sqrt(2).

    Without it the DLT matrix mixes terms of wildly different magnitude (pixels
    squared against pixels against 1) and the SVD loses precision.
    """
    centre = pts.mean(axis=0)
    spread = float(np.linalg.norm(pts - centre, axis=1).mean())
    s = np.sqrt(2.0) / spread if spread > 1e-12 else 1.0
    return np.array([[s, 0.0, -s * centre[0]],
                     [0.0, s, -s * centre[1]],
                     [0.0, 0.0, 1.0]])


def _apply(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homo = np.hstack([pts, np.ones((len(pts), 1))]) @ mat.T
    w = homo[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return homo[:, :2] / w


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares homography src -> dst via the normalised DLT."""
    if len(src) < 4:
        raise ValueError("a homography needs at least 4 point pairs")

    t_src = _conditioning_transform(src)
    t_dst = _conditioning_transform(dst)
    s = _apply(t_src, src)
    d = _apply(t_dst, dst)

    rows = []
    for (x, y), (xp, yp) in zip(s, d):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    h_norm = vt[-1].reshape(3, 3)

    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


# --------------------------------------------------------------------------
# Stage 2: residual correction
# --------------------------------------------------------------------------

def _columns(base_px: np.ndarray, dirs: np.ndarray, scales: np.ndarray,
             screen: tuple[int, int], scale0: float) -> dict[str, np.ndarray]:
    u = base_px[:, 0] / screen[0] * 2.0 - 1.0   # homography output mapped to [-1, 1]
    v = base_px[:, 1] / screen[1] * 2.0 - 1.0
    # Depth: log-ratio of palm size against the calibration distance, so it is 0
    # where the model was trained and symmetric for leaning in vs. out.
    e = np.log(np.maximum(scales, 1e-6) / max(scale0, 1e-6))
    return {
        "u": u, "v": v,
        "dx": dirs[:, 0], "dy": dirs[:, 1],
        "e": e,
        "uu": u * u, "uv": u * v, "vv": v * v,
        "eu": e * u, "ev": e * v,
    }


def _design(cols: dict[str, np.ndarray], names: Sequence[str],
            mean: np.ndarray | None, std: np.ndarray | None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardised design matrix with a leading bias column."""
    rows = len(cols["u"])
    x = (np.stack([cols[n] for n in names], axis=1) if names
         else np.zeros((rows, 0)))
    if mean is None or std is None:
        mean = x.mean(axis=0) if x.size else np.zeros(x.shape[1])
        std = x.std(axis=0) if x.size else np.zeros(x.shape[1])
        std = np.where(std < 1e-8, 1.0, std)
    xs = (x - mean) / std if x.size else x
    phi = np.hstack([np.ones((rows, 1)), xs])
    return phi, mean, std


def _ridge(phi: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge solution.  The bias column is left unpenalised."""
    d = phi.shape[1]
    reg = lam * np.eye(d)
    reg[0, 0] = 0.0
    gram = phi.T @ phi + reg
    try:
        return np.linalg.solve(gram, phi.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, phi.T @ y, rcond=None)[0]


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

class DirectMap:
    """The camera frame *is* the screen. No calibration, nothing fitted.

    Imagine the webcam's picture blown up to fill the monitor, then made
    invisible: your hand is somewhere on that picture, and the cursor goes to
    the same place. A hand a third of the way across the frame puts the cursor
    a third of the way across the screen.

    Its real advantage is that it does not magnify anything. Every fitted
    mapping stretches some smaller region of the camera's view over the whole
    screen, and stretches the tracking noise with it -- a hand that swept a
    quarter of the frame multiplied every wobble by four. Here the scale factor
    is the screen width over the camera width, which is 1 for a 1080p camera on
    a 1080p monitor. One pixel of jitter stays one pixel.

    The price is that the whole screen costs a whole armspan, and the corners
    of the frame are where hand tracking is least reliable.
    """

    def __init__(self, screen: tuple[int, int], mirror: bool = True,
                 margin: float = 0.0):
        self.screen = (int(screen[0]), int(screen[1]))
        self.mirror = mirror
        self.margin = float(np.clip(margin, 0.0, 0.4))
        self.src = "direct"
        self.stale = False
        self.meta = {"aims_by": "where the hand is in the camera view",
                     "mirror": mirror, "margin": self.margin}

    def _map(self, u: float, v: float) -> tuple[float, float]:
        # Mirrored, because the camera faces you: without this, moving your
        # hand right would send the cursor left, as writing on the far side of
        # a window looks backwards from where you stand.
        if self.mirror:
            u = 1.0 - u
        span = 1.0 - 2.0 * self.margin
        if span > 1e-6:
            u = (u - self.margin) / span
            v = (v - self.margin) / span
        x = min(max(u, 0.0), 1.0) * (self.screen[0] - 1)
        y = min(max(v, 0.0), 1.0) * (self.screen[1] - 1)
        return x, y

    def predict(self, sample: AimSample) -> tuple[float, float]:
        return self._map(sample.tip[0], sample.tip[1])

    def predict_many(self, samples: Sequence[AimSample]) -> np.ndarray:
        return np.array([self.predict(s) for s in samples], dtype=float)


class PointerMap:
    """Maps an :class:`AimSample` to a pixel on the screen."""

    def __init__(self, homography: np.ndarray, screen: tuple[int, int],
                 feature_set: str = "none", weights: np.ndarray | None = None,
                 mean: np.ndarray | None = None, std: np.ndarray | None = None,
                 scale0: float = 1.0, meta: dict | None = None,
                 src: str = "tip"):
        # A 3x3 for a homography, an (8, 2) for the combined map, or the
        # (focal, depth, affine) triple for the ray model.
        self.homography = (homography if isinstance(homography, tuple)
                           else np.asarray(homography, dtype=float))
        self.screen = (int(screen[0]), int(screen[1]))
        self.feature_set = feature_set
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.std = None if std is None else np.asarray(std, dtype=float)
        self.scale0 = float(scale0)
        self.src = src   # "tip": aims by where the hand is; "dir": by its facing
        self.meta = meta or {}
        self.stale = False   # set by from_dict for calibrations of an older format

    # -- prediction --------------------------------------------------------

    def predict_many(self, samples: Sequence[AimSample]) -> np.ndarray:
        _, dirs, scales = _stack(samples)
        base = _stage1_apply(self.homography, samples, self.src)
        names = _FEATURE_SETS[self.feature_set]
        if not names or self.weights is None:
            return base
        cols = _columns(base, dirs, scales, self.screen, self.scale0)
        phi, _, _ = _design(cols, names, self.mean, self.std)
        return base + phi @ self.weights

    def predict(self, sample: AimSample) -> tuple[float, float]:
        p = self.predict_many([sample])[0]
        return float(p[0]), float(p[1])

    # -- persistence -------------------------------------------------------

    # 2 marks calibrations recorded after hand pose moved from the flat image
    # to 3D world landmarks.  Version 1 files were captured with the old pinch
    # test, which could fire while the hand was nowhere near the dot, so their
    # samples cannot be trusted however good the fit looks.
    VERSION = 2

    def to_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "screen": list(self.screen),
            "homography": (
                [float(self.homography[0]), float(self.homography[1]),
                 np.asarray(self.homography[2]).tolist()]
                if self.src == "ray" else np.asarray(self.homography).tolist()),
            "feature_set": self.feature_set,
            "weights": None if self.weights is None else self.weights.tolist(),
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "scale0": self.scale0,
            "src": self.src,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PointerMap":
        model = cls._from_dict(d)
        model.stale = int(d.get("version", 1)) < cls.VERSION
        return model

    @classmethod
    def _from_dict(cls, d: dict) -> "PointerMap":
        return cls(
            homography=(
                (float(d["homography"][0]), float(d["homography"][1]),
                 np.asarray(d["homography"][2], dtype=float))
                if d.get("src") == "ray"
                else np.asarray(d["homography"], dtype=float)),
            screen=(d["screen"][0], d["screen"][1]),
            feature_set=d.get("feature_set", "none"),
            weights=None if d.get("weights") is None else np.asarray(d["weights"]),
            mean=None if d.get("mean") is None else np.asarray(d["mean"]),
            std=None if d.get("std") is None else np.asarray(d["std"]),
            scale0=d.get("scale0", 1.0),
            src=d.get("src", "tip"),
            meta=d.get("meta", {}),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PointerMap":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# The ray model: a stick out of the palm, and where it lands on the screen
# --------------------------------------------------------------------------

def _hand_in_space(samples: Sequence[AimSample], focal: float) -> np.ndarray:
    """Palm positions in camera coordinates, in metres.

    The world landmarks give the palm's true size; the image gives how big it
    looks.  The ratio is depth.  Sideways position then follows from where in
    the frame it sits, and -- pleasingly -- the focal length cancels out of X
    and Y entirely, so it only has to be guessed for Z.
    """
    out = np.empty((len(samples), 3))
    for i, s in enumerate(samples):
        seen = max(s.apparent, 1e-6)
        ratio = s.scale / seen                    # metres per image unit
        out[i] = ((s.tip[0] - 0.5) * ratio,
                  (s.tip[1] - 0.5) * ratio,
                  focal * ratio)
    return out


def _ray_hits(samples: Sequence[AimSample], focal: float, plane_z: float
              ) -> np.ndarray:
    """Where each palm's outward ray crosses the plane at depth `plane_z`.

    The screen sits roughly parallel to the camera's own image plane -- a
    webcam clipped to a monitor is aimed the same way the monitor faces -- so
    the plane is taken as flat in Z.  Whatever that approximation leaves over
    is picked up by the affine fitted on top, and by the residual stage.
    """
    origins = _hand_in_space(samples, focal)
    normals = np.array([s.normal for s in samples], dtype=float)
    nz = normals[:, 2]
    # A ray running parallel to the screen never meets it. Clamp rather than
    # divide by zero: the fit will simply score such a configuration badly.
    nz = np.where(np.abs(nz) < 0.15, np.sign(nz) * 0.15 + 1e-9, nz)
    travel = (plane_z - origins[:, 2]) / nz
    return origins[:, :2] + normals[:, :2] * travel[:, None]


def _fit_affine(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Least-squares 2D affine from plane coordinates to screen pixels."""
    phi = np.hstack([source, np.ones((len(source), 1))])
    return np.linalg.lstsq(phi, targets, rcond=None)[0]


def _apply_affine(affine: np.ndarray, source: np.ndarray) -> np.ndarray:
    return np.hstack([source, np.ones((len(source), 1))]) @ affine


def fit_ray_model(samples: Sequence[AimSample], targets: np.ndarray
                  ) -> tuple[float, float, np.ndarray]:
    """Fit focal length, screen depth, and the plane-to-pixel affine.

    Only two of those are awkward: given a focal length and a screen depth the
    affine follows from linear least squares.  So the search is over two
    numbers with an exact solve inside it, which is far steadier than throwing
    all eight at a general optimiser.
    """
    best = (float("inf"), 1.0, 0.0, np.zeros((3, 2)))
    focals = np.linspace(0.6, 2.4, 19)
    depths = np.linspace(-0.35, 0.25, 25)
    for _ in range(3):        # coarse sweep, then twice around the winner
        for focal in focals:
            for depth in depths:
                hits = _ray_hits(samples, focal, depth)
                if not np.all(np.isfinite(hits)):
                    continue
                affine = _fit_affine(hits, targets)
                err = float(np.sqrt(
                    ((_apply_affine(affine, hits) - targets) ** 2).sum(1).mean()))
                if err < best[0]:
                    best = (err, focal, depth, affine)
        _, focal, depth, _ = best
        fs, ds = (focals[1] - focals[0]), (depths[1] - depths[0])
        focals = np.linspace(max(focal - fs, 0.3), focal + fs, 9)
        depths = np.linspace(depth - ds, depth + ds, 9)
    return best[1], best[2], best[3]


def _combined_design(tips: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Position and facing together, with their cross terms."""
    px, py = tips[:, 0], tips[:, 1]
    nx, ny = dirs[:, 0], dirs[:, 1]
    ones = np.ones(len(tips))
    return np.stack([ones, px, py, nx, ny, px * nx, py * ny, px * py], axis=1)


def _fit_combined(tips, dirs, targets, lam: float = 1e-4) -> np.ndarray:
    """Least-squares map from position *and* facing straight to the screen.

    A homography relates two planes, so it can only be given one of the two.
    That is fine for someone who aims purely by moving their hand, or purely by
    turning it, but not for the common case of doing some of each: then neither
    measurement alone determines the target and the homography cannot fit
    either way round.  This is the model for that case.
    """
    phi = _combined_design(tips, dirs)
    reg = lam * np.eye(phi.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(phi.T @ phi + reg, phi.T @ targets)


def _source(tips: np.ndarray, dirs: np.ndarray, src: str) -> np.ndarray:
    """Which measurement the first stage maps from.

    There are two quite different ways a person aims a hand at a screen, and
    they need different mappings.  Some move the hand about, so *where it is*
    carries the signal.  Others hold it roughly still and turn it, so *which
    way it faces* does.  Rather than assume, both are fitted and the one that
    predicts held-out dots better wins.
    """
    return dirs if src == "dir" else tips


def _stage1_fit(samples: Sequence[AimSample], targets, src):
    """Fit whichever first-stage model `src` names."""
    tips, dirs, _ = _stack(samples)
    if src == "ray":
        return fit_ray_model(samples, targets)
    if src == "both":
        return _fit_combined(tips, dirs, targets)
    return fit_homography(_source(tips, dirs, src), targets)


def _stage1_apply(stage1, samples: Sequence[AimSample], src):
    tips, dirs, _ = _stack(samples)
    if src == "ray":
        focal, plane_z, affine = stage1
        return _apply_affine(affine, _ray_hits(samples, focal, plane_z))
    if src == "both":
        return _combined_design(tips, dirs) @ stage1
    return _apply(stage1, _source(tips, dirs, src))


def _pipeline_fit(samples, targets, screen, names, lam, src="tip"):
    """Fit both stages on one training split."""
    _, dirs, scales = _stack(samples)
    h = _stage1_fit(samples, targets, src)
    scale0 = float(np.median(scales))
    if not names:
        return h, scale0, None, None, None

    base = _stage1_apply(h, samples, src)
    cols = _columns(base, dirs, scales, screen, scale0)
    phi, mean, std = _design(cols, names, None, None)
    w = _ridge(phi, targets - base, lam)
    return h, scale0, w, mean, std


def _pipeline_predict(model, samples, screen, names, src="tip"):
    h, scale0, w, mean, std = model
    base = _stage1_apply(h, samples, src)
    if not names or w is None:
        return base
    _, dirs, scales = _stack(samples)
    cols = _columns(base, dirs, scales, screen, scale0)
    phi, _, _ = _design(cols, names, mean, std)
    return base + phi @ w


def _loocv_error(samples, targets, screen, names, lam, src="tip") -> float:
    """RMS distance in pixels on dots the model was not trained on."""
    n = len(samples)
    errs = np.empty(n)
    for i in range(n):
        train = [s for j, s in enumerate(samples) if j != i]
        keep = np.arange(n) != i
        try:
            model = _pipeline_fit(train, targets[keep], screen, names, lam, src)
            pred = _pipeline_predict(model, [samples[i]], screen, names, src)
        except (ValueError, np.linalg.LinAlgError):
            return float("inf")
        errs[i] = np.linalg.norm(pred[0] - targets[i])
    if not np.all(np.isfinite(errs)):
        return float("inf")
    return float(np.sqrt(np.mean(errs ** 2)))


def describe_inputs(samples: Sequence[AimSample]) -> dict:
    """How much signal the calibration dots actually contain.

    A mapping can only be as good as the variation it was shown.  If the hand
    neither moved nor turned much between dots, no fit will work, and knowing
    which of the two was missing is the difference between "move your hand
    further" and "turn it further".
    """
    tips, dirs, scales = _stack(samples)
    span = tips.max(axis=0) - tips.min(axis=0)
    # Facing spread as an angle: the direction vectors are a projection of the
    # unit palm normal, so their spread in the plane is a sine of the turn.
    facing = float(np.linalg.norm(dirs.max(axis=0) - dirs.min(axis=0)))
    return {
        "position_span_x": float(span[0]),
        "position_span_y": float(span[1]),
        "facing_span": facing,
        "facing_span_deg": float(np.degrees(np.arcsin(min(facing, 1.0)))),
        "scale_ratio": float(scales.max() / max(scales.min(), 1e-9)),
    }


def fit_pointer_map(samples: Sequence[AimSample],
                    targets: Iterable[Sequence[float]],
                    screen: tuple[int, int]) -> PointerMap:
    """Fit the full mapping from calibration dots.

    `samples[i]` is the averaged hand pose recorded while the user pointed at
    screen pixel `targets[i]`.
    """
    tips, dirs, scales = _stack(samples)
    tgt = np.asarray(list(targets), dtype=float)
    n = len(tips)
    if n < 4:
        raise ValueError(f"need at least 4 calibration points, got {n}")
    if len(tgt) != n:
        raise ValueError("samples and targets must be the same length")

    # Candidates: plain homography, plus each residual feature set at each ridge
    # strength.  A feature set is only considered once there are enough dots to
    # leave one out and still leave the fit reasonably determined.
    combos: list[tuple[str, str, float]] = []
    has_ray = all(s.apparent > 0 for s in samples)
    for src in ("tip", "dir", "both", "ray"):
        # The combined map already has eight parameters; it needs more dots
        # before leaving one out still leaves it determined.
        if src == "both" and n < 10:
            continue
        # The ray model needs the 3D measurements, which older calibration
        # samples do not carry.
        if src == "ray" and (not has_ray or n < 6):
            continue
        combos.append((src, "none", 0.0))
        if n >= 8:
            combos += [(src, "linear", lam) for lam in _LAMBDAS]
        if n >= 12:
            combos += [(src, "full", lam) for lam in _LAMBDAS]

    scored = []
    for src, key, lam in combos:
        err = _loocv_error(samples, tgt, screen, _FEATURE_SETS[key], lam, src)
        scored.append((err, src, key, float(lam)))

    baseline = next(e for e, s, k, _ in scored if s == "tip" and k == "none")

    # Choosing among many models on a dozen cross-validated points is itself a
    # way to overfit: with enough candidates one of them wins by luck and then
    # generalises worse than the plain fit it displaced.  So each step up in
    # expressiveness has to clear a margin, and the bigger the step the wider
    # the margin.  Positional homography is the default; a directional one is
    # the same size and needs only to be clearly better; the combined map has
    # twice the parameters and has to earn them.
    def best_of(src: str) -> tuple[float, str, str, float]:
        family = [s for s in scored if s[1] == src]
        return min(family, key=lambda s: s[0]) if family else (
            float("inf"), src, "none", 0.0)

    chosen = best_of("tip")
    # The ray model is the most constrained of the lot -- it is real geometry
    # with eight parameters rather than a free-form fit -- so it only needs to
    # be clearly better, not dramatically so.
    for src, margin in (("dir", 0.95), ("ray", 0.95), ("both", 0.90)):
        candidate = best_of(src)
        if np.isfinite(candidate[0]) and candidate[0] < chosen[0] * margin:
            chosen = candidate

    best_err, best_src, best_key, best_lam = chosen
    # Within the chosen family, the residual stage must also earn its place.
    if not np.isfinite(best_err):
        best_err, best_src, best_key, best_lam = baseline, "tip", "none", 0.0
    elif best_key != "none":
        plain = next((e for e, s, k, _ in scored
                      if s == best_src and k == "none"), float("inf"))
        if np.isfinite(plain) and best_err > plain * 0.98:
            best_err, best_key, best_lam = plain, "none", 0.0

    names = _FEATURE_SETS[best_key]
    h, scale0, w, mean, std = _pipeline_fit(samples, tgt, screen, names,
                                            best_lam, best_src)
    model = PointerMap(h, screen, best_key, w, mean, std, scale0, src=best_src)

    fitted = model.predict_many(samples)
    residual = np.linalg.norm(fitted - tgt, axis=1)

    def _finite(x: float) -> float | None:
        # With only 4 dots, leaving one out leaves too few to fit -- report
        # "unknown" rather than writing Infinity into the calibration file.
        return float(x) if np.isfinite(x) else None

    model.meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "points": n,
        "feature_set": best_key,
        "aims_by": {"dir": "orientation", "both": "position and orientation",
                    "ray": "a ray out of the palm"}.get(best_src, "position"),
        "ridge_lambda": best_lam,
        "fit_rms_px": float(np.sqrt(np.mean(residual ** 2))),
        "fit_max_px": float(residual.max()),
        "loocv_rms_px": _finite(best_err),
        "homography_only_loocv_px": _finite(baseline),
    }
    return model
