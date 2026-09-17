"""Hand landmark tracking, and the per-frame hand state everything else reads."""

from __future__ import annotations

import contextlib
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import landmarks as lm
from .camera import Camera
from .config import MODEL_PATH, MODEL_URL, Settings
from .mapping import AimSample


@contextlib.contextmanager
def _quiet_native_stderr():
    """Mute writes to the real stderr file descriptor for a moment.

    Building the landmarker prints a couple of TensorFlow-Lite warnings from
    C++ before any Python logging config can apply, and they are noise every
    single run.  Only the raw fd is redirected, so Python exceptions raised in
    the block still surface normally.
    """
    try:
        saved = os.dup(2)
    except OSError:
        yield
        return
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


class ModelError(RuntimeError):
    """The hand tracking model is missing and could not be fetched."""


def ensure_model(path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Download the MediaPipe hand model on first run (about 7.8 MB)."""
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand tracking model to {path} ...", flush=True)
    tmp = path.with_suffix(".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, \
                open(tmp, "wb") as fh:
            fh.write(response.read())
        tmp.replace(path)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise ModelError(
            f"Could not download the hand model from {url} ({exc}). "
            f"Download it manually and save it as {path}."
        ) from exc
    print("Model ready.", flush=True)
    return path


@dataclass
class HandFrame:
    """One frame of hand state."""

    stamp: float                 # time.monotonic() when the frame was captured
    norm: np.ndarray             # (21, 2) landmarks in normalised image coords
    metric: np.ndarray           # (21, 2) with x scaled by the aspect ratio
    handedness: str
    confidence: float
    world: np.ndarray | None = None   # (21, 3) metres, relative to the hand

    @property
    def pose(self) -> np.ndarray:
        """The coordinates to answer shape questions in.

        World coordinates when MediaPipe supplies them, because they are immune
        to the foreshortening that pointing at the camera causes.  The 2D
        fallback exists so a stripped-down result can still drive the app.
        """
        return self.metric if self.world is None else self.world

    @property
    def sample(self) -> AimSample:
        return self.aim_sample("finger")

    def aim_sample(self, mode: str = "palm") -> AimSample:
        """What the pointing mapping is fed, for the chosen way of aiming."""
        if mode == "finger":
            return AimSample.from_landmarks(self.norm, self.metric)
        # The palm centre is the aim point; which way the palm faces is the
        # extra feature, the same role the finger direction plays in the other
        # mode -- it separates where the hand is from where it is turned.
        centre = lm.palm_center(self.norm)
        if self.world is not None:
            normal = lm.palm_normal(self.world)
        else:
            flat = lm.point_direction(self.metric)
            normal = np.array([flat[0], flat[1], -1.0])
            normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
        # How big the palm looks, against how big it really is: together these
        # give the hand's distance, which is what turns a direction into a ray
        # with an actual origin in space.
        apparent = float(np.linalg.norm(self.metric[lm.MIDDLE_MCP]
                                        - self.metric[lm.WRIST]))
        return AimSample(tip=(float(centre[0]), float(centre[1])),
                         direction=(float(normal[0]), float(normal[1])),
                         scale=self.scale,
                         normal=(float(normal[0]), float(normal[1]), float(normal[2])),
                         apparent=apparent)

    @property
    def curl(self) -> float:
        """1 for a flat open hand, negative for a closed fist."""
        return lm.curl(self.pose)

    @property
    def scale(self) -> float:
        return lm.hand_scale(self.pose)

    def pinch(self, tip: int = lm.INDEX_TIP) -> float:
        """Thumb-to-fingertip gap, as a fraction of palm length.

        Measured in the flat image, not in 3D, which is the opposite of every
        other pose question here and worth explaining.  Measuring a *pinch* in
        3D means trusting MediaPipe's depth estimate for two fingertips that
        are almost touching, and that estimate is noisy enough to inflate the
        gap: on a real hand the pinched and open states came out 0.19 apart in
        3D against 0.59 apart in the image.

        The image lies in one particular way: a thumb held clear of the finger
        but *behind* it projects onto the same spot, reading as a pinch that is
        not happening.  The obvious guard is to take the larger of the flat and
        3D readings, and that is wrong: for fingertips that really are touching
        the 3D distance is almost entirely depth error, so the guard ends up
        setting the threshold from MediaPipe's noise floor rather than from the
        hand.  It put the click 4.5cm wide.

        So the depth is consulted only when it claims a gap far larger than
        that noise -- a genuine thumb-behind-finger, not a jittery estimate of
        two fingers in contact.  Below that the flat image decides, and it is
        precise: a centimetre is about seventeen pixels of a 1080p frame.
        """
        flat = lm.pinch_ratio(self.metric, tip)
        if self.world is None:
            return flat
        scale = lm.hand_scale(self.world)
        if scale < 1e-9:
            return flat
        depth_gap = abs(float(self.world[lm.THUMB_TIP][2] - self.world[tip][2])) / scale
        if depth_gap > 0.45:      # about 4cm of depth: too far to be noise
            return max(flat, depth_gap)
        return flat

    def bones(self) -> np.ndarray | None:
        """Lengths of the pinch-relevant bones, for the plausibility check."""
        return None if self.world is None else lm.bone_lengths(self.world)

    def pinch_cm(self, tip: int = lm.INDEX_TIP) -> float | None:
        """The same gap in centimetres, or None without world landmarks.

        Thresholds are far easier to reason about in real units than in palm
        lengths, and the world landmarks give the palm's true size, so the
        conversion is available for free.
        """
        if self.world is None:
            return None
        palm_m = lm.hand_scale(self.world)
        if palm_m < 1e-6:
            return None
        return self.pinch(tip) * palm_m * 100.0

    def spread(self) -> float:
        return lm.fingers_spread(self.pose)

    def extended(self) -> dict[str, bool]:
        pose = self.pose
        return {
            "thumb": lm.thumb_extended(pose),
            "index": lm.finger_extended(pose, "index"),
            "middle": lm.finger_extended(pose, "middle"),
            "ring": lm.finger_extended(pose, "ring"),
            "pinky": lm.finger_extended(pose, "pinky"),
        }


class HandTracker:
    """Camera plus MediaPipe, running in a thread and publishing the latest hand.

    Inference costs a good few milliseconds; running it on the UI thread would
    make the calibration window stutter and the cursor loop uneven.  Callers
    just ask for :meth:`latest` whenever they are ready to draw or move.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._camera: Camera | None = None
        self._landmarker = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._hand: HandFrame | None = None
        self._preview: np.ndarray | None = None
        self._last_ts_ms = -1
        self._fps = 0.0
        self._frames = 0
        self.error: Exception | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "HandTracker":
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (HandLandmarker,
                                                   HandLandmarkerOptions,
                                                   RunningMode)

        model = ensure_model()
        s = self.settings
        with _quiet_native_stderr():
            self._landmarker = HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model)),
                    running_mode=RunningMode.VIDEO,
                    num_hands=max(1, s.max_hands),
                    min_hand_detection_confidence=s.detection_confidence,
                    min_hand_presence_confidence=s.presence_confidence,
                    min_tracking_confidence=s.tracking_confidence,
                )
            )
        self._camera = Camera(s.camera_index, s.camera_width,
                              s.camera_height, s.camera_fps).open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="tracker", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001 - closing must never mask the real error
                pass
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- worker ------------------------------------------------------------

    def _loop(self) -> None:
        import mediapipe as mp

        assert self._camera is not None and self._landmarker is not None
        last_seq = 0
        fps_mark = time.monotonic()
        fps_count = 0

        while not self._stop.is_set():
            frame, stamp, seq = self._camera.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # detect_for_video demands strictly increasing timestamps.
            ts_ms = max(int(stamp * 1000.0), self._last_ts_ms + 1)
            self._last_ts_ms = ts_ms
            try:
                result = self._landmarker.detect_for_video(image, ts_ms)
            except Exception as exc:  # noqa: BLE001 - surface on the main thread
                self.error = exc
                break

            hand = self._to_hand_frame(result, stamp, frame.shape[1], frame.shape[0])
            with self._lock:
                self._hand = hand
                self._preview = frame
                self._frames += 1

            fps_count += 1
            now = time.monotonic()
            if now - fps_mark >= 0.5:
                self._fps = fps_count / (now - fps_mark)
                fps_count, fps_mark = 0, now

    @staticmethod
    def _to_hand_frame(result, stamp: float, width: int, height: int
                       ) -> HandFrame | None:
        if not result.hand_landmarks:
            return None
        # With more than one hand in view, follow the biggest one: that is the
        # one nearest the camera, and therefore the one being pointed with.
        best_index, best_span = 0, -1.0
        for i, hand in enumerate(result.hand_landmarks):
            pts = np.array([[p.x, p.y] for p in hand], dtype=float)
            span = float(np.ptp(pts, axis=0).max())
            if span > best_span:
                best_index, best_span = i, span

        hand = result.hand_landmarks[best_index]
        norm = np.array([[p.x, p.y] for p in hand], dtype=float)
        aspect = width / height if height else 1.0
        metric = norm * np.array([aspect, 1.0])

        world = None
        if result.hand_world_landmarks and best_index < len(result.hand_world_landmarks):
            world = np.array([[p.x, p.y, p.z]
                              for p in result.hand_world_landmarks[best_index]],
                             dtype=float)

        label, score = "", 0.0
        if result.handedness and best_index < len(result.handedness):
            category = result.handedness[best_index][0]
            label, score = category.category_name, float(category.score)
        return HandFrame(stamp=stamp, norm=norm, metric=metric,
                         handedness=label, confidence=score, world=world)

    # -- readers -----------------------------------------------------------

    def latest(self) -> HandFrame | None:
        with self._lock:
            return self._hand

    def latest_preview(self) -> np.ndarray | None:
        with self._lock:
            return None if self._preview is None else self._preview.copy()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frames_processed(self) -> int:
        with self._lock:
            return self._frames

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise self.error

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """Block until at least one frame has been through MediaPipe."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.raise_if_failed()
            if self.frames_processed > 0:
                return True
            time.sleep(0.02)
        return False

