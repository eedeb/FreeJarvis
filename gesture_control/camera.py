"""Webcam capture on a background thread."""

from __future__ import annotations

import pathlib
import sys
import threading
import time

import cv2
import cv2.utils.logging
import numpy as np

# Probing camera indices logs a warning per backend per index; that is
# expected behaviour here, not something to show the user.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

# Windows exposes cameras through several backends and they are not equally
# cooperative; DirectShow opens fastest and most reliably, Media Foundation is
# the modern one, and CAP_ANY is a last resort.
_BACKENDS = (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY))


class CameraError(RuntimeError):
    pass


_CONSENT_KEY = (r"SOFTWARE\Microsoft\Windows\CurrentVersion"
                r"\CapabilityAccessManager\ConsentStore\webcam")


def permission_hint() -> str | None:
    """Explain the one camera failure that looks like a broken webcam.

    Python installed from the Microsoft Store is a *packaged* app, so Windows
    gives it its own camera permission entry instead of covering it under "let
    desktop apps access your camera".  When that entry is set to Deny, the
    device still enumerates by name and every attempt to open it fails, which
    reads exactly like a missing or busy webcam.  Worth naming explicitly.
    """
    base = pathlib.Path(sys.base_prefix)
    if "windowsapps" not in str(base).lower():
        return None

    # sys.base_prefix carries the package *full* name
    # (Name_Version_Arch__Publisher) while the consent store is keyed by the
    # *family* name (Name_Publisher).  Rather than reassemble one from the
    # other, match on the leading name and let Windows own the format.
    name = base.name.split("_")[0]
    value = None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _CONSENT_KEY) as root:
            for i in range(winreg.QueryInfoKey(root)[0]):
                subkey = winreg.EnumKey(root, i)
                if subkey.split("_")[0] != name:
                    continue
                with winreg.OpenKey(root, subkey) as key:
                    value = str(winreg.QueryValueEx(key, "Value")[0])
                break
    except (OSError, ImportError, IndexError):
        return None
    if value is None or value.lower() != "deny":
        return None
    return (
        "Windows is blocking camera access for this Python.\n"
        "  You are running the Microsoft Store build of Python, which Windows\n"
        "  treats as its own app, and its camera permission is set to Deny.\n"
        "\n"
        "  Fix it in either of these ways:\n"
        "    1. Open Settings > Privacy & security > Camera, find Python in the\n"
        "       list of apps, and turn it on. Then run this again.\n"
        "    2. Or install Python from https://www.python.org/downloads/windows/\n"
        "       (ticking \"Add python.exe to PATH\"), delete the .venv folder in\n"
        "       this project, and run run.bat again. A python.org install is an\n"
        "       ordinary desktop app and is not gated separately."
    )


def probe(max_index: int = 5) -> list[tuple[int, str, tuple[int, int]]]:
    """Find cameras that actually deliver a frame.  Used by `check`.

    CAP_ANY is skipped here: it drags in the FFMPEG backend, which has nothing
    to say about webcams except a warning.  `Camera.open` still falls back to
    it for the rare device the named backends miss.
    """
    found = []
    for index in range(max_index):
        for name, api in _BACKENDS[:2]:
            cap = cv2.VideoCapture(index, api)
            try:
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        found.append((index, name, (frame.shape[1], frame.shape[0])))
                        break
            finally:
                cap.release()
    return found


COMMON_MODES = ((1920, 1080), (1280, 720), (960, 540), (640, 480))


def benchmark(index: int = 0, modes=COMMON_MODES, seconds: float = 1.2
              ) -> list[tuple[int, int, float, str]]:
    """Measure what each resolution actually delivers: (w, h, fps, format).

    Worth measuring rather than assuming.  A webcam offers each resolution in
    particular pixel formats, and the fast compressed one is often available at
    some sizes and not others -- so a *smaller* frame can easily arrive slower.
    Guessing here produces confidently wrong advice.
    """
    results = []
    for want_w, want_h in modes:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
            cap.set(cv2.CAP_PROP_FPS, 60)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            for _ in range(5):
                cap.read()
            start = time.perf_counter()
            count = 0
            while time.perf_counter() - start < seconds:
                if cap.read()[0]:
                    count += 1
            fps = count / (time.perf_counter() - start)
            try:
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(
                    4, "little").decode("ascii", "replace")
            except (ValueError, OverflowError):
                fourcc = "?"
            results.append((frame.shape[1], frame.shape[0], fps, fourcc))
        finally:
            cap.release()
    return results


class Camera:
    """Always-newest-frame webcam reader.

    The grab loop runs in its own thread and keeps only the most recent frame.
    Reading straight from VideoCapture on the main loop would hand back
    whatever is next in the driver queue, and any hitch in processing turns
    into permanent cursor lag as that queue backs up.
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 60):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp: float = 0.0
        self._seq: int = 0
        self.backend = ""

    def open(self) -> "Camera":
        last_error = "no backend accepted the device"
        for name, api in _BACKENDS:
            cap = cv2.VideoCapture(self.index, api)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                last_error = f"{name} opened the device but returned no frames"
                continue
            self._cap = cap
            self.backend = name
            self.height, self.width = frame.shape[:2]
            with self._lock:
                self._frame, self._stamp, self._seq = frame, time.monotonic(), 1
            break
        else:
            hint = permission_hint()
            if hint:
                raise CameraError(
                    f"Could not open camera {self.index}.\n\n{hint}")
            raise CameraError(
                f"Could not open camera {self.index} ({last_error}). "
                "Close anything else using the webcam (Teams, Zoom, the Camera app), "
                "check Settings > Privacy & security > Camera, then run "
                "`run.bat check` to list the devices that work."
            )

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._stamp = time.monotonic()
                self._seq += 1

    def latest(self) -> tuple[np.ndarray | None, float, int]:
        """Newest frame, its capture time, and a sequence number.

        The sequence number lets callers skip work when no new frame arrived.
        """
        with self._lock:
            if self._frame is None:
                return None, 0.0, 0
            return self._frame, self._stamp, self._seq

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
