"""Reading what the computer is playing, and turning it up or down.

Two independent halves, because they fail independently:

  * **`Tap`** watches the output.  First choice is WASAPI loopback -- the
    actual samples Windows is sending to the speakers -- which gives a real
    frequency spectrum.  If that cannot be opened it falls back to the
    endpoint's own peak meter, the same number the Windows volume mixer draws,
    which gives one level rather than a spectrum but always works.
  * **`Volume`** is the master output slider, through Core Audio's
    `IAudioEndpointVolume`, which is the same control the tray icon drives.
    Setting it here moves the real system volume.

**Matching the device by id, not by name.**  The loopback recorder has to be
opened on the same endpoint Windows is actually playing to.  Asking for it by
friendly name silently returned a different one on this machine -- capture ran
happily and recorded pure silence for a second before anyone noticed.  Both
halves therefore resolve the default render endpoint and match on its id.

Everything is optional and everything degrades: no pycaw means no volume
control, no loopback means a level instead of a spectrum, and neither stops
the rest of the app.
"""

from __future__ import annotations

import threading
import time

import numpy as np

# How much audio each spectrum is computed from. 2048 at 48kHz is 43ms: long
# enough to resolve bass, short enough that the bars still move with the beat.
WINDOW = 2048
# The bars, and the range they cover. Spaced by octave rather than evenly,
# because pitch is logarithmic and an even split puts nine tenths of the bars
# above 5kHz where there is usually nothing to see.
BARS = 24
LOW_HZ, HIGH_HZ = 40.0, 14000.0
# Bars fall slower than they rise, which is what makes a meter readable
# instead of a flicker.
RISE, FALL = 0.55, 0.12


class Volume:
    """The system master volume. `available` is False if it could not attach."""

    def __init__(self) -> None:
        self.available = False
        self.unavailable_reason = ""
        self._endpoint = None
        try:
            from pycaw.pycaw import AudioUtilities

            self._endpoint = AudioUtilities.GetSpeakers().EndpointVolume
            self._endpoint.GetMasterVolumeLevelScalar()      # prove it answers
            self.available = True
        except Exception as exc:                             # noqa: BLE001
            self.unavailable_reason = (
                f"No control of the system volume ({type(exc).__name__}). "
                f"pip install pycaw")

    def get(self) -> float:
        """0..1, or 0 if unavailable."""
        if not self.available:
            return 0.0
        try:
            return float(self._endpoint.GetMasterVolumeLevelScalar())
        except Exception:                                    # noqa: BLE001
            return 0.0

    def set(self, level: float) -> None:
        if not self.available:
            return
        try:
            self._endpoint.SetMasterVolumeLevelScalar(
                float(min(max(level, 0.0), 1.0)), None)
        except Exception:                                    # noqa: BLE001
            pass

    def muted(self) -> bool:
        if not self.available:
            return False
        try:
            return bool(self._endpoint.GetMute())
        except Exception:                                    # noqa: BLE001
            return False

    def set_muted(self, muted: bool) -> None:
        if not self.available:
            return
        try:
            self._endpoint.SetMute(bool(muted), None)
        except Exception:                                    # noqa: BLE001
            pass


class Tap:
    """What the speakers are playing, as `bars` (0..1) and `level` (0..1)."""

    def __init__(self) -> None:
        self.available = False
        self.unavailable_reason = ""
        self.spectral = False            # True when the bars are a real FFT
        self.bars = np.zeros(BARS, np.float32)
        self.level = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._edges: np.ndarray | None = None

    def start(self) -> bool:
        self._thread = threading.Thread(target=self._run, name="jarvis-audio",
                                        daemon=True)
        self._thread.start()
        for _ in range(80):              # opening a WASAPI stream is not instant
            if self.available or self.unavailable_reason:
                break
            time.sleep(0.05)
        return self.available

    def stop(self) -> None:
        self._stop.set()
        self.available = False

    # -- the worker --------------------------------------------------------

    def _run(self) -> None:
        if self._loopback():
            return
        self._meter()

    def _loopback(self) -> bool:
        """Real samples off the speakers. False if that cannot be opened."""
        try:
            import soundcard as sc

            speaker = sc.default_speaker()
            # By id. Matching on the friendly name picked a different endpoint
            # here and recorded silence -- see the module docstring.
            devices = [m for m in sc.all_microphones(include_loopback=True)
                       if m.isloopback and m.id == speaker.id]
            if not devices:
                self.unavailable_reason = "No loopback device for the speakers."
                return False
            rate = 48000
            self._plan(rate)
            # A visualiser that drops a block has nothing to recover; the
            # warning would just scroll the terminal the user reads status in.
            import warnings
            warnings.filterwarnings("ignore", message="data discontinuity")
            with devices[0].recorder(samplerate=rate, channels=2,
                                     blocksize=WINDOW // 4) as recorder:
                self.available = True
                self.spectral = True
                while not self._stop.is_set():
                    block = recorder.record(numframes=WINDOW)
                    mono = block.mean(axis=1) if block.ndim > 1 else block
                    self._absorb(mono)
        except Exception as exc:                             # noqa: BLE001
            if not self.available:
                self.unavailable_reason = (
                    f"No loopback capture ({type(exc).__name__}); "
                    f"falling back to the output meter.")
            return False
        return True

    def _meter(self) -> None:
        """The endpoint's own peak level, when the samples are out of reach.

        One number rather than a spectrum, so the bars become a history: the
        display scrolls and each bar is the level a moment ago. Still driven
        by what is genuinely coming out of the speakers.
        """
        try:
            import comtypes
            from pycaw.pycaw import AudioUtilities, IAudioMeterInformation

            device = AudioUtilities.GetSpeakers()
            meter = device._dev.Activate(
                IAudioMeterInformation._iid_, comtypes.CLSCTX_ALL,
                None).QueryInterface(IAudioMeterInformation)
            meter.GetPeakValue()
        except Exception as exc:                             # noqa: BLE001
            self.unavailable_reason = (
                f"Cannot read the output level ({type(exc).__name__}).")
            return
        self.available = True
        self.spectral = False
        while not self._stop.is_set():
            try:
                peak = float(meter.GetPeakValue())
            except Exception:                                # noqa: BLE001
                peak = 0.0
            self.level = peak
            self.bars[:-1] = self.bars[1:]                   # scroll left
            self.bars[-1] = peak
            time.sleep(0.033)

    # -- turning samples into bars -----------------------------------------

    def _plan(self, rate: int) -> None:
        """Which FFT bins belong to which bar, worked out once."""
        edges = np.geomspace(LOW_HZ, HIGH_HZ, BARS + 1)
        self._edges = np.clip((edges * WINDOW / rate).astype(int), 0, WINDOW // 2)
        self._window = np.hanning(WINDOW).astype(np.float32)

    def _absorb(self, mono: np.ndarray) -> None:
        if len(mono) < WINDOW:
            mono = np.pad(mono, (0, WINDOW - len(mono)))
        mono = mono[:WINDOW].astype(np.float32)
        self.level = float(np.sqrt((mono ** 2).mean()) * 6.0)

        spectrum = np.abs(np.fft.rfft(mono * self._window))
        fresh = np.empty(BARS, np.float32)
        for i in range(BARS):
            lo, hi = self._edges[i], max(self._edges[i] + 1, self._edges[i + 1])
            fresh[i] = spectrum[lo:hi].mean()
        # Loudness is logarithmic, so the bars are too -- on a linear scale
        # music is one tall bar at the bottom and nothing else. -60dB..0dB.
        with np.errstate(divide="ignore"):
            decibels = 20.0 * np.log10(np.maximum(fresh, 1e-9) / (WINDOW * 0.02))
        fresh = np.clip((decibels + 60.0) / 60.0, 0.0, 1.0)

        rising = fresh > self.bars
        self.bars += (fresh - self.bars) * np.where(rising, RISE, FALL)
        self.level = float(min(1.0, max(self.level, self.bars.max() * 0.8)))
