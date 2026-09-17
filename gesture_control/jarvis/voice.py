""""Hey Jarvis" — wake word in, text out, and a voice to answer with.

Three models, each picked so the gap between finishing a sentence and Jarvis
acting on it is as short as it can be:

  * openWakeWord's pretrained `hey_jarvis` scores 20ms of audio in a fraction
    of a millisecond, so it can run against every frame the microphone
    produces for the cost of a rounding error.
  * webrtcvad decides, just as cheaply and frame by frame, when the user has
    stopped talking — so the recording ends when they do rather than after a
    fixed timer.
  * faster-whisper, int8 on the CPU, transcribes in well under a second once
    warm.

Nothing after the microphone leaves this machine, and none of it runs on the
overlay's thread: one worker owns the state machine, fed by a queue the audio
callback fills.  PortAudio's callback has to return in microseconds, so the
callback itself does nothing but copy.

    ARMED --(wake word)--> CAPTURING --(silence)--> TRANSCRIBING --> ARMED
      ^                        ^                           |
      |                        +--(voice, during a         |
      +---- on_speech(text) -------- follow-up window)-----+

**Three things here exist to make it feel like talking rather than
operating.**

*The follow-up window.*  For a few seconds after Jarvis finishes speaking,
plain speech starts a capture — no second "hey Jarvis".  Conversation is
turn-taking, and having to re-summon an assistant between every turn is the
single thing that most makes one feel like a command line with a microphone
attached.  Outside that window the wake word is required again, because a
microphone that reacts to any noise in the room is worse than one that
sometimes needs asking twice.

*Barge-in.*  The wake word keeps being scored **while Jarvis is talking**, so
saying "hey Jarvis" over the top of a long answer cuts it off and starts
listening.  The older behaviour deafened the microphone during playback to
stop Jarvis waking itself, which also made it impossible to interrupt — you
had to sit through whatever it had decided to say.  Only the wake word is
scored during playback, never the general voice detector, because the
speakers are what the microphone is mostly hearing at that point.

*Speaking a sentence at a time.*  The speaker is a queue with a synthesiser
running one item ahead of playback, so the first sentence of a reply starts
while the rest of it is still being generated.  Waiting for a whole paragraph
before making any sound is most of what makes an assistant feel slow, and the
fix costs nothing but ordering.

Every import is deferred and every failure is survivable.  A machine with no
microphone, or without the optional packages installed, gets `available =
False` and a reason to show — the overlay still runs, it just cannot be
spoken to.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Callable

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000        # 320

# Audio kept from before the wake word is confirmed and stitched onto the
# front of the capture. openWakeWord can take ~80ms to confirm, so without
# this the start of whatever follows "hey Jarvis" is clipped.
PREROLL_FRAMES = 15                                    # 300ms

# Silence that ends a capture. 600ms cut people off on an ordinary breath;
# this is the actual naturalness/latency knob.
TRAILING_SILENCE_MS = 900
MIN_UTTERANCE_S = 0.35
MAX_UTTERANCE_S = 20.0

# openWakeWord's own examples use 0.5. Lower catches more true positives at
# the cost of more false ones.
WAKE_THRESHOLD = 0.5

# How many consecutive voiced frames start a capture inside the follow-up
# window. The wake word is a deliberate act and one detection is enough; a
# follow-up is triggered by ordinary speech, so it needs to be sure the room
# is being talked *to* rather than just being noisy. 6 frames is 120ms --
# longer than a door closing, shorter than any syllable worth catching.
FOLLOWUP_VOICED_FRAMES = 6

# What tiny.en invents out of silence: no real word, just punctuation. The
# test is deliberately narrow -- a genuine one-word reply ("Yes.", "Stop.")
# has to get through.
_NO_WORDS = re.compile(r"^[\s.,!?;:\-]*$")

# Words this assistant hears constantly and a small model reliably mangles.
# faster-whisper takes a prompt as context for the decoder, which biases it
# towards these spellings without constraining it to them -- "open Chrome"
# stops coming back as "open chrome" or "open crome".
DECODER_PROMPT = (
    "Jarvis, Chrome, Spotify, Discord, Notepad, Explorer, VS Code, "
    "volume, mute, monitor, CPU, memory, battery, Wi-Fi, network, "
    "screenshot, clipboard, minimise, desktop, downloads."
)

# Where a sentence ends, for speaking a reply as it arrives. Requires the
# punctuation to be followed by a space or the end, so "3.5" and "e.g." do
# not each become their own utterance.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*(?:\s+|$)")

# Shortest fragment worth sending to the synthesiser on its own. Below this
# the network round trip costs more than it saves, and the reply comes out
# chopped.
MIN_SPOKEN_CHARS = 24


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Complete sentences out of a growing buffer, and what is left over.

    Used to start speaking a reply before the model has finished writing it.
    The leftover matters as much as the sentences: it is the half-written
    tail, and speaking it would produce a confident fragment ending mid-word.
    """
    out, rest = [], buffer
    while True:
        found = _SENTENCE_END.search(rest)
        if not found:
            break
        piece, rest = rest[:found.end()].strip(), rest[found.end():]
        if piece:
            out.append(piece)
    # Re-join anything too short to be worth its own trip to the synthesiser.
    merged: list[str] = []
    for piece in out:
        if merged and len(merged[-1]) < MIN_SPOKEN_CHARS:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    if merged and len(merged[-1]) < MIN_SPOKEN_CHARS:
        rest = f"{merged.pop()} {rest}"
    return merged, rest


class Voice:
    """The microphone half. `available` is False if it could not start."""

    def __init__(self, on_speech: Callable[[str], None],
                 on_state: Callable[[str], None] | None = None,
                 on_interrupt: Callable[[], None] | None = None,
                 model: str = "tiny.en", device=None) -> None:
        self.available = False
        self.unavailable_reason = ""
        self.notes: list[str] = []
        self.on_speech = on_speech
        self.on_state = on_state or (lambda state: None)
        # Called when the wake word lands while Jarvis is talking. Whoever
        # owns the speaker stops it; this module does not know about that.
        self.on_interrupt = on_interrupt or (lambda: None)
        self.model = model
        self.device = device
        self.level = 0.0                    # 0..1, for the reactor to pulse on
        self._frames: queue.Queue = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._followup_until = 0.0
        self._whisper = None
        self._stream = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        try:
            import sounddevice                                     # noqa: F401
            import webrtcvad                                       # noqa: F401
            from openwakeword.model import Model                   # noqa: F401
            from faster_whisper import WhisperModel                # noqa: F401
        except ImportError as exc:
            self.unavailable_reason = (
                f"Voice needs a package that is not installed ({exc.name}). "
                f"Install the extras:  pip install -r requirements.txt")
            return False
        self._thread = threading.Thread(target=self._run, name="jarvis-voice",
                                        daemon=True)
        self._thread.start()
        # The worker reports back through unavailable_reason if it cannot get
        # going; give it long enough to load three models on a cold start.
        for _ in range(600):
            if self.available or self.unavailable_reason:
                break
            time.sleep(0.05)
        return self.available

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:                                      # noqa: BLE001
                pass
            self._stream = None
        self.available = False

    def pause(self) -> None:
        """Stop reacting to ordinary speech — while Jarvis is talking.

        Not deaf: the wake word is still scored, because interrupting is the
        one thing you most want to do to something that is talking too long.
        """
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def expect_reply(self, seconds: float) -> None:
        """Take speech without the wake word for a while -- a turn just ended."""
        self._followup_until = time.monotonic() + max(0.0, float(seconds))

    def drop_followup(self) -> None:
        self._followup_until = 0.0

    @property
    def expecting(self) -> bool:
        return time.monotonic() < self._followup_until

    # -- the worker --------------------------------------------------------

    def _load_whisper(self, name: str):
        from faster_whisper import WhisperModel

        return WhisperModel(name, device="cpu", compute_type="int8")

    def _cached(self, name: str) -> bool:
        """Whether a Whisper model is already on disk.

        Asked before loading, because a model that is not cached is a download
        of a hundred-odd megabytes, and doing that inside startup makes the
        app look hung with no way to tell that it is working.
        """
        import pathlib

        for root in (pathlib.Path.home() / ".cache/huggingface/hub",
                     pathlib.Path.home() / "AppData/Local/huggingface/hub"):
            if list(root.glob(f"models--*faster-whisper-{name}")):
                return True
        return False

    def _upgrade_later(self, name: str) -> None:
        """Fetch a better model in the background and swap it in when it lands.

        This is why the default can be a model that is not installed yet
        without that costing a slow first run: Jarvis starts listening
        immediately on whatever is already here, and gets better a minute
        later without anybody waiting for it.
        """
        def fetch():
            try:
                better = self._load_whisper(name)
            except Exception:                                      # noqa: BLE001
                return                     # no network, or no such model
            self._whisper = better
            self.notes.append(f"Speech recognition upgraded to {name}.")

        threading.Thread(target=fetch, name="jarvis-stt-upgrade",
                         daemon=True).start()

    def _run(self) -> None:
        try:
            import sounddevice as sd
            import webrtcvad
            from openwakeword.model import Model

            # openWakeWord ships the model separately from the package.
            try:
                wake = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
            except Exception:                                      # noqa: BLE001
                import openwakeword
                openwakeword.utils.download_models(["hey_jarvis"])
                wake = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

            vad = webrtcvad.Vad(2)
            if self._cached(self.model):
                self._whisper = self._load_whisper(self.model)
            else:
                self._whisper = self._load_whisper("tiny.en")
                self.notes.append(
                    f"Listening on tiny.en while {self.model} downloads in the "
                    f"background.")
                self._upgrade_later(self.model)

            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=FRAME_SAMPLES, device=self.device,
                callback=self._on_audio)
            self._stream.start()
        except Exception as exc:                                   # noqa: BLE001
            self.unavailable_reason = f"Could not start listening: {exc}"
            return

        self.available = True
        self.on_state("idle")
        self._listen(wake, vad)

    def _on_audio(self, indata, frames, time_info, status) -> None:
        """The PortAudio callback. Copies and returns, and nothing else."""
        try:
            self._frames.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass                                # sooner drop audio than block

    def _listen(self, wake, vad) -> None:
        from collections import deque

        preroll: deque = deque(maxlen=PREROLL_FRAMES)
        capturing: list[np.ndarray] = []
        silent_ms = 0.0
        voiced_run = 0
        started = 0.0

        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.5)
            except queue.Empty:
                continue

            # A cheap loudness reading for the reactor to pulse on. RMS of an
            # int16 frame, scaled so ordinary speech lands near 1.
            self.level = min(1.0, float(np.sqrt(np.mean(
                (frame.astype(np.float32) / 32768.0) ** 2))) * 12.0)

            if not capturing:
                preroll.append(frame)
                woke = max(wake.predict(frame).values()) >= WAKE_THRESHOLD
                if woke:
                    # Over the top of a reply, this is an interruption: stop
                    # talking and listen. The paused flag is what says Jarvis
                    # currently has the floor.
                    if self._paused.is_set():
                        self.on_interrupt()
                        self._paused.clear()
                elif self._paused.is_set():
                    voiced_run = 0
                    continue           # Jarvis is talking; only the word counts
                elif self.expecting:
                    # A turn just ended, so ordinary speech is the next turn.
                    # Several voiced frames rather than one, because one is
                    # every door and cough in the room.
                    voiced_run = (voiced_run + 1
                                  if vad.is_speech(frame.tobytes(), SAMPLE_RATE)
                                  else 0)
                    if voiced_run < FOLLOWUP_VOICED_FRAMES:
                        continue
                else:
                    voiced_run = 0
                    continue

                voiced_run = 0
                self._followup_until = 0.0
                capturing = list(preroll)     # the pre-roll saves the first word
                preroll.clear()
                silent_ms, started = 0.0, time.monotonic()
                # openWakeWord holds internal state that would re-fire on the
                # same audio; clearing it is what stops a double trigger.
                wake.reset()
                self.on_state("listening")
                continue

            capturing.append(frame)
            voiced = vad.is_speech(frame.tobytes(), SAMPLE_RATE)
            silent_ms = 0.0 if voiced else silent_ms + FRAME_MS
            elapsed = time.monotonic() - started
            if silent_ms < TRAILING_SILENCE_MS and elapsed < MAX_UTTERANCE_S:
                continue

            audio = np.concatenate(capturing)
            capturing = []
            self.on_state("thinking")
            if elapsed >= MIN_UTTERANCE_S:
                text = self._transcribe(audio)
                if text:
                    self.on_speech(text)
                    continue
            self.on_state("idle")

    def _transcribe(self, audio: np.ndarray) -> str:
        try:
            segments, _ = self._whisper.transcribe(
                audio.astype(np.float32) / 32768.0, language="en",
                beam_size=1, vad_filter=False, temperature=0.0,
                initial_prompt=DECODER_PROMPT,
                # Each utterance is its own thing. Carrying the last one in as
                # context makes a small model repeat it when this one is
                # short or unclear, which reads as Jarvis mishearing you as
                # whatever you said a minute ago.
                condition_on_previous_text=False)
            text = " ".join(s.text for s in segments).strip()
        except Exception:                                          # noqa: BLE001
            return ""
        if _NO_WORDS.match(text):
            return ""
        # The wake word usually lands in the transcript too. Strip it, so the
        # agent gets the request rather than the summons.
        return re.sub(r"^\s*(hey|hi|ok|okay)[\s,]+jarvis[\s,.!]*", "", text,
                      flags=re.IGNORECASE).strip() or text


class Speaker:
    """Reads replies aloud, a sentence at a time, and can be cut off.

    A queue rather than a call, for two reasons.  A reply arrives in pieces
    and the first piece should be audible before the last one exists, which
    means something has to hold the order.  And a synthesiser is a network
    round trip, so the only way to avoid a gap between sentences is to be
    fetching the next one while the current one plays -- which is what the
    two stages here are.
    """

    # How long a slice of audio each loudness reading covers. 40ms is roughly
    # a syllable: shorter makes the reactor jitter on individual phonemes,
    # longer smears words together into one flat glow.
    ENVELOPE_MS = 40

    def __init__(self, voice: str = "en-GB-RyanNeural",
                 on_drain: Callable[[], None] | None = None) -> None:
        self.voice = voice
        self.available = False
        self.unavailable_reason = ""
        self.speaking = False
        self.on_drain = on_drain or (lambda: None)
        # 0..1, how loud the reply is *right now*, for the reactor to pulse
        # on. Read from a precomputed envelope by elapsed time rather than
        # measured in an audio callback: the waveform is in memory before
        # playback starts, so the answer is known in advance.
        self.level = 0.0
        self._text: queue.Queue = queue.Queue()
        self._audio: queue.Queue = queue.Queue(maxsize=2)
        self._epoch = 0                  # bumped by stop(), to drop stale work
        self._lock = threading.Lock()
        try:
            import edge_tts                                        # noqa: F401
            import sounddevice                                     # noqa: F401
            import soundfile                                       # noqa: F401
            self.available = True
        except ImportError as exc:
            self.unavailable_reason = (
                f"Speech needs a package that is not installed ({exc.name}).")
            return
        threading.Thread(target=self._synthesise, name="jarvis-tts-synth",
                         daemon=True).start()
        threading.Thread(target=self._play, name="jarvis-tts-play",
                         daemon=True).start()

    # -- what the agent calls ----------------------------------------------

    def say(self, text: str) -> None:
        """Queue something to be said. Returns at once."""
        if not self.available or not text.strip():
            return
        with self._lock:
            self.speaking = True
            self._text.put((self._epoch, text.strip()))

    def stop(self) -> None:
        """Shut up now, and forget anything queued.

        The epoch is what makes this reliable: a sentence already being
        synthesised cannot be recalled, so instead everything carries the
        epoch it was queued under and anything from a previous one is thrown
        away when it surfaces.
        """
        import sounddevice as sd

        with self._lock:
            self._epoch += 1
            for held in (self._text, self._audio):
                while not held.empty():
                    try:
                        held.get_nowait()
                    except queue.Empty:
                        break
            self.speaking = False
        try:
            sd.stop()
        except Exception:                                          # noqa: BLE001
            pass
        self.level = 0.0

    @property
    def idle(self) -> bool:
        return (not self.speaking and self._text.empty()
                and self._audio.empty())

    # -- the two stages ----------------------------------------------------

    def _synthesise(self) -> None:
        import asyncio
        import io

        import edge_tts
        import soundfile as sf

        while True:
            epoch, text = self._text.get()
            if epoch != self._epoch:
                continue                        # stopped while it was waiting
            try:
                buffer = io.BytesIO()

                async def pull():
                    stream = edge_tts.Communicate(text, self.voice)
                    async for chunk in stream.stream():
                        if chunk["type"] == "audio":
                            buffer.write(chunk["data"])

                asyncio.run(pull())
                buffer.seek(0)
                data, rate = sf.read(buffer, dtype="float32")
            except Exception:                                      # noqa: BLE001
                continue          # a mute sentence is still on the session log
            if epoch == self._epoch:
                self._audio.put((epoch, data, rate))

    def _play(self) -> None:
        import sounddevice as sd

        while True:
            epoch, data, rate = self._audio.get()
            if epoch != self._epoch:
                continue
            try:
                mono = data if data.ndim == 1 else data.mean(axis=1)
                envelope = self._envelope(mono, rate)
                started = time.monotonic()
                sd.play(data, rate)
                span = self.ENVELOPE_MS / 1000.0
                # Walk the envelope in step with the clock while the audio
                # plays. sd.wait() would block just as well, but then nothing
                # would be updating the level it was played for.
                while True:
                    stream = sd.get_stream()
                    if stream is None or not stream.active or epoch != self._epoch:
                        break
                    i = int((time.monotonic() - started) / span)
                    self.level = float(envelope[i]) if i < len(envelope) else 0.0
                    time.sleep(span / 2)
            except Exception:                                      # noqa: BLE001
                pass
            finally:
                self.level = 0.0
            # Drained only when nothing is queued behind this and nothing is
            # still being synthesised for it -- otherwise the state machine
            # would call the reply finished between two of its sentences.
            if epoch == self._epoch and self._audio.empty() and self._text.empty():
                with self._lock:
                    self.speaking = False
                try:
                    self.on_drain()
                except Exception:                                  # noqa: BLE001
                    pass

    def _envelope(self, mono, rate: int):
        """One 0..1 loudness reading per ENVELOPE_MS of audio.

        Normalised against this clip's own peak rather than an absolute level,
        so a quiet sentence still swings the reactor the full way -- what is
        wanted here is the shape of the speech, not its volume.
        """
        block = max(1, int(rate * self.ENVELOPE_MS / 1000))
        usable = len(mono) - len(mono) % block
        if usable < block:
            return np.zeros(1, dtype=np.float32)
        blocks = mono[:usable].reshape(-1, block)
        rms = np.sqrt((blocks.astype(np.float32) ** 2).mean(axis=1))
        peak = float(rms.max())
        return rms / peak if peak > 1e-6 else rms
