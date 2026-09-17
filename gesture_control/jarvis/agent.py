"""Jarvis, as the overlay sees it: one object with a state and a line of text.

Everything asynchronous is hidden behind this.  The overlay's draw loop calls
`state()` and `caption()` sixty times a second and must never block, so every
slow thing -- a FreeClaw turn, a transcription, a sentence being spoken --
happens on its own thread and leaves its result in an attribute.

The state machine is the same one the reactor animates:

    idle  --heard "hey jarvis"-->  listening
    listening  --stopped talking-->  thinking
    thinking  --FreeClaw answered-->  speaking  --finished-->  idle

`thinking` is the only state that can last a while, because a FreeClaw turn
can search the web and run tools before it says anything.  That is exactly why
the reactor has a state for it.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from . import client, settings as store
from .terminal import shorten
from .tools import ToolServer, allow_keyboard
from .voice import split_sentences

# How long a reply stays on the overlay after it has been read out.
CAPTION_LINGER = 8.0
# A reply is one line on a translucent overlay, not a document.
CAPTION_LIMIT = 220
# How much of the conversation to keep. This is the terminal's scrollback, so
# it is worth holding a session's worth rather than a screenful: entries are
# short, capped, and 300 of them is a few hundred kilobytes.
TRANSCRIPT_DEPTH = 300


class Jarvis:
    """The whole assistant, or as much of it as could be started."""

    def __init__(self, settings: store.JarvisSettings | None = None) -> None:
        self.settings = settings or store.load()
        self.state = "idle"
        self.level = 0.0
        self.notes: list[str] = []          # what could not be started, and why

        self._caption = ""
        self._caption_until = 0.0
        # What to show in the terminal: (kind, text) newest last, where kind
        # is one of you / note / reasoning / tool / jarvis / error.
        self.transcript: deque[tuple[str, str]] = deque(maxlen=TRANSCRIPT_DEPTH)
        # This turn's still-growing entry, per streamed kind. Cleared between
        # turns so a new question never appends to the last one's reply.
        self._open: dict[str, list] = {}
        # Set when the wake word lands, so the reactor can flash once. A flag
        # rather than a call, because the reactor is drawn on the overlay's
        # thread and this is set on the microphone's.
        self.woke = False
        self._lock = threading.Lock()
        self._busy = threading.Lock()
        self._chat: client.Chat | None = None
        self._tools: ToolServer | None = None
        self._voice = None
        self._speaker = None
        # The part of the reply that has arrived but is not yet a whole
        # sentence, so it cannot be spoken yet without ending mid-word.
        self._unsaid = ""
        self._spoke_any = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Bring up whatever this machine and this configuration allow."""
        if not self.settings.ready():
            self.notes.append("Jarvis is not linked to a FreeClaw yet "
                              "(Ctrl+Alt+J to add one).")
            return

        self._chat = client.Chat(self.settings.url, self.settings.password,
                                 self.settings.user)

        # The sensors first, so that by the time anything asks "how has the
        # CPU been" there is a minute of history to answer with rather than a
        # single reading taken the moment the question arrived.
        from . import sensors, watch
        sensors.sampler()

        # Standing orders can speak on their own, so they need a way to say
        # something and a way to know when not to. Both come from here: this
        # is the only object that knows whether a turn is in flight.
        watch.bind(self.announce, self._free_to_speak)

        # Then the tools: FreeClaw may call one the moment a turn starts.
        allow_keyboard(self.settings.tools_keyboard)
        self._tools = ToolServer(self.settings.tool_token, self.settings.tool_port)
        if not self._tools.running:
            self.notes.append(self._tools.error or "Jarvis's tool server did not start.")
            self._tools = None

        if self.settings.speech:
            from .voice import Speaker
            self._speaker = Speaker(voice=self.settings.voice_name,
                                    on_drain=self._finished_speaking)
            if not self._speaker.available:
                self.notes.append(self._speaker.unavailable_reason)
                self._speaker = None

        if self.settings.voice:
            from .voice import Voice
            self._voice = Voice(on_speech=self._heard, on_state=self._set_state,
                                on_interrupt=self._interrupted,
                                model=self.settings.speech_model)
            if not self._voice.start():
                self.notes.append(self._voice.unavailable_reason)
                self._voice = None
            else:
                self.notes.extend(self._voice.notes)
                self.say_line("Jarvis online.")
                self.speak("Jarvis online.")

    def stop(self) -> None:
        if self._speaker is not None:
            self._speaker.stop()
        if self._voice is not None:
            self._voice.stop()
            self._voice = None
        if self._tools is not None:
            self._tools.stop()
            self._tools = None
        # Cards are windows, and a window nobody owns outlives the app that
        # made it -- they have to be taken down here rather than left to the
        # process exiting.
        from . import cards, watch
        cards.shutdown()
        watch.shutdown()

    @property
    def running(self) -> bool:
        return self._tools is not None or self._voice is not None

    # -- what the overlay reads --------------------------------------------

    def _set_state(self, state: str) -> None:
        # The wake word is the one transition worth reacting to rather than
        # just settling into, so entering "listening" arms a one-off flash.
        if state == "listening" and self.state != "listening":
            self.woke = True
        self.state = state

    def took_wake(self) -> bool:
        """True once per wake word, for whoever draws the flash."""
        if not self.woke:
            return False
        self.woke = False
        return True

    def caption(self) -> str:
        """The line to show, or empty once it has been up long enough."""
        with self._lock:
            if self._caption and time.monotonic() > self._caption_until:
                self._caption = ""
            return self._caption

    def say_line(self, text: str, seconds: float = CAPTION_LINGER) -> None:
        """Put a line on the overlay without involving FreeClaw."""
        text = " ".join(text.split())
        if len(text) > CAPTION_LIMIT:
            text = text[:CAPTION_LIMIT - 1].rstrip() + "…"
        with self._lock:
            self._caption = text
            self._caption_until = time.monotonic() + seconds

    def log(self) -> list[tuple[str, str]]:
        """The whole conversation so far, oldest first.

        No expiry, unlike the caption. This is a terminal's scrollback: the
        column used to be painted over the middle of the screen and had to get
        out of the way, and a window with a size of its own does not.
        """
        with self._lock:
            return [(k, t) for k, t in self.transcript]

    def _add(self, kind: str, text: str) -> None:
        with self._lock:
            self.transcript.append([kind, text])

    def _extend(self, kind: str, text: str) -> None:
        """Grow this turn's entry of that kind, starting one if there is none.

        This is what makes a reply appear a word at a time rather than as a
        new slab per token -- `token` events arrive in fragments and would
        otherwise fill the panel with one letter each.

        It grows *this turn's* entry rather than the last one, because the two
        streamed kinds interleave. A real turn produced:

            token "The"  /  reasoning "."  /  token " file has been created"

        and merging only with the immediately preceding entry split that reply
        into two slabs with a stray full stop between them. Only entries of
        the same kind merge at all, which is why the streamed kinds are
        distinct from the one-shot notes that share their styling ("note") --
        merging by style instead glued reasoning onto the end of "asking Groq".
        """
        with self._lock:
            entry = self._open.get(kind)
            # Identity, not equality: the deque may have evicted it, and
            # appending to something no longer on screen would look like the
            # reply had stopped arriving.
            if entry is not None and any(e is entry for e in self.transcript):
                entry[1] += text
            else:
                entry = [kind, text]
                self.transcript.append(entry)
                self._open[kind] = entry

    def clear(self) -> None:
        with self._lock:
            self.transcript.clear()
            self._open.clear()

    def forget(self) -> None:
        """Start over: clear the session log and FreeClaw's conversation.

        Both, because either alone is a lie. Clearing only the log hides a
        history the next turn still arrives with; clearing only FreeClaw
        leaves the screen showing a conversation that no longer exists.
        """
        self.clear()
        if self._chat is None:
            self.say_line("Nothing to forget, sir.")
            return
        try:
            self._chat.reset()
        except client.FreeClawError as exc:
            self._add("error", str(exc))
            self.say_line(str(exc))
            return
        self._add("note", "[conversation reset]")
        self.say_line("Conversation cleared, sir.")
        self.speak("Conversation cleared, sir.")

    def voice_level(self) -> float:
        """0..1, for the reactor to pulse on."""
        if self._voice is not None and self.state == "listening":
            return self._voice.level
        if self._speaker is not None and self._speaker.speaking:
            # The real playback envelope, so the ring moves with the words
            # rather than sitting at a constant "talking" brightness.
            return self._speaker.level
        return 0.0

    # -- a turn ------------------------------------------------------------

    def _heard(self, text: str) -> None:
        """The microphone produced a sentence. Runs on the voice thread."""
        self.ask(text)

    def ask(self, text: str) -> None:
        """Put a question to FreeClaw. Returns at once; the answer arrives later."""
        if not text.strip():
            self._set_state("idle")
            return
        if self._chat is None:
            self.say_line("Jarvis is not linked to a FreeClaw yet.")
            self._set_state("idle")
            return
        if not self._busy.acquire(blocking=False):
            # A turn is already running. Saying so beats silently dropping it.
            self.say_line("One moment, sir.")
            return
        threading.Thread(target=self._turn, args=(text,),
                         name="jarvis-turn", daemon=True).start()

    def _turn(self, text: str) -> None:
        """One agent turn, narrated into the transcript as it happens."""
        try:
            self._set_state("thinking")
            self._open.clear()
            self._unsaid, self._spoke_any = "", False
            self._add("you", text)
            self.say_line("“" + text + "”", seconds=600.0)
            reply = ""
            try:
                for event in self._chat.turn(text):
                    reply = self._event(event, reply)
                    if event.get("type") in ("done", "error", "stopped"):
                        break
            except client.FreeClawError as exc:
                # Said out loud, not only written: this is an assistant you
                # talk to with your hands full, and an error you cannot hear
                # is indistinguishable from it having ignored you.
                self._add("error", str(exc))
                self.say_line(str(exc))
                self._finish_speaking(str(exc))
                return

            reply = " ".join(reply.split())
            if not reply:
                reply = "I have nothing to say to that, sir."
                self._add("jarvis", reply)
            self.say_line(reply)
            self._finish_speaking(reply)
        finally:
            self._busy.release()

    def _event(self, event: dict, reply: str) -> str:
        """Fold one streamed event into the transcript. Returns the reply so far."""
        kind = event.get("type")
        if kind == "token":
            chunk = event.get("text") or ""
            self._extend("jarvis", chunk)
            # Speak whole sentences as they land, rather than waiting for the
            # turn to end. Most of what makes an assistant feel slow is the
            # silence between asking and the first sound, and that silence is
            # as long as the whole answer takes to write.
            self._say_ready(chunk)
            return reply + chunk
        if kind == "reasoning":
            # Collapsed into one growing entry: reasoning arrives in fragments
            # and is the least important thing on the panel.
            self._extend("reasoning", event.get("text") or "")
        elif kind == "tool_call":
            args = event.get("arguments") or {}
            detail = ", ".join(f"{k}={shorten(v, 60)}" for k, v in args.items())
            self._add("tool", f"{event.get('name')}({detail})")
        elif kind == "tool_result":
            self._add("tool", f"→ {shorten(event.get('result'), 200)}")
        elif kind == "tool_throttled":
            self._add("tool", f"{event.get('name')} held back -- it was looping")
        elif kind == "intent":
            self._add("note", f"[{event.get('tag')}]")
        elif kind == "provider":
            self._add("note", f"asking {event.get('name')}")
        elif kind == "error":
            self._add("error", str(event.get("error") or "FreeClaw failed."))
        elif kind == "stopped":
            self._add("error", "Stopped.")
        return reply

    # -- saying it out loud ------------------------------------------------

    def speak(self, text: str) -> None:
        """Say something that did not come from FreeClaw."""
        if self._speaker is None:
            return
        self._hold_microphone()
        self._speaker.say(text)

    def _hold_microphone(self) -> None:
        """Jarvis has the floor: ordinary speech stops counting.

        The wake word keeps being scored, which is what makes interrupting
        possible -- see voice.Voice._listen.
        """
        self._set_state("speaking")
        if self._voice is not None:
            self._voice.drop_followup()
            self._voice.pause()

    def _say_ready(self, chunk: str) -> None:
        """Hand the synthesiser any complete sentence the reply has produced."""
        if self._speaker is None:
            return
        self._unsaid += chunk
        ready, self._unsaid = split_sentences(self._unsaid)
        for sentence in ready:
            if not self._spoke_any:
                self._spoke_any = True
                self._hold_microphone()
            self._speaker.say(sentence)

    def _finish_speaking(self, reply: str) -> None:
        """Flush whatever is left at the end of a turn."""
        if self._speaker is None:
            self._set_state("idle")
            self._listen_again()
            return
        tail, self._unsaid = self._unsaid.strip(), ""
        if not self._spoke_any:
            # Nothing streamed -- a provider that answers in one piece, or a
            # line Jarvis wrote itself. Say the whole thing.
            self._hold_microphone()
            self._speaker.say(reply)
        elif tail:
            self._speaker.say(tail)
        self._spoke_any = False

    def _finished_speaking(self) -> None:
        """The speaker drained. Give the microphone back, and expect a reply."""
        if self._voice is not None:
            self._voice.resume()
        self._set_state("idle")
        self._listen_again()

    def _listen_again(self) -> None:
        """Open the window where plain speech counts as the next turn."""
        if self._voice is not None and self.settings.followup_seconds > 0:
            self._voice.expect_reply(self.settings.followup_seconds)

    def announce(self, text: str) -> None:
        """Say something nobody asked for -- a standing order tripping."""
        self._add("note", f"[watch] {text}")
        self.say_line(text)
        self.speak(text)

    def _free_to_speak(self) -> bool:
        """Whether now is a reasonable moment to volunteer something.

        Never over the top of a turn or of a reply being read out: an
        assistant that interrupts your own conversation to mention the CPU is
        an alarm clock. The announcement waits rather than being dropped.
        """
        if self._busy.locked() or self.state != "idle":
            return False
        return self._speaker is None or self._speaker.idle

    def _interrupted(self) -> None:
        """"Hey Jarvis" arrived over the top of a reply. Stop talking."""
        if self._speaker is not None:
            self._speaker.stop()
        self._unsaid = ""
        self._spoke_any = False
        self._add("note", "[interrupted]")
