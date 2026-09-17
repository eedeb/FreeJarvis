"""Where FreeClaw is, and the secrets needed to talk to it both ways.

Kept out of the app's own settings.json because the shapes are different: that
file is full of tuning constants a user is meant to read and edit, this one is
two passwords and an address written by a dialog box. Mixing them would put
credentials in a file the README tells people to open.

Both secrets are plain text on a local disk, which is the same protection
FreeClaw gives its own `.env`. Encrypting them here would only move the
problem: the key would have to sit beside them to be readable at startup.
"""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import asdict, dataclass, fields

from ..config import ROOT

JARVIS_PATH = ROOT / "jarvis.json"

# The FreeClaw user this app drives. Its context.md is the persona, and its
# conversation is the one "hey jarvis" continues.
DEFAULT_USER = "Jarvis"

# The port this app serves its MCP tools on. FreeClaw dials in here, so it has
# to be reachable from wherever FreeClaw runs -- which for a remote FreeClaw
# means this port open on this machine.
DEFAULT_TOOL_PORT = 8788


@dataclass
class JarvisSettings:
    # --- reaching FreeClaw ---------------------------------------------
    url: str = ""
    password: str = ""
    user: str = DEFAULT_USER

    # --- FreeClaw reaching back ----------------------------------------
    # The address FreeClaw should dial to run a tool on this machine. Worked
    # out at link time rather than stored as a guess: which of this machine's
    # addresses is the right one depends on where FreeClaw is looking from.
    tool_url: str = ""
    tool_port: int = DEFAULT_TOOL_PORT
    # Bearer token for that server. Minted once and kept -- it is written into
    # FreeClaw's own config when the MCP server is registered, so rotating it
    # on every start would leave FreeClaw holding a stale one.
    tool_token: str = ""

    # --- behaviour ------------------------------------------------------
    linked: bool = False
    # Listen for "hey jarvis" on the default microphone.
    voice: bool = True
    # Speak replies. Off leaves the reply on the overlay to be read.
    speech: bool = True
    # Which faster-whisper model transcribes what you say. base.en is
    # noticeably better than tiny.en on short commands and still real-time on
    # any modern CPU. If it is not downloaded yet Jarvis listens on tiny.en
    # and swaps over in the background, so this never costs a slow start.
    speech_model: str = "base.en"
    # The voice it answers in -- any edge-tts name.
    voice_name: str = "en-GB-RyanNeural"
    # How long after a reply plain speech counts as the next turn, with no
    # second "hey Jarvis". 0 turns conversation off and requires the wake
    # word every time.
    followup_seconds: float = 9.0
    # Where the reactor sits, as a fraction of the shorter screen edge.
    reactor_size: float = 0.26
    # The audio widget: the output spectrum and the system volume slider.
    audio_widget: bool = True
    # The session-log terminal under the orb, and its size as a fraction of
    # the screen. Kept well clear of the bottom edge on purpose -- it is a
    # window onto the conversation, not a second desktop.
    terminal: bool = True
    terminal_width: float = 0.30
    terminal_height: float = 0.30
    # Let the agent type and press keys on this machine. Every other tool it
    # has is bounded -- files to a few folders, windows to ones already open --
    # and this one is not: it can do anything you could do at the keyboard,
    # including typing into a terminal. It is on because an assistant that
    # cannot type is much less useful, but it is the one to turn off first if
    # you are letting it read untrusted pages.
    tools_keyboard: bool = True

    def ready(self) -> bool:
        """Whether there is something to talk to."""
        return bool(self.linked and self.url and self.password)


_lock = threading.Lock()


def load() -> JarvisSettings:
    """The saved settings, ignoring any key this version does not know."""
    known = {f.name for f in fields(JarvisSettings)}
    try:
        saved = json.loads(JARVIS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return JarvisSettings()
    if not isinstance(saved, dict):
        return JarvisSettings()
    return JarvisSettings(**{k: v for k, v in saved.items() if k in known})


def save(settings: JarvisSettings) -> JarvisSettings:
    with _lock:
        tmp = JARVIS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        tmp.replace(JARVIS_PATH)
    return settings


def update(**changes) -> JarvisSettings:
    settings = load()
    for key, value in changes.items():
        setattr(settings, key, value)
    return save(settings)


def token() -> str:
    """The tool server's bearer token, minting one the first time."""
    settings = load()
    if not settings.tool_token:
        settings = update(tool_token=secrets.token_urlsafe(32))
    return settings.tool_token
