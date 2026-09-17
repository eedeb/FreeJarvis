"""Jarvis: a FreeClaw agent wired into the gauntlet overlay.

The pieces, and why they are separate:

    settings.py   where FreeClaw is and how to reach it, on disk
    client.py     the FreeClaw HTTP surfaces -- admin, chat, streaming
    tools.py      the MCP server this machine exposes *to* FreeClaw
    link.py       "Add FreeClaw": one call that wires the two together
    actions.py    the half of those tools that acts on this machine
    reactor.py    the arc reactor drawn in the corner of the overlay
    terminal.py   the session log under it: what was asked, and what ran
    widget.py     a see-through window that answers the mouse
    audio_panel.py / toggle.py / dialog.py   the rest of the interface

The direction of the arrows is the thing to keep straight. This app *calls*
FreeClaw to ask a question (client.py), and FreeClaw *calls back* into this
app to do something on this machine (tools.py). That is why the MCP server is
HTTP rather than the more usual stdio: a stdio server is a child process of
FreeClaw, so it can only ever act on the machine FreeClaw is on. Serving it
over HTTP is what lets you point Jarvis at a FreeClaw on another box and still
have "make me a folder" mean a folder *here*.
"""

from .settings import JarvisSettings, load, save, update       # noqa: F401
