"""The tools that let Jarvis put things on the user's screen.

This is the part that makes it an assistant with a workbench rather than an
assistant with a chat window.  A reply is gone the moment the next one
arrives; a card stays where it was put, and a monitor stays *true* -- it
redraws itself from the sensors whether or not anyone is talking to Jarvis.

The distinction is worth keeping in the tool names, because the model has to
choose between them correctly and the choice is not obvious from the
situation alone:

    show_monitor   something the machine already measures, watched live
    show_card      something Jarvis worked out, held still until it changes
    show_file      a file on disk -- text, markdown or a picture -- followed

"Watch the CPU" is the first. "Remind me what the four steps were" is the
second. Asking for a card of the CPU would produce a number frozen at the
moment it was asked, which is worse than useless -- it looks live and is not.
"""

from __future__ import annotations

from . import cards


def show_monitor(kind: str) -> str:
    """Put a live gauge on the screen and leave it watching."""
    return cards.board().open_monitor(kind)


def show_card(title: str, body: str, gauge: float = None) -> str:
    """Put a card of Jarvis's own text on the screen, or update one."""
    if not str(title).strip():
        raise cards.CardError("A card needs a title -- it is how I update it later.")
    level = None
    if gauge is not None:
        try:
            level = float(gauge)
        except (TypeError, ValueError):
            raise cards.CardError("A gauge has to be a number from 0 to 100.") from None
        # Written as a percentage by anyone sane, stored as a fraction.
        level = min(max(level / 100.0 if level > 1.0 else level, 0.0), 1.0)
    return cards.board().open_note(title, body, level)


def show_file(title: str, path: str) -> str:
    """Put a file on the screen and keep it there as it changes."""
    if not str(title).strip():
        raise cards.CardError("A card needs a title -- it is how I update it later.")
    if not str(path).strip():
        raise cards.CardError("Tell me which file to show.")
    return cards.board().open_file(title, path)


def close_card(name: str) -> str:
    """Take one card off the screen, or "all" of them."""
    return cards.board().close(name)


def list_cards() -> str:
    """What is on the screen right now."""
    return cards.board().listing()


SCREEN = {
    "show_monitor": (
        show_monitor,
        "Put a live gauge on the user's screen that keeps watching on its "
        "own: cpu, memory, network, battery, disk or processes. Use this "
        "whenever the user asks you to watch or keep an eye on something the "
        "machine measures.",
        {"kind": ("string", "cpu, memory, network, battery, disk or processes",
                  True)}),
    "show_card": (
        show_card,
        "Put a card of your own text on the user's screen and leave it there "
        "-- a list, a countdown, a set of steps, anything worth keeping in "
        "front of them. Calling it again with the same title replaces the "
        "contents, which is how you update one.",
        {"title": ("string", "A short name. Reusing one updates that card", True),
         "body": ("string", "The text. One line per line", True),
         "gauge": ("number", "0 to 100, to draw a bar across the top", False)}),
    "show_file": (
        show_file,
        "Put a file on the user's screen -- a text or markdown file, or a "
        "picture -- and keep it there. The card follows the file, so writing "
        "to that file with create_file changes what is on screen without "
        "calling this again. That is how to build a widget you keep updating: "
        "show_file once, then rewrite the file whenever there is news.",
        {"title": ("string", "A short name for the card", True),
         "path": ("string", "The file to show, e.g. ~/Documents/ping.md", True)}),
    "close_card": (
        close_card,
        "Take a card or monitor off the user's screen. Pass 'all' to clear "
        "the screen.",
        {"name": ("string", "The card's title, or 'all'", True)}),
    "list_cards": (
        list_cards, "List what is currently on the user's screen.", {}),
}
