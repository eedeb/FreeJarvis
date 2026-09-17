""""Add FreeClaw": everything that happens after the address and password.

One call, `link()`, run on a worker thread so the dialog can narrate it.  It
is idempotent -- running it again is how you push an edited persona, move to a
different FreeClaw, or repair a registration -- and every step is something
FreeClaw already exposes over HTTP, so none of it requires FreeClaw to be on
this machine.

The order matters in one place.  The MCP server has to be *listening here*
before FreeClaw is told about it, because FreeClaw dials it immediately to
count its tools and reports the registration as failed if nobody answers.
"""

from __future__ import annotations

import hashlib
import socket
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from . import client, settings as store
from .tools import ToolServer

MCP_NAME = "jarvis"
PERSONA = Path(__file__).resolve().parent / "persona.md"


def _fingerprint() -> str:
    """A URL suffix that is different every time. See link().

    It carries the tool names so the URL says something, but it also carries
    the clock, so re-linking always means re-reading. Keying on the tools
    alone was not enough: FreeClaw had already cached a *wrong* answer for
    that key -- an older copy of this app, still holding the port, had
    answered the probe -- and no amount of re-registering the same tools
    would shift it. Re-linking is someone explicitly asking for the two ends
    to be synchronised, so it should never be a no-op.
    """
    from .tools import TOOLS

    stamp = f"{','.join(sorted(TOOLS))}|{time.time()}"
    return hashlib.sha1(stamp.encode()).hexdigest()[:10]


class LinkError(Exception):
    """Setup stopped, with a sentence explaining what to do about it."""


def reachable_address(freeclaw_url: str, port: int) -> str:
    """The address FreeClaw should dial to reach this machine's tools.

    Not a fixed guess.  A FreeClaw on this machine can use loopback, and
    should -- it is the one address that cannot be reached from the network.
    A FreeClaw somewhere else needs whichever of this machine's interfaces
    faces it, which is found by asking the routing table which local address a
    packet to *that host* would leave from.  No packet is actually sent; a UDP
    connect only sets the socket's peer.
    """
    host = freeclaw_url.split("://", 1)[-1].split("/", 1)[0].rsplit(":", 1)[0]
    if host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return f"http://127.0.0.1:{port}"
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((host, 1))
        local = probe.getsockname()[0]
    except OSError:
        local = socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()
    return f"http://{local}:{port}"


def link(url: str, password: str, user: str | None = None,
         server: ToolServer | None = None,
         progress: Callable[[str], None] | None = None) -> Iterator[str]:
    """Wire this app to the FreeClaw at `url`. Yields a line per step.

    `server` is the already-listening tool server. Passing None starts one
    for the duration of the check and leaves it running -- which is what the
    setup dialog does before the app's own server exists.
    """
    def say(line: str) -> str:
        if progress is not None:
            progress(line)
        return line

    saved = store.load()
    user = user or saved.user or store.DEFAULT_USER
    url = client.normalise(url)
    if not url:
        raise LinkError("An address is needed, e.g. 192.168.1.40 or localhost.")
    if not password:
        raise LinkError("A password is needed -- the one FreeClaw's web UI asks for.")

    # 1 -- reach it. Everything else needs the cookie this earns.
    yield say(f"Connecting to {url} ...")
    try:
        session = client.admin(url, password)
    except client.FreeClawError as exc:
        raise LinkError(str(exc)) from exc
    yield say("Connected.")

    # 2 -- the user, created through the API so it gets the whole set-up a
    #      FreeClaw user is meant to have: context.md, ping.md, a conversation.
    if user in client.users(session, url):
        yield say(f"User '{user}' is already there.")
    else:
        client.create_user(session, url, user)
        yield say(f"Created the '{user}' user.")

    # 3 -- the persona. Over HTTP, which also applies it to the conversation
    #      already running rather than waiting for the next reset.
    try:
        persona = PERSONA.read_text(encoding="utf-8")
    except OSError as exc:
        raise LinkError(f"The persona file is missing: {PERSONA}") from exc
    client.set_context(session, url, user, persona)
    yield say("Wrote the Jarvis persona.")

    # 4 -- our own tools, listening before FreeClaw is told where to find them.
    #      A server handed in brings its own token and port: telling FreeClaw
    #      about a *different* token than the one actually being served is a
    #      failure that only shows up at the first tool call, so the running
    #      server is taken as the truth rather than the saved settings.
    if server is None:
        server = ToolServer(store.token(), saved.tool_port or store.DEFAULT_TOOL_PORT)
    if not server.running:
        raise LinkError(server.error or "Could not start Jarvis's tool server.")
    token, port = server.token, server.port
    # FreeClaw caches what a server offers, keyed on (transport, url, token).
    # Re-registering changes none of those, so a FreeClaw that has already
    # talked to us keeps serving its old idea of our tool list for the life of
    # its process -- which showed up as a model that had just been given
    # sixteen new tools stubbornly using the eight it remembered. Naming the
    # tool set in the URL changes the key exactly when the answer would change.
    tool_url = f"{reachable_address(url, port)}/?tools={_fingerprint()}"
    yield say(f"Serving Jarvis's tools on {tool_url}")

    # 5 -- hand it to FreeClaw. Replacing rather than skipping an existing
    #      entry: the address or the token may have changed since last time,
    #      and a stale one fails at the first tool call rather than here.
    if any(s.get("name") == MCP_NAME for s in client.mcp_servers(session, url)):
        client.remove_mcp(session, url, MCP_NAME)
    try:
        count, warning = client.add_mcp(session, url, MCP_NAME, tool_url, token)
    except client.FreeClawError as exc:
        raise LinkError(
            f"{exc}\n\nFreeClaw could not reach this machine at {tool_url}. "
            f"If FreeClaw is on another computer, port {port} has to be open "
            f"through this one's firewall.") from exc
    yield say(f"Registered the tool server with FreeClaw -- {count} tools."
              + (f" ({warning})" if warning else ""))

    # 6 -- the API this app actually asks questions through
    if client.api_enabled(session, url):
        yield say("FreeClaw's API is already on.")
    else:
        client.enable_api(session, url)
        yield say("Switched FreeClaw's API on.")

    # 7 -- the one remaining thing that makes a silent Jarvis
    enabled = [p for p in client.providers(session, url) if p.get("enabled", True)]
    if enabled:
        yield say(f"Providers: {', '.join(p.get('name', '?') for p in enabled)}")
    else:
        yield say("! FreeClaw has no LLM provider enabled, so it cannot answer "
                  "anything yet. Add one under Settings -> Providers.")

    store.update(url=url, password=password, user=user, tool_url=tool_url,
                 tool_port=port, tool_token=token, linked=True)
    yield say("Ready. Say \"hey Jarvis\".")


def unlink() -> None:
    """Forget the connection. Leaves the FreeClaw user and its memory alone."""
    store.update(linked=False, password="")
