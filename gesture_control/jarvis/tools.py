"""The MCP server FreeClaw calls to do things on *this* machine.

FreeClaw already has the web, memory, bash and files -- but its files are
sandboxed to its own static folder, and if FreeClaw is on another machine then
"its own" is the wrong machine entirely.  This server is the way back: it runs
inside the overlay app, and everything it does happens here.

**Why HTTP and not stdio.**  The usual way to give an agent local tools is a
stdio server it spawns as a child process.  That would tie Jarvis to a
FreeClaw on this machine, because a child process runs where its parent does.
Serving the same tools over HTTP inverts the direction -- FreeClaw dials in --
so pointing the setup dialog at `192.168.1.40` still means folders get made
*here*.  It also means the tools can reach the running overlay, which a
separate process could not do without a second bridge.

The protocol is streamable HTTP JSON-RPC 2.0 (MCP 2025-06-18): one POST per
request, a plain JSON body back.  Only the three methods FreeClaw actually
calls are implemented -- `initialize`, `tools/list`, `tools/call` -- plus the
`notifications/initialized` acknowledgement, which takes no reply.

**This is a real hole in the sandbox, and it is meant to be.**  Anyone who can
reach the port and holds the token can write files as this user.  Hence: bound
to one interface rather than all of them, a 32-byte bearer token compared in
constant time, and every path resolved and checked against a root before it is
touched.  `ROOTS` is the whole of the protection -- widen it and you have
widened what a prompt injection in a web page can reach.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "jarvis", "version": "1.0.0"}

def _known_folder(guid: str, fallback: str) -> pathlib.Path | None:
    """A Windows known folder, by its GUID.

    Asking Windows rather than assuming `~/Documents`: OneDrive's "back up
    your folders" moves Desktop, Documents and Pictures into the OneDrive
    directory and leaves the old paths either missing or empty.  Guessing
    would give Jarvis a sandbox around folders the user does not use, and the
    failure would look like a permissions bug rather than a wrong path.
    """
    try:
        import ctypes
        from ctypes import wintypes

        shell = ctypes.WinDLL("shell32", use_last_error=True)
        shell.SHGetKnownFolderPath.argtypes = [
            ctypes.c_char_p, wintypes.DWORD, wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_wchar_p)]
        out = ctypes.c_wchar_p()
        # The GUID goes over as its raw 16 bytes, little-endian in the first
        # three fields -- the layout of a Windows GUID struct, not the order
        # the string is written in.
        parts = guid.split("-")
        raw = (bytes.fromhex(parts[0])[::-1] + bytes.fromhex(parts[1])[::-1]
               + bytes.fromhex(parts[2])[::-1] + bytes.fromhex(parts[3])
               + bytes.fromhex(parts[4]))
        if shell.SHGetKnownFolderPath(raw, 0, None, ctypes.byref(out)) == 0:
            return pathlib.Path(out.value).resolve()
    except (OSError, ValueError, AttributeError):
        pass
    guess = pathlib.Path(fallback).expanduser()
    return guess.resolve() if guess.is_dir() else None


# The only places the tools may touch.  A prompt injection in a scraped web
# page reaches exactly as far as this list does, so it is deliberately the
# user's own documents and desktop rather than the whole drive: nothing in
# here can overwrite a system file, another user's profile, or this app.
ROOTS = tuple(dict.fromkeys(p for p in (
    _known_folder("B4BFCC3A-DB2C-424C-B029-7FE99A87C641", "~/Desktop"),
    _known_folder("FDD39AD0-238F-46AF-ADB4-6C85480369C7", "~/Documents"),
    _known_folder("374DE290-123F-4565-9164-39C4925E467B", "~/Downloads"),
    pathlib.Path(os.environ["TEMP"]).resolve() if os.environ.get("TEMP") else None,
) if p is not None and p.is_dir()))

# A single file write, capped so a runaway generation cannot fill the disk.
MAX_WRITE = 4 * 1024 * 1024


class ToolError(Exception):
    """A failure the model should be told about in words it can act on."""


def _resolve(path: str) -> pathlib.Path:
    """A user-supplied path, resolved and confined to ROOTS.

    Resolution happens before the check, so `~/Documents/../../Windows` is
    compared as the Windows directory it actually means rather than as the
    string it was written as.
    """
    if not path or not str(path).strip():
        raise ToolError("A path is required.")
    target = pathlib.Path(os.path.expandvars(os.path.expanduser(str(path))))
    try:
        target = target.resolve()
    except OSError as exc:
        raise ToolError(f"That path cannot be resolved: {exc}") from exc
    for root in ROOTS:
        if target == root or root in target.parents:
            return target
    allowed = ", ".join(str(r) for r in ROOTS)
    raise ToolError(f"{target} is outside the folders Jarvis may touch. "
                    f"Allowed: {allowed}")


# -- the tools -------------------------------------------------------------

def create_folder(path: str) -> str:
    target = _resolve(path)
    if target.is_file():
        raise ToolError(f"{target} is already a file.")
    target.mkdir(parents=True, exist_ok=True)
    return f"Created {target}"


def create_file(path: str, content: str = "", overwrite: bool = False) -> str:
    target = _resolve(path)
    if target.is_dir():
        raise ToolError(f"{target} is a folder, not a file.")
    if target.exists() and not overwrite:
        raise ToolError(f"{target} already exists. Pass overwrite=true to replace it.")
    content = content or ""
    if len(content.encode("utf-8")) > MAX_WRITE:
        raise ToolError(f"That is larger than the {MAX_WRITE // 1024 // 1024}MB limit.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="")
    return f"Wrote {len(content)} characters to {target}"


def read_file(path: str, max_chars: int = 20000) -> str:
    target = _resolve(path)
    if not target.is_file():
        raise ToolError(f"{target} is not a file.")
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError(f"Could not read {target}: {exc}") from exc
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [{len(text) - max_chars} more characters]"
    return text


def list_folder(path: str) -> str:
    target = _resolve(path)
    if not target.is_dir():
        raise ToolError(f"{target} is not a folder.")
    rows = []
    for entry in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        try:
            size = f"{entry.stat().st_size:,} bytes" if entry.is_file() else "folder"
        except OSError:
            size = "?"
        rows.append(f"  {entry.name}  ({size})")
    return f"{target}\n" + ("\n".join(rows) if rows else "  (empty)")


def move_path(source: str, destination: str) -> str:
    src, dst = _resolve(source), _resolve(destination)
    if not src.exists():
        raise ToolError(f"{src} does not exist.")
    if dst.exists():
        raise ToolError(f"{dst} already exists.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"Moved {src} to {dst}"


def delete_path(path: str, recursive: bool = False) -> str:
    target = _resolve(path)
    if not target.exists():
        raise ToolError(f"{target} does not exist.")
    if target in ROOTS:
        raise ToolError(f"{target} is one of Jarvis's root folders and cannot be deleted.")
    if target.is_dir():
        if not recursive and any(target.iterdir()):
            raise ToolError(f"{target} is not empty. Pass recursive=true to delete it anyway.")
        shutil.rmtree(target)
    else:
        target.unlink()
    return f"Deleted {target}"


def open_path(path: str) -> str:
    """Open a file or folder in whatever Windows uses for it."""
    target = _resolve(path)
    if not target.exists():
        raise ToolError(f"{target} does not exist.")
    try:
        os.startfile(str(target))                      # noqa: S606 (Windows only)
    except (AttributeError, OSError) as exc:
        raise ToolError(f"Could not open {target}: {exc}") from exc
    return f"Opened {target}"


def run_program(program: str, arguments: str = "", timeout: float = 20.0) -> str:
    """Run a command and return what it printed.

    It waits and captures, where an earlier version launched detached and
    returned nothing. That version was worse than useless: a model asked for
    the memory figure would run a command redirecting to a file, get back
    "Started cmd", read the file, find nothing, and try again -- because
    nothing it could do would ever produce an answer. A tool that cannot
    report its result is a tool that will be called in a loop.

    Still not a shell. `shell=True` would make every quoting decision the
    model makes into a command-injection question; arguments are split the way
    a shell would split them and handed over as a list, so a quoted path with
    spaces survives and a `&& rm -rf` does not become a second command.
    """
    if not program or not program.strip():
        raise ToolError("A program is required.")
    try:
        args = [program.strip()] + shlex.split(arguments or "", posix=False)
    except ValueError as exc:
        raise ToolError(f"Could not read those arguments: {exc}") from exc
    try:
        done = subprocess.run(args, shell=False, capture_output=True, text=True,
                              errors="replace", timeout=max(1.0, float(timeout)))
    except FileNotFoundError:
        raise ToolError(f"No program called '{program}' was found. "
                        f"To start an application, use open_app.") from None
    except subprocess.TimeoutExpired:
        raise ToolError(f"{program} was still running after {timeout:.0f}s "
                        f"and was stopped.") from None
    except OSError as exc:
        raise ToolError(f"Could not run {program}: {exc}") from exc
    output = (done.stdout or "") + (done.stderr or "")
    output = output.strip() or "(no output)"
    if len(output) > 8000:
        output = output[:8000] + "\n... [truncated]"
    return f"exit {done.returncode}\n{output}"


# name -> (function, description, {argument: (type, description, required)})
TOOLS = {
    "create_folder": (
        create_folder,
        "Create a folder on the user's computer, making parent folders as needed.",
        {"path": ("string", "Where to create it, e.g. ~/Desktop/Notes", True)}),
    "create_file": (
        create_file,
        "Write a file on the user's computer, making parent folders as needed.",
        {"path": ("string", "Where to write it", True),
         "content": ("string", "What to put in it", False),
         "overwrite": ("boolean", "Replace the file if it already exists", False)}),
    "read_file": (
        read_file,
        "Read a text file from the user's computer.",
        {"path": ("string", "The file to read", True),
         "max_chars": ("integer", "Stop after this many characters", False)}),
    "list_folder": (
        list_folder,
        "List what is in a folder on the user's computer.",
        {"path": ("string", "The folder to list", True)}),
    "move_path": (
        move_path,
        "Move or rename a file or folder on the user's computer.",
        {"source": ("string", "What to move", True),
         "destination": ("string", "Where to move it to", True)}),
    "delete_path": (
        delete_path,
        "Delete a file or folder on the user's computer. Permanent.",
        {"path": ("string", "What to delete", True),
         "recursive": ("boolean", "Delete a folder that still has things in it", False)}),
    "open_path": (
        open_path,
        "Open a file or folder in the application Windows associates with it.",
        {"path": ("string", "What to open", True)}),
    "run_program": (
        run_program,
        "Run a command-line program on the user's computer and return what it "
        "printed. To start an application with a window, use open_app instead.",
        {"program": ("string", "The program to run", True),
         "arguments": ("string", "Arguments to pass it", False),
         "timeout": ("number", "Seconds to wait before giving up", False)}),
}


from . import actions as _actions                          # noqa: E402
from . import cards as _cards                              # noqa: E402
from . import status as _status                            # noqa: E402
from . import screen as _screen                            # noqa: E402
from . import watch as _watch                              # noqa: E402

# The file tools, everything that acts (actions.py) and everything that only
# looks (status.py). Keyboard input is added only when jarvis.json allows it:
# unlike every other tool here it is not bounded by anything, and an agent
# that reads web pages can be told what to type.
TOOLS.update(_actions.ACTIONS)
TOOLS.update(_status.STATUS)
TOOLS.update(_screen.SCREEN)
TOOLS.update(_watch.WATCHES)


def allow_keyboard(allowed: bool) -> None:
    """Add or remove the two tools that can type anywhere."""
    for name, entry in _actions.KEYBOARD_ACTIONS.items():
        if allowed:
            TOOLS[name] = entry
        else:
            TOOLS.pop(name, None)


def schema() -> list[dict]:
    """TOOLS in the shape `tools/list` has to answer with."""
    out = []
    for name, (_, description, args) in TOOLS.items():
        properties = {a: {"type": t, "description": d} for a, (t, d, _) in args.items()}
        required = [a for a, (_, _, req) in args.items() if req]
        out.append({"name": name, "description": description,
                    "inputSchema": {"type": "object", "properties": properties,
                                    "required": required}})
    return out


def call(name: str, arguments: dict):
    entry = TOOLS.get(name)
    if entry is None:
        raise ToolError(f"There is no tool called '{name}'.")
    function, _, spec = entry
    unknown = set(arguments or {}) - set(spec)
    if unknown:
        raise ToolError(f"{name} has no argument {', '.join(sorted(unknown))}.")
    try:
        return function(**(arguments or {}))
    except (ToolError, _actions.ActionError, _status.StatusError,
            _cards.CardError, _watch.WatchError) as exc:
        raise ToolError(str(exc)) from exc
    except TypeError as exc:
        raise ToolError(f"{name}: {exc}") from exc
    except Exception as exc:                            # noqa: BLE001
        raise ToolError(f"{name} failed: {exc}") from exc


# -- the server ------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_version = "JarvisMCP/1.0"
    token = ""

    def log_message(self, fmt, *args):
        """Silence. The default writes every request to stderr, which in this
        app is the terminal the user is reading status out of."""

    def _send(self, code: int, payload: dict | None) -> None:
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        self.send_response(code)
        if body:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _authorised(self) -> bool:
        offered = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not offered.startswith(prefix):
            return False
        # Constant time: a plain == leaks the token one byte at a time to
        # anyone who can measure the reply.
        return hmac.compare_digest(offered[len(prefix):], self.token)

    def do_POST(self):                                   # noqa: N802
        if not self._authorised():
            self._send(401, {"error": "Unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            message = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            self._send(400, {"jsonrpc": "2.0", "id": None,
                             "error": {"code": -32700, "message": "Parse error"}})
            return
        reply = dispatch(message)
        # A notification has no id and takes no reply; the spec wants 202.
        self._send(200 if reply else 202, reply)

    def do_GET(self):                                    # noqa: N802
        # FreeClaw only POSTs, but a GET is how a person checks the port is
        # alive, so answer something honest rather than a stack trace.
        self._send(200 if self._authorised() else 401,
                   {"server": SERVER_INFO, "tools": len(TOOLS)})


def dispatch(message: dict) -> dict | None:
    """One JSON-RPC message in, its reply out (or None for a notification)."""
    request_id = message.get("id")
    method = message.get("method")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": request_id,
                "result": {"protocolVersion": PROTOCOL_VERSION,
                           "capabilities": {"tools": {}},
                           "serverInfo": SERVER_INFO}}
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": schema()}}
    if method == "tools/call":
        params = message.get("params") or {}
        try:
            text = call(params.get("name"), params.get("arguments") or {})
        except ToolError as exc:
            # A tool that failed is a result the model should read and work
            # around, not a protocol error -- isError is how MCP says that.
            return {"jsonrpc": "2.0", "id": request_id,
                    "result": {"content": [{"type": "text", "text": str(exc)}],
                               "isError": True}}
        # A tool may answer with a picture as well as words -- a screenshot
        # returns (caption, png). MCP carries that as a second content block,
        # and FreeClaw passes it to the model where the provider accepts one.
        image = None
        if isinstance(text, tuple):
            text, image = text
        content = [{"type": "text", "text": text}]
        if image:
            content.append({"type": "image", "mimeType": "image/png",
                            "data": base64.b64encode(image).decode("ascii")})
        return {"jsonrpc": "2.0", "id": request_id,
                "result": {"content": content, "isError": False}}
    if request_id is None:
        return None
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"}}


class ToolServer:
    """The MCP server, on a background thread for the life of the app."""

    def __init__(self, token: str, port: int, host: str = "0.0.0.0") -> None:
        self.port = port
        self.host = host
        self.token = token
        self.error: str | None = None
        self._server: ThreadingHTTPServer | None = None
        # An empty token would authenticate an empty header -- compare_digest
        # is perfectly happy comparing "" to "" -- which would leave these
        # tools open to anything that can reach the port. Refuse instead.
        if not token or len(token) < 16:
            self.error = ("Jarvis's tool server needs a token of its own before "
                          "it will listen. Re-run Add FreeClaw.")
            return
        handler = type("_Bound", (_Handler,), {"token": token})
        # A taken port has to be an error, not a coin flip. HTTPServer sets
        # allow_reuse_address, which on Windows lets a second process bind a
        # port another one is already listening on -- and then Windows decides
        # per connection which of them gets it. That is how a second copy of
        # this app, or a test run beside a live one, ends up answering half of
        # FreeClaw's calls with a different set of tools.
        server = type("_Exclusive", (ThreadingHTTPServer,),
                      {"allow_reuse_address": False})
        try:
            self._server = server((host, port), handler)
        except OSError as exc:
            self.error = (f"Could not open port {port} for Jarvis's tools: "
                          f"{exc}. Another copy of this app is probably "
                          f"already running.")
            return
        self.port = self._server.server_address[1]      # port 0 asks for any
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="jarvis-tools", daemon=True)
        self._thread.start()

    @property
    def running(self) -> bool:
        return self._server is not None

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
