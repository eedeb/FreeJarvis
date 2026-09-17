"""Jarvis: the FreeClaw link, the tools it exposes back, and the reactor.

FreeClaw itself is stubbed -- but stubbed from its own handlers, not invented:
a wrong password answering 200 with no cookie, /api/mcp dialling the tool
server to count its tools before saving, 409 on a duplicate user. Those are
the behaviours link() reads, so a stub that got them wrong would prove
nothing. The tool server under test is the real one.
"""
import sys, pathlib, json, threading, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ctypes
from ctypes import wintypes

import numpy as np
import requests

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetTopWindow.restype = wintypes.HWND
user32.GetTopWindow.argtypes = [wintypes.HWND]
user32.GetWindow.restype = wintypes.HWND
user32.GetWindow.argtypes = [wintypes.HWND, ctypes.c_uint]
user32.FindWindowW.restype = wintypes.HWND
GWL_EXSTYLE, GW_HWNDNEXT = -20, 2
WS_EX_TOPMOST, WS_EX_NOACTIVATE = 0x00000008, 0x08000000

from gesture_control.jarvis import client, link, settings as store, tools
import cv2
from gesture_control.jarvis import terminal as jterm
from gesture_control.jarvis import reactor as rmod
from gesture_control.jarvis.reactor import MOOD, STATES, Reactor

ok = True


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


def _time_orb(orb, canvas):
    """One frame of the reactor, in milliseconds."""
    start = time.perf_counter()
    orb.draw(canvas, 1 / 30, top=31)
    return (time.perf_counter() - start) * 1000.0


def _refused(name, args, wanted):
    """Whether a tool call is refused with a reason that mentions `wanted`."""
    try:
        tools.call(name, args)
        return False
    except Exception as exc:                                     # noqa: BLE001
        return wanted.lower() in str(exc).lower()


# Keep the real jarvis.json out of the way -- this test writes settings.
store.JARVIS_PATH = pathlib.Path(tempfile.mkdtemp()) / "jarvis.json"

PASSWORD = "hunter2"
STATE = {"users": ["Default"], "context": {}, "mcp": [], "api": False,
         "providers": [{"name": "Groq", "enabled": True}]}


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            return json.loads(raw or b"{}")
        except ValueError:
            return {"_form": raw.decode()}

    def _in(self):
        return "session=yes" in (self.headers.get("Cookie") or "")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/login":
            good = f"password={PASSWORD}" in self._read().get("_form", "")
            self.send_response(302 if good else 200)
            if good:
                self.send_header("Set-Cookie", "session=yes; Path=/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if not self._in():
            return self._json(401, {"error": "Unauthorized"})
        if path == "/api/users":
            name = self._read().get("name")
            if name in STATE["users"]:
                return self._json(409, {"error": "exists"})
            STATE["users"].append(name)
            return self._json(200, {"ok": True})
        if path == "/api/mcp":
            data = self._read()
            if any(s["name"] == data.get("name") for s in STATE["mcp"]):
                return self._json(409, {"error": "exists"})
            try:            # FreeClaw dials the server before saving it
                r = requests.post(data["url"], timeout=10, json={
                    "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                    headers={"Authorization": f"Bearer {data.get('token', '')}"})
                found = r.json()["result"]["tools"]
            except Exception as e:
                return self._json(200, {"tool_count": 0, "warning": f"unreachable: {e}"})
            STATE["mcp"].append(dict(data))
            return self._json(200, {"tool_count": len(found)})
        if path == "/api/api-status":
            STATE["api"] = bool(self._read().get("enabled"))
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "no route"})

    def do_PUT(self):
        if not self._in():
            return self._json(401, {"error": "Unauthorized"})
        parts = self.path.strip("/").split("/")
        if len(parts) == 4 and parts[3] == "context":
            if parts[2] not in STATE["users"]:
                return self._json(404, {"error": "No such user"})
            STATE["context"][parts[2]] = self._read().get("context")
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "no route"})

    def do_GET(self):
        path = self.path.split("?")[0]
        if not self._in():
            return self._json(401, {"error": "Unauthorized"})
        if path == "/api/users":
            return self._json(200, {"users": [{"name": u} for u in STATE["users"]]})
        if path == "/api/mcp":
            return self._json(200, {"servers": STATE["mcp"]})
        if path == "/api/api-status":
            return self._json(200, {"enabled": STATE["api"]})
        if path == "/api/providers":
            return self._json(200, {"providers": STATE["providers"]})
        return self._json(404, {"error": "no route"})

    def do_DELETE(self):
        if not self._in():
            return self._json(401, {"error": "Unauthorized"})
        name = self.path.rsplit("/", 1)[-1]
        STATE["mcp"][:] = [s for s in STATE["mcp"] if s["name"] != name]
        return self._json(200, {"ok": True})


stub = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
threading.Thread(target=stub.serve_forever, daemon=True).start()
FC = f"127.0.0.1:{stub.server_address[1]}"

print("An address someone typed becomes a URL:")
for typed, want in (("192.168.1.40", "http://192.168.1.40:6767"),
                    ("localhost", "http://localhost:6767"),
                    ("http://10.0.0.5:6767", "http://10.0.0.5:6767"),
                    ("1.2.3.4:9000", "http://1.2.3.4:9000")):
    got = client.normalise(typed)
    check(f"{typed} -> {got}", got == want)

print("\nThe tool server, which is what FreeClaw calls back into:")
TOKEN = "a-token-long-enough-to-be-accepted"
srv = tools.ToolServer(TOKEN, 0, host="127.0.0.1")
check("it started", srv.running)
BASE = f"http://127.0.0.1:{srv.port}"


def rpc(method, params=None, token=TOKEN):
    r = requests.post(BASE, timeout=10,
                      headers={"Authorization": f"Bearer {token}"},
                      json={"jsonrpc": "2.0", "id": 1, "method": method,
                            "params": params or {}})
    return r.status_code, (r.json() if r.content else None)


check("an unauthenticated call is refused", rpc("tools/list", token="wrong")[0] == 401)
code, msg = rpc("initialize", {"protocolVersion": tools.PROTOCOL_VERSION})
check("it speaks the protocol version FreeClaw sends",
      msg["result"]["protocolVersion"] == tools.PROTOCOL_VERSION)
code, msg = rpc("tools/list")
names = [t["name"] for t in msg["result"]["tools"]]
check(f"it offers the file and folder tools ({len(names)})",
      {"create_file", "create_folder", "read_file", "delete_path"} <= set(names))
check("every tool declares its required arguments",
      all("required" in t["inputSchema"] for t in msg["result"]["tools"]))

print("\nThe sandbox is the whole of the protection, so:")
check("the roots are real folders on this machine",
      tools.ROOTS and all(r.is_dir() for r in tools.ROOTS))
sandbox = pathlib.Path(tempfile.mkdtemp(dir=str(tools.ROOTS[-1])))
inside = sandbox / "deep" / "note.txt"
code, msg = rpc("tools/call", {"name": "create_file",
                               "arguments": {"path": str(inside), "content": "hello"}})
check("a file inside a root is written, parents and all",
      not msg["result"]["isError"] and inside.is_file())
code, msg = rpc("tools/call", {"name": "read_file", "arguments": {"path": str(inside)}})
check("and read back", msg["result"]["content"][0]["text"] == "hello")
code, msg = rpc("tools/call", {"name": "create_file",
                               "arguments": {"path": str(inside), "content": "x"}})
check("overwriting takes an explicit flag", msg["result"]["isError"])

# The escapes worth having a test for: an absolute path somewhere else, and
# a relative one that only *resolves* to somewhere else. Checking the string
# instead of the resolved path would let the second one straight through.
for bad in ("C:/Windows/System32/drivers/etc/hosts",
            str(sandbox / ".." / ".." / ".." / ".." / "Windows" / "x.txt"),
            "~/../Public/x.txt"):
    code, msg = rpc("tools/call", {"name": "create_file",
                                   "arguments": {"path": bad, "content": "x"}})
    check(f"refused: {bad[:48]}", msg["result"]["isError"])

code, msg = rpc("tools/call", {"name": "delete_path",
                               "arguments": {"path": str(tools.ROOTS[0])}})
check("a root folder itself cannot be deleted", msg["result"]["isError"])
code, msg = rpc("tools/call", {"name": "no_such_tool", "arguments": {}})
check("an unknown tool is an error the model can read, not a crash",
      msg["result"]["isError"] and "no tool" in msg["result"]["content"][0]["text"])
# A failed tool is a *result*, not a JSON-RPC error: the model has to see the
# text to work around it, and a protocol error would never reach it.
check("a failing tool still answers with a result, not an rpc error",
      "result" in msg and "error" not in msg)

print("\nDoing things to this machine that are not files:")
from gesture_control.jarvis import actions as jact

# Finding an app by the name a person calls it. Nothing here launches
# anything: starting a real application mid-test is not a test, it is a
# side effect on the machine someone is using.
links = jact._shortcuts()
check(f"it indexes the shortcuts on this machine ({len(links)})", len(links) > 5)
check("from the Start Menu and every desktop",
      any("Start Menu" in str(v) for v in links.values()))
some = sorted(links)[0]
check(f"an exact name resolves ({some})", jact._match(some)[0] == some)
check("a partial one does too",
      jact._match(some[: max(3, len(some) // 2)]) is not None)
check("and something that is not installed does not",
      jact._match("no-such-application-anywhere-xyz") is None)
# An exact match must never lose to a longer name that merely contains it.
saved = dict(links)
try:
    links.clear()
    links.update({"code": pathlib.Path("a.lnk"),
                  "code insiders": pathlib.Path("b.lnk"),
                  "vscode": pathlib.Path("c.lnk")})
    check("an exact name beats a longer one containing it",
          jact._match("code")[0] == "code")
finally:
    links.clear()
    links.update(saved)

print("\n  Windows:")
rows = jact.list_windows()
check(f"it can list what is open ({len(rows.splitlines())} rows)",
      "Nothing is open." not in rows)
try:
    jact.focus_window("no such window anywhere at all")
    check("a missing window is an error", False)
except jact.ActionError as exc:
    check("a missing window says so, and says how to look",
          "list_windows" in str(exc))
try:
    jact.window_state("anything", "sideways")
    check("an unknown window state is refused", False)
except jact.ActionError as exc:
    check(f"an unknown window state is refused", "minimise" in str(exc))

print("\n  Sound:")
try:
    before = jact._volume().get()
    jact.set_volume(31)
    check(f"set_volume moves the real volume ({jact.get_volume()})",
          abs(jact._volume().get() - 0.31) < 0.02)
    jact.set_volume(before * 100)
    check(f"and the test puts it back ({jact.get_volume()})",
          abs(jact._volume().get() - before) < 0.02)
    try:
        jact.set_volume("loud")
        check("a non-numeric volume is refused", False)
    except jact.ActionError:
        check("a non-numeric volume is refused", True)
    try:
        jact.media_key("teleport")
        check("an unknown media key is refused", False)
    except jact.ActionError as exc:
        check("an unknown media key is refused, and lists the real ones",
              "playpause" in str(exc))
except jact.ActionError as exc:
    check(f"sound is reachable ({exc})", False)

print("\n  Clipboard and the machine:")
keep = jact.clipboard_get(4000)
jact.clipboard_set("jarvis test string")
check("what goes on the clipboard comes back",
      jact.clipboard_get(50) == "jarvis test string")
if keep and "empty" not in keep:
    jact.clipboard_set(keep)
info = jact.system_info()
check(f"system_info reports the machine ({info.splitlines()[0]})",
      "CPU" in info and "Memory" in info)

caption, png = jact.take_screenshot()
check(f"a screenshot comes back as a picture ({len(png):,} bytes)",
      png[:4] == b"\x89PNG")
check("and says where it was saved", "saved to" in caption)

print("\n  The two that can type:")
# These are the only tools here that nothing bounds, so the thing worth
# testing is that they are optional and that they refuse nonsense.
tools.allow_keyboard(False)
check("they can be switched off entirely", "type_text" not in tools.TOOLS)
tools.allow_keyboard(True)
check("and back on", "type_text" in tools.TOOLS)
check("the setting exists and defaults to on", store.JarvisSettings().tools_keyboard)
for bad in ("", "nonsense+key", "ctrl+notakey"):
    try:
        jact.press_keys(bad)
        check(f"press_keys refuses {bad!r}", False)
    except jact.ActionError:
        check(f"press_keys refuses {bad!r}", True)
try:
    jact.type_text("")
    check("type_text refuses nothing to type", False)
except jact.ActionError:
    check("type_text refuses nothing to type", True)

print("\n  All of it reaches the MCP server:")
tools.allow_keyboard(True)
names = set(tools.TOOLS)
for expected in ("open_app", "list_windows", "focus_window", "set_volume",
                 "take_screenshot", "clipboard_get", "system_info",
                 "type_text", "press_keys"):
    check(f"  {expected} is offered", expected in names)
listed = {t["name"] for t in tools.schema()}
check(f"the schema matches the registry ({len(listed)} tools)", listed == names)
check("every tool describes itself",
      all(t.get("description") for t in tools.schema()))

print("\nLinking to FreeClaw:")
try:
    list(link.link(FC, "wrong-password"))
    check("a bad password is rejected", False)
except link.LinkError as exc:
    check(f"a bad password is rejected ({exc})", "rejected" in str(exc).lower())

store.update(tool_port=srv.port)
steps = list(link.link(FC, PASSWORD, server=srv))
check("it created the Jarvis user", "Jarvis" in STATE["users"])
check("it wrote a persona with the Jarvis instructions in it",
      "JARVIS" in STATE["context"].get("Jarvis", ""))
check("the persona tells it the tools reach the user's own machine",
      "jarvis` MCP tools" in STATE["context"].get("Jarvis", "")
      or "jarvis" in STATE["context"].get("Jarvis", "").lower())
check("it registered the tool server over http, not stdio",
      len(STATE["mcp"]) == 1 and STATE["mcp"][0]["transport"] == "http")
check("FreeClaw reached it and counted every tool",
      any(f"{len(names)} tools" in s for s in steps))
check("it switched FreeClaw's API on", STATE["api"] is True)
check("it saved the connection", store.load().ready())

before = len(STATE["mcp"]), len(STATE["users"])
list(link.link(FC, PASSWORD, server=srv))
check("running it again changes nothing",
      (len(STATE["mcp"]), len(STATE["users"])) == before)

print("\nThe address it gives FreeClaw to dial back on:")
check("loopback for a FreeClaw on this machine",
      link.reachable_address("http://127.0.0.1:6767", 8788)
      == "http://127.0.0.1:8788")
routed = link.reachable_address("http://192.168.1.40:6767", 8788)
check(f"a routable address for one elsewhere ({routed})",
      routed.startswith("http://") and "127.0.0.1" not in routed)

print("\nThe reactor:")
r = Reactor(160)
check("a state for each thing it can be doing", set(MOOD) == set(STATES))
frame = np.full((400, 700, 3), 40, np.uint8)
box = r.draw(frame, 1 / 30)
check(f"it draws in the top right {box}",
      box[0] > 700 * 0.6 and box[1] < 400 * 0.3)
check("it lit up the pixels it claimed",
      frame[box[1]:box[3], box[0]:box[2]].max() > 150)
untouched = frame.copy()
untouched[box[1]:box[3], box[0]:box[2]] = 40
check("and touched nothing else", int(untouched.min()) == 40 == int(untouched.max()))

# The overlay treats "differs from the bare camera frame" as "solid", so a
# reactor that did not change the pixels would be drawn see-through.
bare = np.full((400, 700, 3), 40, np.uint8)
check("it differs from the bare frame, which is what makes it opaque",
      int(np.abs(frame.astype(int) - bare.astype(int)).max()) > 50)

r2 = Reactor(160)
for _ in range(30):
    r2.draw(np.full((400, 700, 3), 40, np.uint8), 1 / 30)
check("it animates", abs(r2._phase - r.phase_after_one()) > 1e-6
      if hasattr(r, "phase_after_one") else r2._phase > 0.0)

# Against a yardstick rather than a clock, for the same reason test_overlay
# does it: absolute timings on this machine move by 2-3x between runs. One
# blur of the orb's own canvas is the unit -- the renderer is a fixed handful
# of whole-canvas passes, so a regression means more of them.
def timed(fn, n=40):
    """The fastest of `n` runs, not the average of them.

    An average measures whatever else the machine is doing; the minimum
    measures what this code costs when it gets the machine to itself, which is
    the only part a test can hold it responsible for.
    """
    for _ in range(4):
        fn()
    best = float("inf")
    for _ in range(n):
        start = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - start) * 1000)
    return best


per = timed(lambda: r.draw(frame, 1 / 30))
unit = timed(lambda: cv2.GaussianBlur(r._canvas, (0, 0), r.size * 0.055))
check(f"drawing it stays a fixed handful of passes "
      f"({per:.2f} ms = {per / unit:.1f}x one blur of {unit:.2f} ms)",
      per / unit < 14.0)
check(f"and it fits many times over inside a camera frame ({per:.2f} ms)",
      per < 12.0)

print("\nIt is actually three-dimensional:")
# Perspective: the same point is magnified when it swings towards the camera
# and shrunk when it swings away. A 2D rotation cannot do this, and it is what
# makes the near side of the orb bulge as it turns.
probe = np.array([[0.6, 0.0, 0.8], [0.6, 0.0, -0.8]], np.float32)
scale = rmod.FOCAL / (rmod.FOCAL - probe[:, 2])
check(f"a point swung towards the camera projects wider than the same point "
      f"swung away ({scale[0]:.2f}x vs {scale[1]:.2f}x)", scale[0] > scale[1] * 1.4)

# Depth shading: the near hemisphere is lit and the far one sinks towards the
# shadow colour. Rendered through the real _shell, with a cloud of two points.
lone = Reactor(120)
near_only = np.zeros((120, 120, 3), np.float32)
far_only = np.zeros((120, 120, 3), np.float32)
one = np.ones(1, np.float32)
lone._shell(near_only, np.array([[0.0, 0.0, 1.0]], np.float32), one, 0.0, 60.0, 30.0, 1.0)
lone._shell(far_only, np.array([[0.0, 0.0, -1.0]], np.float32), one, 0.0, 60.0, 30.0, 1.0)
check(f"the near side is drawn brighter than the far side "
      f"({far_only.sum():.0f} -> {near_only.sum():.0f})",
      near_only.sum() > far_only.sum() * 2.0)
check("but the far side is still visible, so it reads as a shell not a disc",
      far_only.sum() > 0.0)

# Turning it changes the picture -- that is the parallax, and the point of it.
spinner = Reactor(150)
spinner.set_state("thinking")
first = np.zeros((150, 150, 3), np.uint8)
spinner.draw(first, 1 / 30)
later = np.zeros((150, 150, 3), np.uint8)
for _ in range(30):
    later = np.zeros((150, 150, 3), np.uint8)
    spinner.draw(later, 1 / 30)
moved = float(np.abs(first.astype(int) - later.astype(int)).mean())
check(f"a second of spin visibly moves the detail ({moved:.1f} levels/pixel)",
      moved > 1.5)
# ...while the ball itself stays put. A wobbling silhouette would mean the
# projection was drifting rather than the object turning.
def extent(img):
    lit = np.argwhere(img.max(axis=2) > 30)
    return 0 if not len(lit) else int(lit[:, 1].max() - lit[:, 1].min())


check(f"while the silhouette stays the same size "
      f"({extent(first)}px vs {extent(later)}px)",
      abs(extent(first) - extent(later)) <= 6)

srv.stop()
stub.shutdown()
import shutil
shutil.rmtree(sandbox, ignore_errors=True)

print("\nIt holds up over whatever is behind it:")
# The bug this catches: the overlay works out opacity by comparing the drawn
# frame against the bare camera, and the orb's backdrop works by *darkening*.
# Darkening a dim room barely changes anything, so the disc came out
# see-through over a bright wallpaper -- and because the difference then
# tracked the camera's own noise, what did show up was speckle. The backdrop
# now states its own alpha instead of earning it.
for room, label in ((14, "a dark room"), (128, "a bright room")):
    camera = np.full((300, 300, 3), room, np.uint8)
    floor = np.zeros((300, 300), np.uint8)
    orb = Reactor(220)
    shot = camera.copy()
    disc = orb.draw(shot, 1 / 30, solid=floor)
    cy, cx = (disc[1] + disc[3]) // 2, (disc[0] + disc[2]) // 2
    check(f"the orb's backdrop is fully opaque over {label} "
          f"(alpha {floor[cy, cx]})", floor[cy, cx] == 255)
# Somewhere in the falloff between the solid middle and the clear outside.
edge = int(floor[cy, min(299, cx + int(orb.size * 0.41))])
check(f"and the floor fades out rather than ending on a hard edge ({edge})",
      0 < edge < 255)
check("it claims nothing outside the orb", int(floor[4, 4]) == 0)


print("\nIt makes room for the button in the corner:")
pushed = Reactor(150)
low = pushed.draw(np.zeros((648, 1152, 3), np.uint8), 1 / 30, top=90)
check(f"a `top` pushes the orb down to clear it ({low[1]}px)", low[1] == 90)
free = Reactor(150)
high = free.draw(np.zeros((648, 1152, 3), np.uint8), 1 / 30)
check(f"and without one it sits in the corner ({high[1]}px)", high[1] < 90)

print("\nThe orb costs what it should:")
# The wide bloom used to be an exact Gaussian over the whole canvas and was
# the single most expensive thing the overlay did -- 3.5ms of a 9.7ms orb.
# It is now blurred at a quarter size and scaled back up, which is the same
# glow for a sixteenth of the arithmetic.
import cv2 as _cv
fast = Reactor(280)
canvas = np.zeros((648, 1152, 3), np.uint8)
best = min(_time_orb(fast, canvas) for _ in range(25))
print(f"      one frame of the orb: {best:.2f} ms")
check(f"a frame of it is quick ({best:.2f}ms, was 9.7)", best < 6.0)
check("the bloom really is done small",
      fast._small.shape[0] == fast.size // rmod.BLOOM_SHRINK)

# The shading table has to give the same colours the arithmetic did, or the
# orb is a different object rather than a cheaper one.
near = np.linspace(0.0, 1.0, 512, dtype=np.float32)
was = (rmod.DEEP[None, :] * (1.0 - near[:, None])
       + rmod.WARM[None, :] * near[:, None])
hot = np.clip((near - 0.86) / 0.14, 0.0, 1.0)[:, None]
was = was * (1.0 - hot) + rmod.HOT[None, :] * hot
now = rmod._SHADE[np.clip(near * 255, 0, 255).astype(np.int32)]
off = float(np.abs(was - now).max())
check(f"the shading table matches the arithmetic it replaced ({off:.1f}/255)",
      off < 6.0)

print("\nThe frame is cropped before it is mirrored, not after:")
# Flipping 1080p and then throwing two thirds of it away is most of a
# millisecond a frame for pixels nobody sees. The surviving pixels have to be
# identical either way, or the glove lands somewhere the hand is not.
for width in (1920, 1366, 1280, 641):
    raw = np.random.randint(0, 255, (360, width, 3), dtype=np.uint8)
    margin = 0.20
    ax0, ax1 = int(width * margin), int(width * (1 - margin))
    before = cv2.flip(raw, 1)[:, ax0:ax1]
    after = cv2.flip(raw[:, width - ax1:width - ax0], 1)
    check(f"  identical at {width}px wide", np.array_equal(before, after))

print("\nWith the camera hidden it never mirrors at all:")
from gesture_control.controller import GestureController as _GC
blanked = _GC.__new__(_GC)
blanked._blank = blanked._blank_clean = blanked._clean_frame = None
surface = blanked._blank_for((100, 200, 3))
bare = blanked._clean_frame
check("there is a surface to draw on", surface.shape == (100, 200, 3))
# These cannot be the same array: the overlay decides what is opaque by
# comparing them, and comparing a thing with itself finds nothing.
check("and a separate one to compare it against", bare is not surface)
cv2.circle(surface, (100, 50), 20, (255, 200, 80), -1)
again = blanked._blank_for((100, 200, 3))
check("the surface is reused rather than allocated per frame", again is surface)
check("but wiped, so last frame's drawing does not linger", int(again.max()) == 0)
check("and the thing it is compared against stays black",
      int(blanked._clean_frame.max()) == 0)

print("\nThe reactor reacts to the hotword:")
r3 = Reactor(160)
quiet = np.zeros((300, 300, 3), np.uint8)
r3.draw(quiet, 1 / 30)
lit = np.zeros((300, 300, 3), np.uint8)
r3.flare()
r3.draw(lit, 1 / 30)
check(f"hearing it makes the reactor brighter "
      f"({int(quiet.mean() * 10) / 10} -> {int(lit.mean() * 10) / 10})",
      lit.mean() > quiet.mean() * 1.25)
check("and wider, so it reads as a pulse going outward",
      int((lit > 40).sum()) > int((quiet > 40).sum()))
for _ in range(20):                       # about two thirds of a second
    r3.draw(np.zeros((300, 300, 3), np.uint8), 1 / 30)
check("the flash is over quickly, not a new state", r3._flare == 0.0)
# Frame-rate independent: the same wall-clock decay however fast it is drawn.
a, b = Reactor(160), Reactor(160)
a.flare(); b.flare()
for _ in range(6):
    a.draw(np.zeros((300, 300, 3), np.uint8), 1 / 30)
for _ in range(12):
    b.draw(np.zeros((300, 300, 3), np.uint8), 1 / 60)
check(f"and lasts the same time at 30fps and 60fps "
      f"({a._flare:.3f} vs {b._flare:.3f})", abs(a._flare - b._flare) < 0.02)

print("\nThe reactor reacts to the reply being spoken:")
r4 = Reactor(160)
r4.set_state("speaking")
loud = np.zeros((300, 300, 3), np.uint8)
r4.set_state("speaking", 1.0)
r4.draw(loud, 1 / 30)
soft = np.zeros((300, 300, 3), np.uint8)
r4.set_state("speaking", 0.0)
r4.level = 0.0
r4.draw(soft, 1 / 30)
check(f"a loud syllable draws brighter than a silent one "
      f"({soft.mean():.1f} -> {loud.mean():.1f})", loud.mean() > soft.mean())

print("\nThe session log under the orb:")
from gesture_control.jarvis.terminal import Console, Glyphs

entries = [("you", "make me a folder"), ("note", "asking Groq"),
           ("reasoning", "they want a folder"),
           ("tool", "create_folder(path=~/Desktop/X)"),
           ("tool", "\u2192 Created C:/Users/x/Desktop/X"),
           ("tool", "list_folder(path=~/Desktop)"),
           ("tool", "\u2192 a b c"), ("tool", "\u2192 d e f"),
           ("jarvis", "Done, sir.")]

glyphs = Glyphs()
check(f"the glyph atlas builds ({glyphs.source})",
      glyphs.atlas.shape == (96, jterm.CELL_H, jterm.CELL_W))
check("every printable character has ink in it, and the space has none",
      int(glyphs.atlas[0].max()) == 0
      and all(int(glyphs.atlas[i].max()) > 0 for i in range(1, 95)))
# The whole point of the atlas is that the columns line up. If the font
# advanced by anything but the cell width the grid would drift, and the
# builder is meant to reject it rather than draw over the next column.
if glyphs.source == "consolas":
    from PIL import ImageFont
    f = ImageFont.truetype(jterm.FONT_PATH, jterm.FONT_SIZE)
    widths = {int(round(f.getlength(chr(c)))) for c in range(33, 127)}
    check(f"the font really is fixed-pitch ({widths})", widths == {jterm.CELL_W})
try:
    Glyphs.__dict__["_truetype"].__func__("C:/Windows/Fonts/arial.ttf", 16)
    check("a proportional font is refused, not drawn crookedly", False)
except Exception as exc:                                         # noqa: BLE001
    check(f"a proportional font is refused, not drawn crookedly ({exc})", True)
# Refusing it has to leave a working terminal, not no terminal: the grid, the
# wrapping and the scrollback do not depend on the shapes.
spare = Glyphs(path="C:/nowhere/no-such-font.ttf")
check(f"and falls back to a font that is always there ({spare.source})",
      spare.source == "hershey" and int(spare.atlas[0].max()) == 0
      and all(int(spare.atlas[i].max()) > 0 for i in range(1, 95)))
plain_con = Console(576, 240, glyphs=spare)
plain_con.feed([("you", "still readable"), ("jarvis", "Quite, sir.")])
check("and still draws a log in it",
      plain_con.render("idle") and int(plain_con.colour.max()) > 0)

con = Console(576, 324)
check(f"it lays out a character grid ({con.cols}x{con.rows})",
      con.cols > 30 and 8 < con.rows < 40)
check("wide enough for a tool call to read as one",
      con.cols >= len("create_folder(path=~/Desktop/X)"))
con.feed(entries)
check(f"every entry wraps into the grid ({len(con._wrapped)} lines)",
      len(con._wrapped) >= len(entries))
check("no line is wider than the grid",
      max(len(t) for _, t in con._wrapped) <= con.cols)
check("each kind opens with its own gutter",
      [t[:2] for t, in [(con._wrapped[0][1],)]] == ["> "]
      and con._wrapped[-1][1].startswith("< "))
check("a wrapped line is indented under its gutter, not re-marked",
      all(not line.startswith(("> ", "$ ", "< ")) or line[2:3] != " "
          for _, line in con._wrapped))

# Re-wrapping is meant to be free, so it happens whenever anything changed
# and never when nothing did.
con.feed(entries)
before = con._wrapped
con.feed(entries)
check("an unchanged log is not re-wrapped", con._wrapped is before)
con.feed(entries + [("jarvis", "One more thing.")])
check("a changed one is", con._wrapped is not before)

print("\n  It draws, and only redraws when it has to:")
con.feed(entries)
check("the first render draws", con.render("idle") is True)
check("and an identical second one declines", con.render("idle") is False)
check("a state change draws again", con.render("thinking") is True)
check("it fills the whole panel", con.colour.shape == (324, 576, 3))
check("the glass is see-through and the frame is not",
      int(con.alpha[con.height // 2, con.width // 2]) < 255
      and int(con.alpha[0, con.width // 2]) > 200)
check("text is opaque, whatever is behind the window",
      int(con.alpha.max()) == 255)
check("and it is drawn in the same light as the orb",
      int(con.colour[:, :, 0].max()) > int(con.colour[:, :, 2].max()))

print("\n  Scrollback:")
long_log = []
for n in range(1, 13):
    long_log += [("you", f"question number {n}"),
                 ("tool", f"list_folder(path=~/Desktop/project-{n})"),
                 ("tool", "\u2192 " + " ".join("file%02d.txt" % i for i in range(9))),
                 ("jarvis", f"That is answer {n}, sir.")]
con.feed(long_log)
total = len(con._wrapped)
check(f"a session's worth of output is held ({total} lines)", total > con.rows * 2)


def view(c):
    end = len(c._wrapped) - c.scroll
    return [t for _, t in c._wrapped[max(0, end - c.rows):end]]


check("it starts at the newest output", con.live and view(con)[-1].endswith("12, sir."))
con.render("idle")
con.scroll_by(jterm.WHEEL_LINES)
check(f"one wheel notch moves a few lines ({con.scroll})",
      con.scroll == jterm.WHEEL_LINES)
check("and that stops it following", not con.live)
check("and scrolling is a reason to redraw", con.render("idle") is True)
con.scroll_by(10_000)
check(f"scrolling past the start stops at the start ({con.scroll})",
      con.scroll == total - con.rows and view(con)[0] == con._wrapped[0][1])
con.scroll_by(-10_000)
check("and past the end stops at the end", con.scroll == 0 and con.live)

# The bug this exists to prevent: a reply streaming in a token at a time
# walking the view down a line per token while it is being read.
con.scroll_by(20)
held = view(con)
for extra in range(6):
    long_log.append(("jarvis", f"unread line {extra}"))
    con.feed(long_log)
check("new output does not move text out from under the reader", view(con) == held)
check("and the header says how far back it is holding", con.scroll > 20)
con.to_bottom()
con.feed(long_log)
check("going back to the bottom follows again",
      con.live and view(con)[-1].endswith("unread line 5"))

# Anchored to the floor of the panel when there is not enough to fill it, the
# way a shell's output sits on the bottom of the window rather than floating.
short = Console(576, 324)
short.feed([("you", "hi")])
short.render("idle")
rows = np.where(short._cover[short._body_y:, :].max(axis=1) > 0)[0]
check("a short log sits on the floor rather than floating at the top",
      int(rows.max()) > (short.rows - 1.5) * short.glyphs.cell_h)

print("\n  What it does with awkward text:")
check("a long result loses its middle, keeping both ends",
      jterm.shorten("START" + "x" * 4000 + "END", 80).startswith("START")
      and jterm.shorten("START" + "x" * 4000 + "END", 80).endswith("END"))
check("the model's curly quotes and arrows become ASCII it can draw",
      jterm.plain("\u201cdone\u201d \u2192 fine\u2026") == '"done" -> fine...')
odd = Console(576, 324)
odd.feed([("jarvis", "caf\u00e9 \u5b57 \u2603"), ("tool", "x" * 400)])
odd.render("idle")
check("anything it cannot draw does not crash it", odd.colour.max() > 0)
check("and a single unbroken word still wraps instead of looping",
      len(odd._wrapped) >= 400 // odd.cols)

print("\n  Scrolling it with a mouse:")
from gesture_control.jarvis.terminal import Terminal
term = Terminal.__new__(Terminal)
term.console = Console(576, 324)
term.console.feed(long_log)
term._drag_from = None
term._drag_scroll = 0
term._mouse("wheel", 40, 2)
check(f"the wheel scrolls back ({term.console.scroll})",
      term.console.scroll == 2 * jterm.WHEEL_LINES)
term._mouse("wheel", 40, -2)
check("and forward again", term.console.scroll == 0)
# Dragging is not a nicety: the window never takes focus, so it only gets
# wheel messages while Windows is routing them by hover -- and the
# hand-tracked cursor has no wheel at all.
cell = term.console.glyphs.cell_h
term._mouse("down", 200, 100)
term._mouse("drag", 200, 100 + cell * 5)
check(f"dragging down pulls older output into view ({term.console.scroll})",
      term.console.scroll == 5)
term._mouse("drag", 200, 100 + cell * 2)
check("dragging back up is measured from where the drag began, not the last "
      "event", term.console.scroll == 2)
term._mouse("up", 200, 100)
term._mouse("drag", 200, 900)
check("and a drag that never started is ignored", term.console.scroll == 2)

print("\n  And as a real window on the screen:")
live_log = [("you", "what is on my desktop")]
live_state = ["thinking"]
real = Terminal(600, 300, 576, 324, source=lambda: (live_log, live_state[0]))
check("it opens", real.available)
if real.available:
    hwnd = real._window._hwnd
    ex = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    check("it is see-through", bool(ex & 0x00080000))
    check("it never steals focus from what you were typing in",
          bool(ex & WS_EX_NOACTIVATE))
    check("and it does not float over every other window",
          not (ex & WS_EX_TOPMOST))
    time.sleep(0.25)
    check("its own clock draws it, without a camera frame",
          int(real._window.bgra[:, :, 3].max()) > 0)
    for i in range(40):
        live_log.append(("tool", f"→ entry {i} some file name.txt"))
    time.sleep(0.25)
    check(f"output arriving is picked up ({len(real.console._wrapped)} lines)",
          len(real.console._wrapped) > 40 and real.console.live)
    # The wheel exactly as Windows sends it: screen coordinates in lParam,
    # the notch count in the high word of wParam. Reading either the wrong
    # way is the whole failure mode, so this posts the real message rather
    # than calling the handler.
    point = (300 + 100) << 16 | (600 + 100)
    user32.SendMessageW(hwnd, 0x020A, 120 << 16, point)
    check(f"a real wheel message scrolls it back ({real.console.scroll})",
          real.console.scroll == jterm.WHEEL_LINES)
    user32.SendMessageW(hwnd, 0x020A, (-120 & 0xFFFF) << 16, point)
    check("and the other way", real.console.scroll == 0)
    real.close()
    check("it closes", not real.available)

print("\n  Where it lands, on a screen of any size:")
# The one piece of arithmetic in all of this that is easy to get wrong. The
# orb is drawn into the *camera frame*, which the overlay then stretches over
# the whole monitor, so where it ends up on screen depends on the camera's
# resolution and on how much of its view still reaches the screen. The
# terminal has to hang off the bottom of it wherever that turns out to be.
import types
from gesture_control.controller import GestureController

placed = []


class FakeTerminal:
    def __init__(self, x, y, w, h, source=None):
        self.available, self.notes = True, []
        placed.append((x, y, w, h))

    def close(self):
        pass


from gesture_control.jarvis import terminal as _tmod
_real_terminal, _tmod.Terminal = _tmod.Terminal, FakeTerminal
try:
    for screen, cam in (((1920, 1080), (648, 1152)), ((2560, 1440), (720, 1280)),
                        ((1366, 768), (432, 768))):
        placed.clear()
        ctl = GestureController.__new__(GestureController)
        ctl._terminal, ctl._terminal_placed = None, False
        ctl.model = types.SimpleNamespace(screen=screen)
        ctl.jarvis = types.SimpleNamespace(settings=store.JarvisSettings(),
                                           log=lambda: [], state="idle")
        # As _start_jarvis works it out, with the audio widget present.
        ctl._panel_floor = screen[1] - 132 - 46 - 10
        orb = Reactor(int(min(screen) * 0.26))
        frame = np.zeros((*cam, 3), np.uint8)
        orb_box = orb.draw(frame, 1 / 30, top=int(52 * cam[1] / screen[0]))
        ctl._place_terminal(frame.shape, orb_box)

        kx, ky = screen[0] / cam[1], screen[1] / cam[0]
        orb_right, orb_bottom = orb_box[2] * kx, orb_box[3] * ky
        (x, y, w, h), = placed
        label = f"{screen[0]}x{screen[1]}"
        check(f"{label}: it sits below the orb, not over it ({y} vs {orb_bottom:.0f})",
              y >= orb_bottom)
        check(f"{label}: aligned to the orb's right edge, so it reads as one column",
              abs((x + w) - orb_right) < 30)
        check(f"{label}: fully on the screen", x >= 0 and x + w <= screen[0])
        check(f"{label}: and it stops well short of the bottom "
              f"({y + h} of {screen[1]})",
              y + h <= ctl._panel_floor and y + h < screen[1] * 0.92)
        check(f"{label}: with room to read ({(w - 26) // jterm.CELL_W} columns)",
              (w - 26) // jterm.CELL_W >= 38)

    # No room at all is a printed line, not a window jammed into 20 pixels.
    placed.clear()
    tight = GestureController.__new__(GestureController)
    tight._terminal, tight._terminal_placed = None, False
    tight.model = types.SimpleNamespace(screen=(1920, 1080))
    tight.jarvis = types.SimpleNamespace(settings=store.JarvisSettings(),
                                         log=lambda: [], state="idle")
    tight._panel_floor = 560
    tight._place_terminal((648, 1152, 3), (900, 60, 1100, 500))
    check("with no room under the orb it opens nothing rather than a sliver",
          placed == [] and tight._terminal is None)
    check("and it does not try again on every frame", tight._terminal_placed)

    off = GestureController.__new__(GestureController)
    off._terminal, off._terminal_placed = None, False
    off.model = types.SimpleNamespace(screen=(1920, 1080))
    off.jarvis = types.SimpleNamespace(
        settings=store.JarvisSettings(terminal=False), log=lambda: [],
        state="idle")
    off._panel_floor = 900
    placed.clear()
    off._place_terminal((648, 1152, 3), (900, 60, 1100, 400))
    check("and turning it off in jarvis.json leaves it shut", placed == [])
finally:
    _tmod.Terminal = _real_terminal

print("\nA streamed turn folds into the transcript:")
from gesture_control.jarvis.agent import Jarvis
j = Jarvis.__new__(Jarvis)
j.transcript = __import__("collections").deque(maxlen=24)
j._lock = threading.Lock()
j._open = {}
j._speaker = None
j._unsaid, j._spoke_any = "", False
reply = ""
for event in ({"type": "intent", "tag": "tool_use"},
              {"type": "provider", "name": "Groq"},
              {"type": "reasoning", "text": "they want "},
              {"type": "reasoning", "text": "a folder"},
              {"type": "tool_call", "name": "create_folder", "arguments": {"path": "~/X"}},
              {"type": "tool_result", "name": "create_folder", "result": "Created X"},
              {"type": "token", "text": "Done, "},
              {"type": "token", "text": "sir."}):
    reply = j._event(event, reply)
kinds = [k for k, _ in j.transcript]
check(f"one entry per thing that happened ({kinds})",
      kinds == ["note", "note", "reasoning", "tool", "tool", "jarvis"])
# The bug this catches: reasoning merged onto the end of "asking Groq",
# because both were styled the same and _extend merged by style.
check("the model's reasoning does not glue onto the provider line",
      dict(zip(kinds, [t for _, t in j.transcript]))["note"] == "asking Groq"
      or [t for k, t in j.transcript if k == "note"][-1] == "asking Groq")
check("reasoning arriving in fragments becomes one entry",
      [t for k, t in j.transcript if k == "reasoning"] == ["they want a folder"])
check("the reply builds a token at a time rather than a slab each",
      [t for k, t in j.transcript if k == "jarvis"] == ["Done, sir."])
check("and the assembled reply comes back for speaking", reply == "Done, sir.")

# Taken from a real turn against FreeClaw: a reasoning fragment arrived
# between two token chunks, and merging only with the *previous* entry split
# the reply into "The" / "." / "file has been created, sir."
j2 = Jarvis.__new__(Jarvis)
j2.transcript = __import__("collections").deque(maxlen=24)
j2._lock = threading.Lock()
j2._open = {}
j2._speaker = None
j2._unsaid, j2._spoke_any = "", False
reply2 = ""
for event in ({"type": "token", "text": "The"},
              {"type": "reasoning", "text": "\u2026"},
              {"type": "token", "text": " file has been created, sir."}):
    reply2 = j2._event(event, reply2)
speech = [t for k, t in j2.transcript if k == "jarvis"]
check(f"a reasoning chunk mid-reply does not split it ({speech})",
      speech == ["The file has been created, sir."])
check("and the spoken reply matches what is in the log",
      reply2 == speech[0])


print("\nReading the machine it is running on:")
from gesture_control.jarvis import sensors as jsense

tap = jsense.sampler()
check("the sampler starts", tap.available or "psutil" in tap.unavailable_reason)
time.sleep(2.2)
check(f"it keeps a history, not just a reading ({len(tap.history('cpu'))} samples)",
      len(tap.history("cpu")) >= 2)
# Network and disk are counters that only go up, so the first sample cannot
# produce a rate -- only the second one can. Getting this wrong shows up as a
# throughput of several gigabytes a second, being the total since boot.
check("traffic is reported as a rate, not as the total since boot",
      tap.now("net_down") < 2e9)
check("and a rate needs two samples, so it appears one tick in",
      len(tap.history("net_down")) >= 1)

cpu = jsense.cpu()
check(f"it reads the processor ({cpu.get('threads')} threads, "
      f"{cpu.get('percent', -1):.0f}%)",
      cpu.get("threads", 0) >= 1 and 0 <= cpu.get("percent", -1) <= 100)
ram = jsense.memory()
check(f"and the memory ({jsense.human_bytes(ram['total'])})",
      ram["total"] > 1e9 and 0 <= ram["percent"] <= 100)
check(f"and the drives ({len(jsense.disks())})", len(jsense.disks()) >= 1)
check(f"and how long it has been up ({jsense.human_span(jsense.uptime())})",
      jsense.uptime() > 0)
power = jsense.battery()
check(f"it says plainly whether there is a battery ({power.get('present')})",
      "present" in power)

print("\n  Sizes and spans, as a person would say them:")
for raw, want in ((512, "512B"), (2048, "2.0KB"), (5_368_709_120, "5.0GB")):
    check(f"{raw} -> {jsense.human_bytes(raw)}", jsense.human_bytes(raw) == want)
for raw, want in ((45, "45s"), (600, "10m"), (7200, "2h 0m"), (180_000, "2d 2h")):
    check(f"{raw}s -> {jsense.human_span(raw)}", jsense.human_span(raw) == want)

print("\n  The network it is actually on:")
net = jsense.network()
check(f"it finds an adapter with an address ({len(net['adapters'])})",
      len(net["adapters"]) >= 1
      and all("." in a["address"] for a in net["adapters"]))
check("and none of them is the loopback",
      not any(a["address"].startswith("127.") for a in net["adapters"]))
check(f"it reads the default gateway ({net['gateway'] or 'none'})",
      net["gateway"] == "" or net["gateway"].count(".") == 3)
check(f"and says whether the internet answers ({net['online']})",
      isinstance(net["online"], bool))
link = net["wifi"]
if link.get("connected"):
    check(f"the wireless link is read from netsh (signal "
          f"{link.get('signal_percent')}%)",
          bool(link.get("ssid")) and 0 <= link.get("signal_percent", -1) <= 100)
else:
    check(f"there is no wireless link, and it says so ({link})", True)
# netsh is a process launch, so asking twice in a row must not cost twice.
start = time.perf_counter()
for _ in range(50):
    jsense.wifi()
check(f"asking for it fifty times costs one launch "
      f"({(time.perf_counter() - start) * 1000:.0f}ms)",
      time.perf_counter() - start < 1.0)

print("\n  What is running:")
# The bug this catches: psutil reports 0.0% for a process the first time it is
# asked, so a single walk of the table says the whole machine is idle for ever.
rows = jsense.processes(40)
check(f"it lists processes ({len(rows)})", len(rows) > 5)
check("with real CPU shares on the very first call, not a table of zeros",
      any(r["cpu"] > 0.01 for r in rows))
check("and shares that add up to about one machine, not one core",
      sum(r["cpu"] for r in rows) <= 130.0)
check("with memory for each", all(r["memory"] >= 0 for r in rows))

print("\nThe tools that answer questions:")
from gesture_control.jarvis import status as jstatus

for name in ("system_info", "battery_status", "network_status", "disk_usage",
             "list_processes", "sensor_history"):
    try:
        answer = tools.call(name, {})
        first = answer.splitlines()[0][:58]
        check(f"{name} answers ({first})", len(answer) > 10)
    except Exception as exc:                                     # noqa: BLE001
        check(f"{name} answers ({exc})", False)
check("system_info reports on the machine, not on FreeClaw's",
      "Hostname" in tools.call("system_info", {})
      and "Memory" in tools.call("system_info", {}))
check("list_processes leaves out the idle process, which is always the "
      "busiest thing on a quiet machine",
      "Idle" not in tools.call("list_processes", {"limit": 5}))

print("\n  ...and refuse nonsense rather than inventing an answer:")
for name, args, expect in (
        ("sensor_history", {"series": "vibes"}, "no reading"),
        ("disk_usage", {"path": "Q:"}, "no drive"),
        ("fetch_url", {"url": "file:///C:/Windows"}, "http"),
        ("fetch_url", {"url": ""}, "address"),
        ("end_process", {}, "which process"),
        ("ping", {"host": "no-such-host-anywhere.invalid"}, "resolve")):
    try:
        tools.call(name, args)
        check(f"{name}{args} is refused", False)
    except Exception as exc:                                     # noqa: BLE001
        check(f"{name} refuses it and says why ({str(exc)[:46]})",
              expect.lower() in str(exc).lower())
# The guard that matters: an agent told to "close whatever is using the CPU"
# must not be able to resolve that to part of Windows.
for guarded in ("lsass", "csrss", "System Idle Process"):
    try:
        tools.call("end_process", {"name": guarded})
        check(f"end_process refuses {guarded}", False)
    except Exception as exc:                                     # noqa: BLE001
        check(f"end_process refuses {guarded} ({str(exc)[:40]})",
              "will not" in str(exc).lower() or "nothing is running" in str(exc).lower())

found = tools.call("search_files", {"pattern": "*.txt", "limit": 5})
check(f"search_files searches the real folders ({found.splitlines()[0][:46]})",
      "match" in found or "Nothing matching" in found)

print("\nThe widgets it can put on the screen:")
from gesture_control.jarvis import cards as jcards

deck = jcards.Board(screen=(1920, 1080))
try:
    check(f"every monitor has a producer and a height "
          f"({len(jcards.MONITORS)})",
          set(jcards.MONITORS) <= set(jcards.MONITOR_HEIGHT))
    for kind in sorted(jcards.MONITORS):
        spec = jcards.MONITORS[kind]()
        check(f"  the {kind} monitor produces something to draw "
              f"({spec['title']}, {len(spec['lines'])} lines)",
              spec["title"] and (spec["lines"] or spec["series"]))
    opened = deck.open_monitor("cpu")
    check(f"opening one works ({opened})", "cpu" in deck.cards)
    check("opening it twice does not stack two on top of each other",
          "already" in deck.open_monitor("cpu") and len(deck.cards) == 1)
    deck.open_monitor("memory")
    # Cards must not land on each other: the whole value of one is that it
    # stays where it was put and stays readable.
    boxes = [(c.x, c.y, c.x + c.width, c.y + c.height) for c in deck.cards.values()]
    clash = any(a is not b and a[0] < b[2] and b[0] < a[2]
                and a[1] < b[3] and b[1] < a[3]
                for a in boxes for b in boxes)
    check(f"a second card is placed clear of the first ({boxes})", not clash)
    check("and both are on the screen",
          all(0 <= b[0] and b[2] <= 1920 and 0 <= b[1] and b[3] <= 1080
              for b in boxes))

    deck.open_note("Checklist", "one\ntwo\nthree", gauge=0.5)
    check("a note of its own text opens", "checklist" in deck.cards)
    before = deck.cards["checklist"]
    deck.open_note("Checklist", "one\ntwo\nthree\nfour")
    check("writing to the same title updates it rather than opening another",
          deck.cards["checklist"] is before and len(deck.cards) == 3)
    check("and the new text is what it now says",
          "four" in deck._note_spec("checklist")["lines"])
    check("a card is at most so tall, whatever it is given",
          deck.cards["checklist"].height <= 1080 // 2)
    check(f"it lists what is up ({len(deck.cards)})",
          "checklist" in deck.listing() and "cpu" in deck.listing())
    try:
        deck.close("nothing called this")
        check("closing a card that is not there is an error", False)
    except jcards.CardError as exc:
        check("closing a card that is not there says what is",
              "no card" in str(exc).lower())
    deck.close("cpu")
    check("closing one takes it down", "cpu" not in deck.cards)
    check("and 'all' clears the screen",
          "Cleared" in deck.close("all") and not deck.cards)
finally:
    deck.shutdown()

print("\n  A card draws itself:")
drawn = jcards.Card("probe", "note", 0, 0, 200,
                    lambda: jcards._spec("PROBE", ["a\t1", "b\t2"], gauge=0.5,
                                         series=[1, 5, 3, 9, 2]))
try:
    if drawn.available:
        drawn.tick()
        check("it fills a picture", int(drawn.colour.max()) > 0)
        check("the glass is see-through and the frame is not",
              int(drawn.alpha[100, 158]) < 255 and int(drawn.alpha[0, 158]) > 200)
        check("drawn in the same light as everything else",
              int(drawn.colour[:, :, 0].max()) > int(drawn.colour[:, :, 2].max()))
        seen = drawn._drawn
        drawn.tick()
        check("and it declines to redraw a picture that has not changed",
              drawn._drawn is seen)
    else:
        check(f"a card window opens ({drawn.unavailable_reason})", False)
finally:
    drawn.close()

print("\n  Cards backed by a file:")
import cv2 as _cv2

sandbox = pathlib.Path(tempfile.mkdtemp())
note_file = sandbox / "ping.md"
note_file.write_text("# Ping\n\n- 09:14 build green\n- 09:20 deploy queued\n",
                     encoding="utf-8")
picture_file = sandbox / "probe.png"
_cv2.imwrite(str(picture_file), np.full((240, 360, 3), 120, np.uint8))

# The card resolves through the same guard the file tools use, so the test
# points the roots at a sandbox rather than writing into the real Desktop.
real_roots = tools.ROOTS
tools.ROOTS = (sandbox,)
deck2 = jcards.Board(screen=(1920, 1080))
try:
    said = deck2.open_file("Ping", str(note_file))
    check(f"a text file opens as a card ({said[:40]})", "ping" in deck2.cards)
    shown = deck2._files["ping"]["lines"]
    check(f"it shows what is in the file ({shown})",
          any("build green" in line for line in shown))
    check("and markdown heading marks are dropped, since the card has a title",
          not any(line.startswith("#") for line in shown))

    # The whole point of following the file: Jarvis rewrites it and the card
    # changes, with no second tool call. A card that has to be re-issued is
    # silently stale the moment anything else happens.
    note_file.write_text("# Ping\n\n- 09:31 DEPLOY DONE\n", encoding="utf-8")
    deck2._files["ping"]["checked"] = 0.0
    after = deck2._file_spec("ping")["lines"]
    check(f"rewriting the file changes the card ({after})",
          any("DEPLOY DONE" in line for line in after) and after != shown)

    deck2.open_file("Probe", str(picture_file))
    spec = deck2._file_spec("probe")
    check("a picture opens as a card", spec["image"] is not None)
    check(f"sized to the picture's shape ({deck2.cards['probe'].height}px tall)",
          96 < deck2.cards["probe"].height <= jcards.IMAGE_MAX_H + 64)
    check("and it says how big the picture is", "360x240" in spec["note"])
    deck2.cards["probe"].tick()
    card = deck2.cards["probe"]
    check("the picture is drawn opaque -- a photograph you can see the "
          "wallpaper through is not a photograph",
          int(card.alpha[card.height // 2, card.width // 2]) == 255)

    # A file that goes away mid-life has to say so rather than keep showing a
    # stale copy as though it were current.
    note_file.unlink()
    deck2._files["ping"]["checked"] = 0.0
    gone = deck2._file_spec("ping")
    check(f"a file that disappears is reported, not quietly kept ({gone['note']})",
          "missing" in gone["note"])

    outside = pathlib.Path(tempfile.mkdtemp()) / "elsewhere.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        deck2.open_file("Nope", str(outside))
        check("a file outside the allowed folders is refused", False)
    except jcards.CardError as exc:
        check(f"a file outside the allowed folders is refused ({str(exc)[:34]})",
              "outside" in str(exc).lower())
    try:
        deck2.open_file("Nope", str(sandbox / "not-here.md"))
        check("a file that is not there is refused", False)
    except jcards.CardError as exc:
        check("a file that is not there is refused", "no file" in str(exc).lower())
finally:
    deck2.shutdown()
    tools.ROOTS = real_roots

print("\n  Through the tools, as the model reaches it:")
check("a gauge given as a percentage becomes a fraction",
      "on the screen" in tools.call(
          "show_card", {"title": "Probe", "body": "x", "gauge": 60}))
check("list_cards sees it", "probe" in tools.call("list_cards", {}).lower())
check("close_card takes it away",
      "Cleared" in tools.call("close_card", {"name": "all"}))
try:
    tools.call("show_card", {"title": "  ", "body": "x"})
    check("a card with no title is refused", False)
except Exception as exc:                                         # noqa: BLE001
    check(f"a card with no title is refused ({str(exc)[:40]})", "title" in str(exc))
jcards.shutdown()

print("\nForgetting the conversation, as the RESET button does:")


class FakeChat:
    def __init__(self):
        self.resets = 0
        self.fail = False

    def reset(self):
        if self.fail:
            raise client.FreeClawError("FreeClaw is not answering.")
        self.resets += 1


gone = Jarvis.__new__(Jarvis)
gone.transcript = __import__("collections").deque(maxlen=24)
gone.transcript.extend([["you", "something"], ["jarvis", "something else"]])
gone._lock = threading.Lock()
gone._open = {"jarvis": ["jarvis", "x"]}
gone._speaker = None
gone._caption, gone._caption_until = "", 0.0
gone._unsaid, gone._spoke_any = "", False
gone._voice = None
gone.state = "idle"
gone.settings = store.JarvisSettings()
chat = FakeChat()
gone._chat = chat
gone.forget()
# Both sides, because either alone is a lie: clearing only the log hides a
# history the next turn still arrives carrying.
check("it clears the session log", len(gone.transcript) == 1)
check("and tells FreeClaw to drop the conversation too", chat.resets == 1)
check("and says so, rather than doing it silently",
      "cleared" in gone.caption().lower())
check("the half-finished reply goes with it", gone._open == {})

chat.fail = True
gone.forget()
check("a FreeClaw that will not reset is reported, not swallowed",
      any("not answering" in t for _, t in gone.transcript))

alone = Jarvis.__new__(Jarvis)
alone.transcript = __import__("collections").deque(maxlen=24)
alone._lock = threading.Lock()
alone._open = {}
alone._caption, alone._caption_until = "", 0.0
alone._chat = None
alone.forget()
check("with no FreeClaw linked it says there is nothing to forget",
      "nothing to forget" in alone.caption().lower())

print("\nStanding orders -- the part that speaks up by itself:")
from gesture_control.jarvis import watch as jwatch

check(f"it knows what it can watch ({len(jwatch.READINGS)})",
      {"cpu", "memory", "battery", "disk_free", "latency"} <= set(jwatch.READINGS))
check("an unreachable internet reads as a number, not as nothing -- a watch "
      "whose reading vanishes at the moment of interest could never fire",
      jwatch.READINGS["latency"][0]() is not None)

said = []
held = jwatch.FOR_SECONDS
jwatch.FOR_SECONDS = 1.2
real_cpu = jwatch.READINGS["cpu"]
level = [10.0]
jwatch.READINGS["cpu"] = (lambda: level[0], "%", True)
orders = jwatch.Watcher(announce=said.append, period=0.4)
try:
    opened = orders.add("cpu", True, 50, "The processor is pinned, sir.", "busy")
    check(f"a standing order is set ({opened[:44]})", "busy" in orders.watches)
    time.sleep(1.0)
    check("nothing is said while the condition is false", said == [])

    # The whole design point: CPU touches 100% every time anything launches,
    # and an assistant that announces each of those gets turned off.
    level[0] = 90.0
    time.sleep(0.9)
    check("and nothing is said the moment it becomes true, either", said == [])
    time.sleep(1.6)
    check(f"it speaks once the condition has held ({said[:1]})", len(said) == 1)
    check("and says what the reading actually was", "90%" in said[0])

    fired = len(said)
    time.sleep(1.6)
    check("it does not keep saying it while the condition stays true",
          len(said) == fired)
    # Hysteresis rather than a cooldown: a value sitting exactly on the
    # threshold must not chatter across it.
    level[0] = 49.0
    time.sleep(1.0)
    level[0] = 51.0
    time.sleep(2.2)
    check(f"a reading hovering on the threshold does not chatter ({len(said)})",
          len(said) == fired)
    level[0] = 20.0
    time.sleep(1.0)
    level[0] = 95.0
    time.sleep(2.2)
    check(f"but once it has properly recovered it can fire again ({len(said)})",
          len(said) > fired)

    # It must not talk over a turn. This is the difference between an
    # assistant and an alarm clock.
    quiet = jwatch.Watcher(announce=said.append, period=0.3)
    quiet.may_speak = lambda: False
    jwatch.FOR_SECONDS = 0.5
    quiet.add("cpu", True, 50, "second watcher", "two")
    before = len(said)
    time.sleep(2.0)
    check("an announcement waits while Jarvis is busy rather than cutting in",
          len(said) == before and quiet._pending)
    quiet.may_speak = lambda: True
    time.sleep(1.0)
    check("and lands as soon as it is free", len(said) > before)
    quiet.stop()

    check(f"it lists its orders ({orders.listing().splitlines()[0]})",
          "busy" in orders.listing())
    check("it reports how many times one has fired", "fired" in orders.listing())
    try:
        orders.remove("nothing called this")
        check("cancelling one that is not there is an error", False)
    except jwatch.WatchError as exc:
        check("cancelling one that is not there says so", "not watching" in str(exc))
    check("cancelling works", "Stopped" in orders.remove("busy"))
    orders.add("cpu", True, 50, "", "again")
    check("and 'all' clears them", "Cleared" in orders.remove("all")
          and not orders.watches)
finally:
    orders.stop()
    jwatch.FOR_SECONDS = held
    jwatch.READINGS["cpu"] = real_cpu

print("\n  Through the tools:")
check("a direction it does not understand is refused, with the words it wants",
      _refused("watch_for", {"reading": "cpu", "direction": "sideways",
                             "threshold": 5}, "above"))
check("a reading it cannot take is refused, and it lists the ones it can",
      _refused("watch_for", {"reading": "the weather", "direction": "above",
                             "threshold": 5}, "cpu"))
check("a threshold that is not a number is refused",
      _refused("watch_for", {"reading": "cpu", "direction": "above",
                             "threshold": "quite high"}, "number"))
check("setting one through the tool works",
      "Watching" in tools.call("watch_for", {"reading": "disk_free",
                                             "direction": "below",
                                             "threshold": 1,
                                             "name": "space"}))
check("list_watches sees it", "space" in tools.call("list_watches", {}))
check("stop_watching clears it",
      "Cleared" in tools.call("stop_watching", {"name": "all"}))
jwatch.shutdown()

print("\nTalking to it:")
from gesture_control.jarvis import voice as jvoice

print("\n  Speaking a reply before it has finished arriving:")
for text, sentences, rest in (
        ("Chrome is up, sir. You have 12 gigabytes free. Anything else?",
         ["Chrome is up, sir. You have 12 gigabytes free."], "Anything else?"),
        ("Half a sent", [], "Half a sent"),
        ("Version 3.5 is installed and working fine.",
         ["Version 3.5 is installed and working fine."], "")):
    got, left = jvoice.split_sentences(text)
    check(f"{text[:38]!r} -> {len(got)} to say",
          got == sentences and (rest is None or left.strip() == rest))
# The decimal point is the trap: splitting on every full stop turns "3.5"
# into two utterances and the reply comes out as "three point" "five".
whole, _ = jvoice.split_sentences("It is 3.5 gigabytes, sir. Plenty left.")
check(f"a decimal point does not end a sentence ({whole})",
      whole and "3.5" in whole[0])
tiny, left = jvoice.split_sentences("Yes. No. Done.")
check(f"fragments are joined rather than sent one word at a time ({tiny}, {left!r})",
      len(tiny) <= 1)

print("\n  The follow-up window, so you do not have to summon it every turn:")
ear = jvoice.Voice.__new__(jvoice.Voice)
ear._followup_until = 0.0
check("it is not expecting anything to begin with", not ear.expecting)
ear.expect_reply(5.0)
check("a finished reply opens a window for the next thing you say", ear.expecting)
ear.drop_followup()
check("and the window closes the moment a turn starts", not ear.expecting)
ear.expect_reply(0.0)
check("zero seconds means the wake word is always required", not ear.expecting)

print("\n  Being interrupted:")
stopped = []


class FakeSpeaker:
    def stop(self):
        stopped.append(True)


from gesture_control.jarvis.agent import Jarvis
mid = Jarvis.__new__(Jarvis)
mid.transcript = __import__("collections").deque(maxlen=24)
mid._lock = threading.Lock()
mid._open = {}
mid._speaker = FakeSpeaker()
mid._unsaid, mid._spoke_any = "half a senten", True
mid._interrupted()
check("saying the wake word over a reply stops it talking", stopped == [True])
check("and it forgets the half-sentence it was about to say",
      mid._unsaid == "" and mid._spoke_any is False)
check("the session log records that you cut it off",
      any("interrupt" in t for _, t in mid.transcript))

print("\n  The speaker's queue:")
mute = jvoice.Speaker.__new__(jvoice.Speaker)
mute.available, mute.speaking, mute.level = True, False, 0.0
mute._epoch, mute._lock = 0, threading.Lock()
mute._text, mute._audio = __import__("queue").Queue(), __import__("queue").Queue()
mute.say("first sentence")
mute.say("second sentence")
check("sentences queue up in order rather than talking over each other",
      mute._text.qsize() == 2 and mute.speaking)
was = mute._epoch
mute.stop()
check("stopping empties the queue", mute._text.qsize() == 0 and not mute.speaking)
# A sentence already handed to the synthesiser cannot be recalled, so it has
# to be thrown away when it comes back instead -- which is what the epoch is.
check("and marks anything already in flight as stale", mute._epoch > was)

print("\nThe buttons in the corner:")
from gesture_control.jarvis.toggle import (HEIGHT as B_H, MARGIN as B_M, Strip,
                                           Toggle, WIDTH as B_W)

pressed = []
strip = Strip(1920, on_gestures=lambda g: pressed.append(("gestures", g)),
              on_reset=lambda: pressed.append("reset"),
              on_quit=lambda: pressed.append("quit"))
try:
    check(f"all three open ({[b.text() for b in strip.all]})",
          all(b.available for b in strip.all))
    # Each needs its own window class: a class name is registered once per
    # process, and two buttons sharing one would share the first one's
    # handler -- so RESET would quit.
    classes = [b._class for b in strip.all]
    check(f"each has a window class of its own ({len(set(classes))})",
          len(set(classes)) == 3)
    boxes = [(b._x, b._x + b.width) for b in strip.all]
    overlap = any(a is not c and a[0] < c[1] and c[0] < a[1]
                  for a in boxes for c in boxes)
    check(f"they do not overlap each other ({boxes})", not overlap)
    check("and they all fit on the screen",
          all(0 <= a and b <= 1920 for a, b in boxes))
    check(f"the toggle is the one flush in the corner ({strip.toggle._x})",
          strip.toggle._x + strip.toggle.width == 1920 - B_M)
    # QUIT is deliberately the far one: it is what you least want to hit
    # while reaching for the button you actually use.
    check("QUIT sits furthest from the toggle, not next to it",
          strip.quit._x < strip.reset._x < strip.toggle._x)
    check("the orb is told where the strip ends", strip.bottom == B_M + B_H)
    strip.reset.clicked()
    check(f"RESET calls what it was given ({pressed})", pressed == ["reset"])
    strip.quit.clicked()
    check("QUIT calls what it was given", pressed == ["reset", "quit"])
    check("and neither of them touches the gesture state",
          strip.toggle.gestures is True)
    for button in strip.all:
        ex = user32.GetWindowLongPtrW(button._hwnd, GWL_EXSTYLE)
        check(f"  {button.text()} never steals focus when clicked",
              bool(ex & WS_EX_NOACTIVATE))
        check(f"  {button.text()} does not float over every other window",
              not (ex & WS_EX_TOPMOST))
finally:
    strip.close()
check("they close", not any(b.available for b in strip.all))

print("\nThe toggle that hands the mouse back:")
seen = []
tog = Toggle(40, 40, on_change=seen.append)
check("it opened", tog.available)
if tog.available:
    check("it starts driving the mouse", tog.gestures is True)
    tog.set(False)
    check("switching off is reported once", seen == [False] and tog.gestures is False)
    tog.set(False)
    check("setting it to what it already is changes nothing", seen == [False])
    tog.set(True)
    check("and back on", seen == [False, True])
    # It used to be topmost so it could never be buried. The overlay is
    # pinned under the applications, and a control floating over everything
    # did not belong to the same interface -- so this went under with it, and
    # Ctrl+Alt+M became the way to reach it when something is covering it.
    style = user32.GetWindowLongPtrW(tog._hwnd, GWL_EXSTYLE)
    check("it does not float over every other window",
          not (style & WS_EX_TOPMOST))
    check("and it never steals focus when clicked", style & WS_EX_NOACTIVATE)
    tog.close()
    check("it closes", not tog.available)

print("\nThe hologram window the slider lives in:")
from gesture_control.jarvis.widget import HoloWindow, premultiply
seen = []
holo = HoloWindow(300, 300, 200, 80, on_mouse=lambda e, x, y: seen.append((e, x, y)),
                  name="GestureControlTestWidget")
check("it opened", holo.available)
if holo.available:
    style = user32.GetWindowLongPtrW(holo._hwnd, GWL_EXSTYLE)
    check("it is layered, so it can be see-through", style & 0x00080000)
    # The whole reason it is not just part of the overlay.
    check("it is NOT click-through -- a slider has to be draggable",
          not (style & 0x00000020))
    check("it never takes focus, so dragging it does not interrupt typing",
          style & WS_EX_NOACTIVATE)
    check("it exposes a BGRA buffer the size of the window",
          holo.bgra is not None and holo.bgra.shape == (80, 200, 4))

    # Premultiplication: the same rule the overlay follows, for the same
    # reason -- UpdateLayeredWindow assumes it, and skipping it haloes edges.
    colour = np.full((80, 200, 3), 200, np.uint8)
    alpha = np.full((80, 200), 128, np.uint8)
    premultiply(colour, alpha, holo.bgra)
    check(f"colour goes in premultiplied by alpha ({holo.bgra[10, 10, 0]})",
          abs(int(holo.bgra[10, 10, 0]) - 200 * 128 // 255) <= 1)
    check("and the alpha channel is carried through",
          int(holo.bgra[10, 10, 3]) == 128)
    holo.blit()
    holo.close()
    check("it closes", not holo.available)

print("\nThe audio widget:")
from gesture_control.jarvis import audio as jaudio
from gesture_control.jarvis.audio_panel import AudioPanel, SLIDER_LEFT, SLIDER_RIGHT, SLIDER_Y

vol = jaudio.Volume()
check(f"the system volume is reachable ({vol.unavailable_reason or 'yes'})",
      vol.available)
if vol.available:
    before = vol.get()
    vol.set(0.37)
    time.sleep(0.1)
    got = vol.get()
    check(f"setting it moves the real system volume ({got:.2f})",
          abs(got - 0.37) < 0.02)
    vol.set(before)
    time.sleep(0.1)
    check(f"and the test puts it back ({vol.get():.2f} was {before:.2f})",
          abs(vol.get() - before) < 0.02)

tap = jaudio.Tap()
started = tap.start()
check(f"the output tap starts ({'spectrum' if tap.spectral else 'level meter'})",
      started)
if started:
    check(f"it reports one value per bar ({len(tap.bars)})",
          len(tap.bars) == jaudio.BARS)
    check("and they are all in range",
          bool(np.all(tap.bars >= 0.0) and np.all(tap.bars <= 1.0)))
    tap.stop()

# The slider maths, against a stub -- the real one would move the machine's
# volume, and a test that leaves the speakers somewhere else is a bad test.
class FakeVolume:
    available = True

    def __init__(self):
        self.level = 0.5

    def get(self):
        return self.level

    def set(self, v):
        self.level = v

    def muted(self):
        return False


panel = AudioPanel.__new__(AudioPanel)
panel.volume = FakeVolume()
panel._dragging = False
panel._shown = 0.5
for label, x, want in (("the far left is silence", SLIDER_LEFT, 0.0),
                       ("the far right is full", SLIDER_RIGHT, 1.0),
                       ("the middle is half", (SLIDER_LEFT + SLIDER_RIGHT) // 2, 0.5)):
    panel._dragging = False
    panel._mouse("down", x, SLIDER_Y)
    check(f"{label} ({panel.volume.level:.2f})", abs(panel.volume.level - want) < 0.02)
panel._mouse("up", SLIDER_RIGHT, SLIDER_Y)

panel.volume.level = 0.5
panel._dragging = False
panel._mouse("down", SLIDER_RIGHT, SLIDER_Y - 60)      # nowhere near the track
check("a click away from the slider is ignored", abs(panel.volume.level - 0.5) < 1e-6)
# Dragging off the end of the window reports a negative x; reading it as
# unsigned would send the volume to maximum instead of zero.
panel._dragging = True
panel._mouse("drag", -40, SLIDER_Y)
check(f"dragging past the left end pins to silence ({panel.volume.level:.2f})",
      panel.volume.level == 0.0)

print(chr(10) + "RESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
