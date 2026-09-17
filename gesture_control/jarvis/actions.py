"""The things Jarvis can do to this machine that are not files.

Launching applications, moving windows around, the volume and the media keys,
the clipboard, a screenshot, and what the machine is currently doing.  The file
tools live in tools.py; this is everything else, kept apart because the two
have entirely different failure modes -- a file tool fails on a path, these
fail on a window that closed while you were talking about it.

**How an application gets found.**  `run_program("chrome")` only works if
chrome is on PATH, which on Windows it usually is not.  What people mean by an
app's name is the name on its Start Menu shortcut, so that is what is searched:
the two Start Menu trees, then the App Paths registry key that installers
write, then PATH.  Shortcuts are launched with `os.startfile`, which follows
the .lnk the way Explorer would -- no need to parse it or know where the
executable really lives.

**Keyboard input is different in kind from everything else here.**  Every
other tool does one bounded thing; `type_text` and `press_keys` can do anything
the person sitting here could do, including typing into a terminal.  Nothing
sandboxes them, and they are reachable by an agent that also reads web pages.
They are behind `tools_keyboard` in jarvis.json for that reason.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import shutil
import subprocess
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

# How many windows or apps a listing will name before it stops. A model does
# not need four hundred; it needs enough to pick one.
MAX_ROWS = 60

# Media and volume keys, by virtual key code.
MEDIA_KEYS = {"playpause": 0xB3, "next": 0xB0, "previous": 0xB1,
              "stop": 0xB2, "mute": 0xAD,
              "volumeup": 0xAF, "volumedown": 0xAE}

# The named keys `press_keys` understands, on top of single characters.
NAMED_KEYS = {
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B,
    "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
    "space": 0x20, "backspace": 0x08, "delete": 0x2E, "del": 0x2E,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
}

KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 0x0002, 0x0004
SW_RESTORE, SW_MINIMIZE, SW_MAXIMIZE = 9, 6, 3
WM_CLOSE = 0x0010

user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.PostMessageW.argtypes = [wintypes.HWND, ctypes.c_uint,
                                ctypes.c_size_t, ctypes.c_ssize_t]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.IsIconic.argtypes = [wintypes.HWND]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND,
                                            ctypes.POINTER(wintypes.DWORD)]
ENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [ENUMPROC, wintypes.LPARAM]


class ActionError(Exception):
    """A failure the model should be told about in words it can act on."""


# -- finding applications ---------------------------------------------------

def _shortcuts() -> dict[str, pathlib.Path]:
    """Every shortcut worth launching, by its lowercased name.

    Cached: this walks a couple of hundred directory entries, and the Start
    Menu does not change during a conversation.
    """
    if getattr(_shortcuts, "_cache", None) is not None:
        return _shortcuts._cache
    from .tools import ROOTS

    found: dict[str, pathlib.Path] = {}
    roots = [pathlib.Path(base) / "Microsoft/Windows/Start Menu/Programs"
             for base in (os.environ.get("ProgramData"), os.environ.get("APPDATA"))
             if base]
    # And every desktop, of which there are three: the user's (which OneDrive
    # may have moved), the original location it was moved from, which usually
    # still has things in it, and the shared one installers write to. Plenty
    # of what people launch daily never gets a Start Menu entry at all.
    roots.extend(r for r in ROOTS if r.name.lower() == "desktop")
    roots.append(pathlib.Path.home() / "Desktop")
    roots.append(pathlib.Path(os.environ.get("PUBLIC", r"C:\Users\Public"))
                 / "Desktop")
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for link in root.rglob("*.lnk"):
                found.setdefault(link.stem.lower(), link)
        except OSError:
            pass
    _shortcuts._cache = found
    return found


def _app_paths(name: str) -> pathlib.Path | None:
    """The App Paths registry entry an installer writes, if there is one."""
    try:
        import winreg

        stem = name if name.lower().endswith(".exe") else name + ".exe"
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                key = winreg.OpenKey(
                    hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion"
                          rf"\App Paths\{stem}")
            except OSError:
                continue
            with key:
                target, _ = winreg.QueryValueEx(key, "")
                path = pathlib.Path(target.strip('"'))
                if path.is_file():
                    return path
    except Exception:                                        # noqa: BLE001
        pass
    return None


def _match(name: str) -> tuple[str, pathlib.Path] | None:
    """Resolve what someone called an app to something launchable.

    Exact name first, then a name that starts with it, then one that contains
    it -- so "chrome" finds "Google Chrome" but "Google Chrome" is never beaten
    by "Google Chrome Canary".
    """
    wanted = (name or "").strip().lower()
    if not wanted:
        raise ActionError("Which application?")
    links = _shortcuts()
    if wanted in links:
        return name, links[wanted]
    for test in (str.startswith, str.__contains__):
        hits = sorted(k for k in links if test(k, wanted))
        if hits:
            return hits[0], links[hits[0]]
    registered = _app_paths(wanted)
    if registered is not None:
        return registered.stem, registered
    on_path = shutil.which(wanted)
    if on_path:
        return pathlib.Path(on_path).stem, pathlib.Path(on_path)
    return None


def open_app(name: str) -> str:
    """Start an application by the name it has in the Start Menu."""
    hit = _match(name)
    if hit is None:
        near = ", ".join(sorted(_shortcuts())[:12])
        raise ActionError(f"No application called '{name}'. Some that are "
                          f"installed: {near}. Use list_apps to see them all.")
    label, target = hit
    try:
        os.startfile(str(target))                            # noqa: S606
    except OSError as exc:
        raise ActionError(f"Could not start {label}: {exc}") from exc
    return f"Started {label}"


def list_apps(contains: str = "") -> str:
    """The applications on this machine, optionally filtered."""
    names = sorted(_shortcuts())
    if contains.strip():
        needle = contains.strip().lower()
        names = [n for n in names if needle in n]
    if not names:
        return f"No application matches '{contains}'."
    shown = names[:MAX_ROWS]
    more = f"\n... and {len(names) - len(shown)} more" if len(names) > len(shown) else ""
    return "\n".join(shown) + more


def open_url(url: str) -> str:
    """Open a web address in the default browser."""
    url = (url or "").strip()
    if not url:
        raise ActionError("A URL is required.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    # Through the browser rather than os.startfile, which would happily
    # "open" a file: or a shell: URL and run something local.
    import webbrowser

    if not webbrowser.open(url):
        raise ActionError(f"Nothing could open {url}.")
    return f"Opened {url}"


# -- windows ----------------------------------------------------------------

def _windows() -> list[tuple[int, str]]:
    """Every visible window with a title, front to back."""
    out: list[tuple[int, str]] = []

    def visit(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if title and title != "Program Manager":
            out.append((ctypes.cast(hwnd, ctypes.c_void_p).value, title))
        return True

    user32.EnumWindows(ENUMPROC(visit), 0)
    return out


def _find_window(title: str) -> tuple[int, str]:
    wanted = (title or "").strip().lower()
    if not wanted:
        raise ActionError("Which window?")
    rows = _windows()
    for test in (str.__eq__, str.startswith, str.__contains__):
        for hwnd, name in rows:
            if test(name.lower(), wanted):
                return hwnd, name
    raise ActionError(f"No open window matches '{title}'. "
                      f"Use list_windows to see what is open.")


def list_windows() -> str:
    """What is open, front to back."""
    rows = _windows()
    if not rows:
        return "Nothing is open."
    front = user32.GetForegroundWindow()
    front = ctypes.cast(front, ctypes.c_void_p).value
    lines = []
    for hwnd, title in rows[:MAX_ROWS]:
        mark = " (in front)" if hwnd == front else ""
        state = " (minimised)" if user32.IsIconic(wintypes.HWND(hwnd)) else ""
        lines.append(f"  {title}{state}{mark}")
    if len(rows) > MAX_ROWS:
        lines.append(f"  ... and {len(rows) - MAX_ROWS} more")
    return "\n".join(lines)


def focus_window(title: str) -> str:
    """Bring a window to the front, restoring it if it was minimised."""
    hwnd, name = _find_window(title)
    handle = wintypes.HWND(hwnd)
    if user32.IsIconic(handle):
        user32.ShowWindow(handle, SW_RESTORE)
    user32.SetForegroundWindow(handle)
    return f"Brought '{name}' to the front"


def window_state(title: str, state: str) -> str:
    """Minimise, maximise or restore a window."""
    modes = {"minimise": SW_MINIMIZE, "minimize": SW_MINIMIZE,
             "maximise": SW_MAXIMIZE, "maximize": SW_MAXIMIZE,
             "restore": SW_RESTORE}
    mode = modes.get((state or "").strip().lower())
    if mode is None:
        raise ActionError(f"state must be one of: {', '.join(sorted(modes))}")
    hwnd, name = _find_window(title)
    user32.ShowWindow(wintypes.HWND(hwnd), mode)
    return f"{state.capitalize()}d '{name}'"


def close_window(title: str) -> str:
    """Ask a window to close.

    Asks rather than kills: WM_CLOSE is the same message the X button sends,
    so an application with unsaved work still gets to put its dialog up
    instead of losing it.
    """
    hwnd, name = _find_window(title)
    user32.PostMessageW(wintypes.HWND(hwnd), WM_CLOSE, 0, 0)
    return f"Asked '{name}' to close"


# -- sound ------------------------------------------------------------------

def _volume():
    from .audio import Volume

    if getattr(_volume, "_cache", None) is None:
        _volume._cache = Volume()
    if not _volume._cache.available:
        raise ActionError("The system volume is not reachable on this machine.")
    return _volume._cache


def get_volume() -> str:
    volume = _volume()
    return (f"Volume is {round(volume.get() * 100)}%"
            + (" and muted" if volume.muted() else ""))


def set_volume(percent: float) -> str:
    volume = _volume()
    try:
        level = float(percent)
    except (TypeError, ValueError):
        raise ActionError("percent must be a number from 0 to 100.") from None
    volume.set(min(max(level, 0.0), 100.0) / 100.0)
    return f"Volume set to {round(min(max(level, 0.0), 100.0))}%"


def set_muted(muted: bool = True) -> str:
    volume = _volume()
    volume.set_muted(bool(muted))
    return "Muted" if muted else "Unmuted"


def media_key(action: str) -> str:
    """Play/pause, skip, or change the volume, as the keyboard's media keys do."""
    key = (action or "").strip().lower().replace(" ", "").replace("_", "")
    code = MEDIA_KEYS.get(key)
    if code is None:
        raise ActionError(f"action must be one of: {', '.join(sorted(MEDIA_KEYS))}")
    user32.keybd_event(code, 0, 0, 0)
    user32.keybd_event(code, 0, KEYEVENTF_KEYUP, 0)
    return f"Sent {key}"


# -- the clipboard ----------------------------------------------------------

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
user32.GetClipboardData.restype = wintypes.HANDLE
user32.GetClipboardData.argtypes = [ctypes.c_uint]
user32.SetClipboardData.restype = wintypes.HANDLE
user32.SetClipboardData.argtypes = [ctypes.c_uint, wintypes.HANDLE]
user32.OpenClipboard.argtypes = [wintypes.HWND]


def _with_clipboard(work):
    """Open the clipboard, do one thing, close it again.

    Retried: the clipboard is a single system-wide lock and any application
    can be holding it for a moment, which is a normal thing to wait out rather
    than an error to report.
    """
    for _ in range(10):
        if user32.OpenClipboard(None):
            try:
                return work()
            finally:
                user32.CloseClipboard()
        time.sleep(0.02)
    raise ActionError("Another application is holding the clipboard.")


def clipboard_get(max_chars: int = 8000) -> str:
    """What is on the clipboard, if it is text."""
    def read():
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        pointer = kernel32.GlobalLock(handle)
        try:
            return ctypes.c_wchar_p(pointer).value or ""
        finally:
            kernel32.GlobalUnlock(handle)

    text = _with_clipboard(read)
    if not text:
        return "The clipboard is empty, or does not hold text."
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def clipboard_set(text: str) -> str:
    """Put text on the clipboard."""
    payload = str(text or "")

    def write():
        user32.EmptyClipboard()
        size = (len(payload) + 1) * ctypes.sizeof(ctypes.c_wchar)
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        pointer = kernel32.GlobalLock(handle)
        ctypes.memmove(pointer, ctypes.create_unicode_buffer(payload), size)
        kernel32.GlobalUnlock(handle)
        user32.SetClipboardData(CF_UNICODETEXT, handle)
        return True

    _with_clipboard(write)
    return f"Copied {len(payload)} characters to the clipboard"


# -- looking at the machine -------------------------------------------------

def system_info() -> str:
    """Battery, memory, disks and how long this machine has been up."""
    try:
        import psutil
    except ImportError:
        raise ActionError("psutil is not installed.") from None
    lines = [f"CPU: {psutil.cpu_percent(interval=0.15):.0f}% across "
             f"{psutil.cpu_count(logical=True)} threads"]
    memory = psutil.virtual_memory()
    lines.append(f"Memory: {memory.percent:.0f}% of "
                 f"{memory.total / 1e9:.0f}GB used")
    battery = psutil.sensors_battery()
    if battery is not None:
        plug = "charging" if battery.power_plugged else "on battery"
        lines.append(f"Battery: {battery.percent:.0f}%, {plug}")
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        lines.append(f"Disk {part.device.rstrip(chr(92))} "
                     f"{usage.free / 1e9:.0f}GB free of {usage.total / 1e9:.0f}GB")
    up = time.time() - psutil.boot_time()
    lines.append(f"Up for {int(up // 3600)}h {int(up % 3600 // 60)}m")
    return "\n".join(lines)


def take_screenshot() -> tuple[str, bytes]:
    """A picture of the primary screen, as text and a PNG.

    Returned as an image block as well as a path, because "what is on my
    screen" is the question this exists for and a path alone cannot answer it.
    Whether the answering model actually sees the image depends on the
    provider FreeClaw picks; the path is there either way.
    """
    import cv2
    import numpy as np

    from ..mouse import primary_screen_size

    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    for lib, fn, restype, argtypes in (
            (user32, "GetDC", wintypes.HDC, [wintypes.HWND]),
            (gdi32, "CreateCompatibleDC", wintypes.HDC, [wintypes.HDC]),
            (gdi32, "CreateCompatibleBitmap", wintypes.HBITMAP,
             [wintypes.HDC, ctypes.c_int, ctypes.c_int]),
            (gdi32, "SelectObject", wintypes.HGDIOBJ,
             [wintypes.HDC, wintypes.HGDIOBJ])):
        getattr(lib, fn).restype = restype
        getattr(lib, fn).argtypes = argtypes
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    # Declared, not left to ctypes' defaults: a device context is a pointer,
    # and the default int conversion overflows the moment Windows hands back
    # a handle above 2^31, which on a 64-bit machine it routinely does.
    gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, wintypes.HDC,
                             ctypes.c_int, ctypes.c_int, wintypes.DWORD]
    gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, ctypes.c_uint,
                                ctypes.c_uint, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_uint]

    width, height = primary_screen_size()
    screen = user32.GetDC(None)
    dc = gdi32.CreateCompatibleDC(screen)
    bitmap = gdi32.CreateCompatibleBitmap(screen, width, height)
    old = gdi32.SelectObject(dc, bitmap)
    gdi32.BitBlt(dc, 0, 0, width, height, screen, 0, 0, 0x00CC0020)  # SRCCOPY

    class Head(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("width", wintypes.LONG),
                    ("height", wintypes.LONG), ("planes", wintypes.WORD),
                    ("bits", wintypes.WORD), ("compression", wintypes.DWORD),
                    ("image", wintypes.DWORD), ("xppm", wintypes.LONG),
                    ("yppm", wintypes.LONG), ("used", wintypes.DWORD),
                    ("important", wintypes.DWORD), ("pad", wintypes.DWORD * 3)]

    head = Head()
    head.size, head.width, head.height = 40, width, -height
    head.planes, head.bits, head.compression = 1, 32, 0
    shot = np.empty((height, width, 4), np.uint8)
    gdi32.GetDIBits(dc, bitmap, 0, height, shot.ctypes.data_as(ctypes.c_void_p),
                    ctypes.byref(head), 0)
    gdi32.SelectObject(dc, old)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(dc)
    user32.ReleaseDC(None, screen)

    from .tools import ROOTS

    folder = ROOTS[-1] / "jarvis-screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"screen_{time.strftime('%H%M%S')}.png"
    # Half size: a 1080p screenshot is a megabyte of base64 through the model's
    # context, and half of it is still legible.
    small = cv2.resize(shot[:, :, :3], (width // 2, height // 2))
    cv2.imwrite(str(path), small)
    return f"Screenshot of the {width}x{height} screen, saved to {path}", \
        cv2.imencode(".png", small)[1].tobytes()


# -- typing, which is the one that escapes the sandbox ----------------------

def type_text(text: str) -> str:
    """Type into whatever window has focus."""
    payload = str(text or "")
    if not payload:
        raise ActionError("Nothing to type.")
    for char in payload:
        # KEYEVENTF_UNICODE takes the character itself, so this needs no
        # keyboard layout and gets accents and symbols right.
        user32.keybd_event(0, ord(char), KEYEVENTF_UNICODE, 0)
        user32.keybd_event(0, ord(char), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0)
        time.sleep(0.002)
    return f"Typed {len(payload)} characters into the focused window"


def press_keys(combination: str) -> str:
    """Press a key combination, e.g. "ctrl+s" or "alt+tab"."""
    parts = [p.strip().lower() for p in (combination or "").split("+") if p.strip()]
    if not parts:
        raise ActionError("A key combination is required, e.g. ctrl+s.")
    codes = []
    for part in parts:
        if part in NAMED_KEYS:
            codes.append(NAMED_KEYS[part])
        elif len(part) == 1:
            code = user32.VkKeyScanW(ord(part))
            if code == -1:
                raise ActionError(f"Cannot press '{part}' on this keyboard.")
            codes.append(code & 0xFF)
        else:
            raise ActionError(f"Unknown key '{part}'. Named keys: "
                              f"{', '.join(sorted(NAMED_KEYS))}")
    for code in codes:
        user32.keybd_event(code, 0, 0, 0)
    for code in reversed(codes):                 # released in reverse, as a
        user32.keybd_event(code, 0, KEYEVENTF_KEYUP, 0)   # real hand would
    return f"Pressed {'+'.join(parts)}"


def lock_screen() -> str:
    """Lock the machine. Nothing is closed and nothing is lost."""
    if not ctypes.WinDLL("user32").LockWorkStation():
        raise ActionError("Windows refused to lock the screen.")
    return "Locked"


# name -> (function, description, {argument: (type, description, required)})
ACTIONS = {
    "open_app": (
        open_app, "Start an application by name, e.g. Chrome, Notepad, Spotify.",
        {"name": ("string", "What the app is called", True)}),
    "list_apps": (
        list_apps, "List the applications installed on this computer.",
        {"contains": ("string", "Only those whose name contains this", False)}),
    "open_url": (
        open_url, "Open a web address in the user's default browser.",
        {"url": ("string", "The address to open", True)}),
    "list_windows": (
        list_windows, "List the windows currently open, front to back.", {}),
    "focus_window": (
        focus_window, "Bring an open window to the front by its title.",
        {"title": ("string", "Part of the window's title", True)}),
    "window_state": (
        window_state, "Minimise, maximise or restore an open window.",
        {"title": ("string", "Part of the window's title", True),
         "state": ("string", "minimise, maximise or restore", True)}),
    "close_window": (
        close_window, "Ask an open window to close, as its X button would.",
        {"title": ("string", "Part of the window's title", True)}),
    "get_volume": (get_volume, "Read the system output volume.", {}),
    "set_volume": (
        set_volume, "Set the system output volume.",
        {"percent": ("number", "0 to 100", True)}),
    "set_muted": (
        set_muted, "Mute or unmute the system output.",
        {"muted": ("boolean", "True to mute", False)}),
    "media_key": (
        media_key, "Press a media key: playpause, next, previous, stop, mute.",
        {"action": ("string", "Which media key", True)}),
    "clipboard_get": (
        clipboard_get, "Read the text currently on the clipboard.",
        {"max_chars": ("integer", "Stop after this many characters", False)}),
    "clipboard_set": (
        clipboard_set, "Put text on the clipboard.",
        {"text": ("string", "What to copy", True)}),
    "take_screenshot": (
        take_screenshot, "Capture what is currently on the user's screen.", {}),
    "lock_screen": (lock_screen, "Lock the computer.", {}),
}

# Kept out of ACTIONS so the server can add them only when they are allowed.
KEYBOARD_ACTIONS = {
    "type_text": (
        type_text, "Type text into whatever window is focused on the user's "
                   "computer. Check with them before typing anywhere that "
                   "runs commands.",
        {"text": ("string", "What to type", True)}),
    "press_keys": (
        press_keys, "Press a key combination on the user's computer, "
                    "e.g. ctrl+s or alt+tab.",
        {"combination": ("string", "Keys joined by +", True)}),
}
