"""Driving the real Windows cursor through SendInput.

SendInput is used rather than the older mouse_event or SetCursorPos because it
feeds the same input queue as a physical mouse, so applications that read raw
input (games, remote desktop, some drawing apps) react to it normally.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

# GetSystemMetrics indices
SM_CXSCREEN, SM_CYSCREEN = 0, 1
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
WHEEL_DELTA = 120

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def enable_dpi_awareness() -> None:
    """Ask Windows for real pixels.

    Without this a scaled display (150% is the Windows default on a laptop)
    reports a virtual, smaller desktop, and every coordinate computed from it
    lands in the wrong place.  Newest API first, older ones as fallbacks.
    """
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except AttributeError:
        pass
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
        return
    except (OSError, AttributeError):
        pass
    try:
        user32.SetProcessDPIAware()
    except AttributeError:
        pass


def primary_screen_size() -> tuple[int, int]:
    return user32.GetSystemMetrics(SM_CXSCREEN), user32.GetSystemMetrics(SM_CYSCREEN)


def virtual_screen_rect() -> tuple[int, int, int, int]:
    """(left, top, width, height) of the whole desktop, all monitors included."""
    return (user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))


SM_CXDOUBLECLK, SM_CYDOUBLECLK = 36, 37


def double_click_time_ms() -> float:
    """How long Windows allows between two clicks of a double click."""
    return float(user32.GetDoubleClickTime())


def double_click_slop() -> tuple[int, int]:
    """How far apart those two clicks may land, in pixels.

    Typically four, which is far tighter than a hand can repeat -- hence the
    assist in the controller that reuses the first click's exact position.
    """
    return (user32.GetSystemMetrics(SM_CXDOUBLECLK),
            user32.GetSystemMetrics(SM_CYDOUBLECLK))


def cursor_position() -> tuple[int, int]:
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return int(pt.x), int(pt.y)


def _send(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> None:
    inp = INPUT(type=INPUT_MOUSE,
                mi=MOUSEINPUT(dx=dx, dy=dy, mouseData=ctypes.c_uint32(data).value,
                              dwFlags=flags, time=0, dwExtraInfo=0))
    sent = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise ctypes.WinError(ctypes.get_last_error())


class Mouse:
    """Absolute-positioned mouse control over the full virtual desktop."""

    def __init__(self) -> None:
        enable_dpi_awareness()
        self.refresh_geometry()
        self._left_down = False
        self._scroll_remainder = 0.0

    def refresh_geometry(self) -> None:
        self.vleft, self.vtop, self.vwidth, self.vheight = virtual_screen_rect()
        self.width, self.height = primary_screen_size()

    def move_to(self, x: float, y: float) -> None:
        """Move to a pixel on the primary display (clamped to the desktop)."""
        # SendInput absolute coordinates are 0..65535 spanning the virtual
        # desktop, so a primary-monitor pixel has to be re-expressed in it.
        vx = min(max(x, self.vleft), self.vleft + self.vwidth - 1)
        vy = min(max(y, self.vtop), self.vtop + self.vheight - 1)
        nx = int(round((vx - self.vleft) * 65535.0 / max(self.vwidth - 1, 1)))
        ny = int(round((vy - self.vtop) * 65535.0 / max(self.vheight - 1, 1)))
        _send(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
              dx=nx, dy=ny)

    # -- buttons -----------------------------------------------------------

    def left_down(self) -> None:
        if not self._left_down:
            _send(MOUSEEVENTF_LEFTDOWN)
            self._left_down = True

    def left_up(self) -> None:
        if self._left_down:
            _send(MOUSEEVENTF_LEFTUP)
            self._left_down = False

    @property
    def left_is_down(self) -> bool:
        return self._left_down

    def click(self) -> None:
        self.left_down()
        self.left_up()

    def right_click(self) -> None:
        _send(MOUSEEVENTF_RIGHTDOWN)
        _send(MOUSEEVENTF_RIGHTUP)

    def middle_click(self) -> None:
        _send(MOUSEEVENTF_MIDDLEDOWN)
        _send(MOUSEEVENTF_MIDDLEUP)

    def scroll(self, notches: float) -> None:
        """Scroll by a fractional number of wheel notches.

        Fractions are accumulated instead of discarded, so slow hand movement
        still scrolls rather than rounding to nothing every frame.
        """
        self._scroll_remainder += notches
        whole = int(self._scroll_remainder)
        if whole:
            self._scroll_remainder -= whole
            _send(MOUSEEVENTF_WHEEL, data=whole * WHEEL_DELTA)

    def release_all(self) -> None:
        self.left_up()
        self._scroll_remainder = 0.0
