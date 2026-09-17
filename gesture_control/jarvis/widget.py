"""A see-through panel you can actually touch.

The overlay draws holograms beautifully and cannot be clicked -- it is
`WS_EX_TRANSPARENT`, which is what stops it swallowing the clicks the app makes
with the cursor it is driving.  The toggle button can be clicked and is drawn
with flat GDI rectangles.  A volume slider needs both halves at once: the
translucent look of the one and the mouse of the other.

So this is a layered window with per-pixel alpha, like the overlay, but
*without* the click-through flag -- the one window in the app that answers the
mouse on purpose.  Content is a BGRA numpy array the owner redraws whenever it
likes; `UpdateLayeredWindow` takes it straight from the DIB the array is a view
of, so there is no copy between drawing and compositing.

It sits under the applications with the rest of the interface, and it never
takes focus, so dragging the slider does not pull you out of what you were
typing in.

Mouse events are handed back as (event, x, y) in the window's own pixels, on
the window's thread.  Whatever consumes them must be quick and must not touch
anything the draw loop owns without a lock.  The one exception to the pixel
rule is "wheel", whose y is a signed number of notches, positive away from
you.

**About the wheel.**  `WM_MOUSEWHEEL` goes to the *focused* window, and this
window is `WS_EX_NOACTIVATE` and so is never focused.  It arrives here anyway
because Windows 10 and 11 ship with "scroll inactive windows when I hover over
them" on, which routes the wheel to whatever is under the pointer instead.
That setting can be turned off, and there is no way to get the message back
without a system-wide low-level mouse hook -- which would put a Python callback
in the path of every mouse event on the machine, and this app is one whose
whole purpose is a cursor that moves smoothly.  So the wheel is the convenient
way to scroll, not the only one: anything scrollable here also has to answer
a drag.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Callable
from ctypes import wintypes

import numpy as np

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080

SW_SHOWNOACTIVATE = 4
HWND_BOTTOM = ctypes.c_void_p(1)
SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = 0x0002, 0x0001, 0x0010

ULW_ALPHA = 0x00000002
AC_SRC_OVER, AC_SRC_ALPHA = 0x00, 0x01
BI_RGB, DIB_RGB_COLORS = 0, 0

WM_DESTROY, WM_LBUTTONDOWN, WM_LBUTTONUP = 0x0002, 0x0201, 0x0202
WM_MOUSEMOVE, WM_MOUSEWHEEL = 0x0200, 0x020A
WHEEL_DELTA = 120                # one notch, per the wheel's own documentation

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, ctypes.c_uint,
                             ctypes.c_size_t, ctypes.c_ssize_t)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("style", ctypes.c_uint),
                ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON), ("hCursor", wintypes.HICON),
                ("hbrBackground", wintypes.HBRUSH), ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR), ("hIconSm", wintypes.HICON)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte),
                ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_ubyte)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
user32.DefWindowProcW.restype = LRESULT
user32.DefWindowProcW.argtypes = [wintypes.HWND, ctypes.c_uint,
                                  ctypes.c_size_t, ctypes.c_ssize_t]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_uint]
user32.DestroyWindow.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetCapture.argtypes = [wintypes.HWND]
user32.ScreenToClient.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.LoadCursorW.restype = wintypes.HICON
user32.UpdateLayeredWindow.argtypes = [
    wintypes.HWND, wintypes.HDC, ctypes.POINTER(wintypes.POINT),
    ctypes.POINTER(wintypes.SIZE), wintypes.HDC, ctypes.POINTER(wintypes.POINT),
    wintypes.DWORD, ctypes.POINTER(BLENDFUNCTION), wintypes.DWORD]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, ctypes.c_uint,
                                   ctypes.POINTER(ctypes.c_void_p),
                                   wintypes.HANDLE, wintypes.DWORD]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]


def premultiply(colour: np.ndarray, alpha: np.ndarray, into: np.ndarray) -> None:
    """Pack BGR + alpha into the BGRA a layered window expects.

    `UpdateLayeredWindow` blends colour it assumes is already multiplied by
    its own alpha; skipping that haloes every edge against whatever is behind.
    """
    import cv2

    wide = cv2.merge((alpha, alpha, alpha))
    cv2.mixChannels([cv2.multiply(colour, wide, scale=1.0 / 255.0), alpha],
                    [into], (0, 0, 1, 1, 2, 2, 3, 3))


class HoloWindow:
    """A translucent, clickable panel. `available` is False if it won't open."""

    def __init__(self, x: int, y: int, width: int, height: int,
                 on_mouse: Callable[[str, int, int], None] | None = None,
                 name: str = "GestureControlWidget") -> None:
        self.available = False
        self.unavailable_reason = ""
        self.width, self.height = int(width), int(height)
        self.on_mouse = on_mouse
        self.bgra: np.ndarray | None = None
        self._name = name
        self._x, self._y = int(x), int(y)
        self._hwnd = None
        self._down = False
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="jarvis-widget",
                                        daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    # -- the window --------------------------------------------------------

    def _run(self) -> None:
        try:
            self._create()
            self.available = True
        except OSError as exc:
            self.unavailable_reason = f"Could not create the {self._name} ({exc})."
            self._ready.set()
            return
        self._ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def _create(self) -> None:
        self._proc = WNDPROC(self._wndproc)      # kept alive: Windows calls it
        cls = WNDCLASSEXW()
        cls.cbSize = ctypes.sizeof(WNDCLASSEXW)
        cls.lpfnWndProc = self._proc
        cls.hInstance = wintypes.HINSTANCE(0)
        cls.hCursor = user32.LoadCursorW(None, ctypes.c_wchar_p(32649))  # IDC_HAND
        cls.lpszClassName = self._name
        user32.RegisterClassExW(ctypes.byref(cls))
        # Layered for the alpha, but deliberately *not* WS_EX_TRANSPARENT:
        # this is the one surface meant to answer the mouse.
        self._hwnd = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
            cls.lpszClassName, "Gesture Control", WS_POPUP,
            self._x, self._y, self.width, self.height, None, None,
            wintypes.HINSTANCE(0), None)
        if not self._hwnd:
            raise ctypes.WinError(ctypes.get_last_error())

        head = BITMAPINFOHEADER()
        head.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        head.biWidth, head.biHeight = self.width, -self.height   # top-down
        head.biPlanes, head.biBitCount, head.biCompression = 1, 32, BI_RGB
        bits = ctypes.c_void_p()
        screen = user32.GetDC(None)
        self._dc = gdi32.CreateCompatibleDC(screen)
        self._bitmap = gdi32.CreateDIBSection(screen, ctypes.byref(head),
                                              DIB_RGB_COLORS, ctypes.byref(bits),
                                              None, 0)
        user32.ReleaseDC(None, screen)
        if not self._bitmap:
            raise ctypes.WinError(ctypes.get_last_error())
        self._old = gdi32.SelectObject(self._dc, self._bitmap)
        buf = (ctypes.c_uint8 * (self.width * self.height * 4)).from_address(bits.value)
        self.bgra = np.ctypeslib.as_array(buf).reshape(self.height, self.width, 4)
        self._pos = wintypes.POINT(self._x, self._y)
        self._origin = wintypes.POINT(0, 0)
        self._size = wintypes.SIZE(self.width, self.height)
        self._blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        user32.ShowWindow(self._hwnd, SW_SHOWNOACTIVATE)
        self.sink()

    def _wndproc(self, hwnd, message, wparam, lparam):
        if message in (WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MOUSEMOVE):
            # lParam packs y in the high word and x in the low, both signed --
            # a drag that leaves the window to the left reports a negative x,
            # and reading it unsigned would jump the slider to the far end.
            x = ctypes.c_short(lparam & 0xFFFF).value
            y = ctypes.c_short((lparam >> 16) & 0xFFFF).value
            if message == WM_LBUTTONDOWN:
                self._down = True
                # Capture, so a drag keeps arriving here after the pointer
                # leaves the window. Without it the slider sticks the moment
                # you overshoot the panel.
                user32.SetCapture(hwnd)
                self._emit("down", x, y)
            elif message == WM_LBUTTONUP:
                self._down = False
                user32.ReleaseCapture()
                self._emit("up", x, y)
            elif self._down:
                self._emit("drag", x, y)
            return 0
        if message == WM_MOUSEWHEEL:
            # Unlike every other mouse message, this one carries *screen*
            # coordinates -- reading them as client ones would report a
            # position off the panel on any monitor but the primary one.
            point = wintypes.POINT(ctypes.c_short(lparam & 0xFFFF).value,
                                   ctypes.c_short((lparam >> 16) & 0xFFFF).value)
            user32.ScreenToClient(hwnd, ctypes.byref(point))
            notches = ctypes.c_short((wparam >> 16) & 0xFFFF).value // WHEEL_DELTA
            self._emit("wheel", point.x, notches)
            return 0
        if message == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, message, ctypes.c_size_t(wparam),
                                     ctypes.c_ssize_t(lparam))

    def _emit(self, event: str, x: int, y: int) -> None:
        if self.on_mouse is None:
            return
        try:
            self.on_mouse(event, x, y)
        except Exception:                                        # noqa: BLE001
            pass          # a widget bug must not kill the message loop

    # -- drawing -----------------------------------------------------------

    def blit(self) -> None:
        """Push whatever is in `bgra` to the screen."""
        if not self._hwnd:
            return
        screen = user32.GetDC(None)
        user32.UpdateLayeredWindow(self._hwnd, screen, ctypes.byref(self._pos),
                                   ctypes.byref(self._size), self._dc,
                                   ctypes.byref(self._origin), 0,
                                   ctypes.byref(self._blend), ULW_ALPHA)
        user32.ReleaseDC(None, screen)

    def sink(self) -> None:
        """Drop under every application window, where the rest of this lives."""
        if self._hwnd:
            user32.SetWindowPos(self._hwnd, HWND_BOTTOM, 0, 0, 0, 0,
                                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)

    def close(self) -> None:
        self.bgra = None
        if self._hwnd:
            user32.PostMessageW(self._hwnd, WM_DESTROY, 0, 0)
            self._hwnd = None
        if getattr(self, "_dc", None):
            if getattr(self, "_old", None):
                gdi32.SelectObject(self._dc, self._old)
            gdi32.DeleteDC(self._dc)
            self._dc = None
        if getattr(self, "_bitmap", None):
            gdi32.DeleteObject(self._bitmap)
            self._bitmap = None
        self.available = False
