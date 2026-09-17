"""The small clickable buttons in the corner: gestures, reset, and quit.

The overlay itself cannot hold a button.  `WS_EX_TRANSPARENT` is what stops it
swallowing the clicks the app makes with the cursor it is driving, and that
flag is per-window -- there is no way to carve one clickable rectangle out of
a click-through window.  So each button is its own window: tiny, and among the
only parts of the whole interface that answer the mouse.

Pinned under the applications, the same as the overlay, so they belong to the
same layer rather than floating over everything: open a browser and the
browser covers them.  That does mean they are only clickable when nothing is
on top -- `Ctrl+Alt+M` and `Ctrl+Alt+Q` are the way to reach the two important
ones the rest of the time, and are why the keyboard routes exist at all.

They sink to the bottom once, where the overlay sinks every frame, so the
overlay ends up underneath them without either having to know about the other.

`Button` is the window and the paint; `Toggle` is the one that holds a state
and changes its own label.  They are the same code because they have to look
identical -- a strip where one button is drawn by different arithmetic from
the one beside it never quite lines up, and the misalignment is the sort of
thing you see without being able to say why.
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

WS_POPUP = 0x80000000
WS_EX_TOOLWINDOW = 0x00000080         # no taskbar button, no Alt+Tab entry
WS_EX_NOACTIVATE = 0x08000000         # clicking it does not steal focus

SW_SHOWNOACTIVATE = 4
HWND_BOTTOM = ctypes.c_void_p(1)
SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = 0x0002, 0x0001, 0x0010

WM_DESTROY, WM_PAINT, WM_LBUTTONUP = 0x0002, 0x000F, 0x0202
WM_MOUSEMOVE, WM_MOUSELEAVE, WM_ERASEBKGND = 0x0200, 0x02A3, 0x0014
DT_CENTER, DT_VCENTER, DT_SINGLELINE = 0x1, 0x4, 0x20
TRANSPARENT = 1

WIDTH, HEIGHT = 132, 34
# The narrower buttons that sit beside it.
SMALL_WIDTH = 74
# How far the strip sits in from the corner of the screen, and the gap
# between one button and the next.
MARGIN = 10
SPACING = 6

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, ctypes.c_uint,
                             ctypes.c_size_t, ctypes.c_ssize_t)

# BGR, the order Windows wants a COLORREF in.
ON_FACE, ON_TEXT = 0x120C06, 0xFFAA2C        # cyan on near-black: driving
OFF_FACE, OFF_TEXT = 0x1E1712, 0x9A8874      # muted: hands off the mouse
EDGE = 0x54361A
# The quit button, in the same red the session log uses for errors. Not
# because it is dangerous -- it closes an app -- but because it is the one
# button in the strip you do not want to hit by accident, and colour is the
# only thing that makes a 74-pixel rectangle distinguishable at a glance.
QUIT_FACE, QUIT_TEXT = 0x0D0A20, 0x7878FF


class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [("hdc", wintypes.HDC), ("fErase", wintypes.BOOL),
                ("rcPaint", wintypes.RECT), ("fRestore", wintypes.BOOL),
                ("fIncremental", wintypes.BOOL), ("rgbReserved", ctypes.c_byte * 32)]


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("style", ctypes.c_uint),
                ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON), ("hCursor", wintypes.HICON),
                ("hbrBackground", wintypes.HBRUSH), ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR), ("hIconSm", wintypes.HICON)]


user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
user32.DefWindowProcW.restype = LRESULT
user32.DefWindowProcW.argtypes = [wintypes.HWND, ctypes.c_uint,
                                  ctypes.c_size_t, ctypes.c_ssize_t]
user32.BeginPaint.restype = wintypes.HDC
user32.BeginPaint.argtypes = [wintypes.HWND, ctypes.POINTER(PAINTSTRUCT)]
user32.EndPaint.argtypes = [wintypes.HWND, ctypes.POINTER(PAINTSTRUCT)]
user32.FillRect.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                            wintypes.HBRUSH]
user32.DrawTextW.argtypes = [wintypes.HDC, wintypes.LPCWSTR, ctypes.c_int,
                             ctypes.POINTER(wintypes.RECT), ctypes.c_uint]
user32.InvalidateRect.argtypes = [wintypes.HWND, ctypes.c_void_p, wintypes.BOOL]
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_uint]
user32.DestroyWindow.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.LoadCursorW.restype = wintypes.HICON
gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
gdi32.CreateSolidBrush.argtypes = [wintypes.COLORREF]
gdi32.SetTextColor.argtypes = [wintypes.HDC, wintypes.COLORREF]
gdi32.SetBkMode.argtypes = [wintypes.HDC, ctypes.c_int]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.CreateFontW.restype = wintypes.HFONT
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]

# Every button needs a window class of its own, because a class name is
# registered once per process and two windows sharing one would share the
# first one's handler -- so the second button's clicks would run the first
# button's action.
_serial = 0


class Button:
    """One small clickable window. `available` is False if it was not created.

    Owns a thread, because a window's messages must be pumped by the thread
    that created it and the overlay's loop has its own pace to keep.
    """

    def __init__(self, x: int, y: int, label: str, on_click=None,
                 width: int = SMALL_WIDTH, height: int = HEIGHT,
                 face: int = OFF_FACE, ink: int = OFF_TEXT,
                 pip: bool = False, name: str = "button") -> None:
        global _serial
        _serial += 1
        self.available = False
        self.unavailable_reason = ""
        self.label = label
        self.on_click = on_click
        self.face, self.ink, self.pip = face, ink, pip
        self.width, self.height = int(width), int(height)
        self._name = name
        self._class = f"GestureControlButton{_serial}"
        self._hwnd = None
        self._x, self._y = int(x), int(y)
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run,
                                        name=f"jarvis-{name}", daemon=True)
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
        # A plain blocking message loop. This thread does nothing else.
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
        cls.lpszClassName = self._class
        user32.RegisterClassExW(ctypes.byref(cls))
        self._hwnd = user32.CreateWindowExW(
            WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            cls.lpszClassName, "Gesture Control", WS_POPUP,
            self._x, self._y, self.width, self.height, None, None,
            wintypes.HINSTANCE(0), None)
        if not self._hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self._font = gdi32.CreateFontW(
            -13, 0, 0, 0, 600, 0, 0, 0, 0, 0, 0, 0, 0, "Segoe UI")
        user32.ShowWindow(self._hwnd, SW_SHOWNOACTIVATE)
        self.sink()

    def _wndproc(self, hwnd, message, wparam, lparam):
        if message == WM_LBUTTONUP:
            self.clicked()
            return 0
        if message == WM_ERASEBKGND:
            return 1                             # painted whole in WM_PAINT
        if message == WM_PAINT:
            self._paint(hwnd)
            return 0
        if message == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, message, ctypes.c_size_t(wparam),
                                     ctypes.c_ssize_t(lparam))

    def clicked(self) -> None:
        """What a click does. Subclasses that hold a state override this."""
        self.sink()
        if self.on_click is not None:
            try:
                self.on_click()
            except Exception:                                    # noqa: BLE001
                pass         # a button's action must not kill its message loop

    def face_and_ink(self) -> tuple[int, int]:
        """The colours to paint with right now."""
        return self.face, self.ink

    def text(self) -> str:
        return self.label

    def _paint(self, hwnd) -> None:
        ps = PAINTSTRUCT()
        hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
        face, ink = self.face_and_ink()
        width, height = self.width, self.height
        border = gdi32.CreateSolidBrush(wintypes.COLORREF(EDGE))
        inside = gdi32.CreateSolidBrush(wintypes.COLORREF(face))
        user32.FillRect(hdc, ctypes.byref(wintypes.RECT(0, 0, width, height)),
                        border)
        user32.FillRect(hdc, ctypes.byref(wintypes.RECT(1, 1, width - 1,
                                                        height - 1)), inside)
        left = 6
        if self.pip:
            # A filled pip: a state has to be readable at a glance, and the
            # label alone is ambiguous about which way it is describing.
            dot = gdi32.CreateSolidBrush(wintypes.COLORREF(ink))
            user32.FillRect(hdc, ctypes.byref(wintypes.RECT(
                9, height // 2 - 4, 17, height // 2 + 4)), dot)
            gdi32.DeleteObject(dot)
            left = 24
        gdi32.SetBkMode(hdc, TRANSPARENT)
        gdi32.SetTextColor(hdc, wintypes.COLORREF(ink))
        old = gdi32.SelectObject(hdc, self._font)
        box = wintypes.RECT(left, 0, width - 6, height)
        user32.DrawTextW(hdc, self.text(), -1, ctypes.byref(box),
                         DT_CENTER | DT_VCENTER | DT_SINGLELINE)
        gdi32.SelectObject(hdc, old)
        gdi32.DeleteObject(border)
        gdi32.DeleteObject(inside)
        user32.EndPaint(hwnd, ctypes.byref(ps))

    def repaint(self) -> None:
        if self._hwnd:
            user32.InvalidateRect(self._hwnd, None, True)

    def sink(self) -> None:
        """Drop to the bottom of the z-order, under every application window.

        Clicking a window normally brings it forward; these are not meant to
        come forward, so every click puts them back.
        """
        if self._hwnd:
            user32.SetWindowPos(self._hwnd, HWND_BOTTOM, 0, 0, 0, 0,
                                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)

    def close(self) -> None:
        if self._hwnd:
            user32.PostMessageW(self._hwnd, WM_DESTROY, 0, 0)
            self._hwnd = None
        self.available = False


class Toggle(Button):
    """The button that hands the mouse back.

    Two states.  **Gestures** -- the hand drives the cursor.  **Mouse** -- the
    hand is still tracked and the gauntlet still drawn, but nothing is sent to
    the mouse, so your real mouse has the screen to itself.  Ctrl+Alt+M does
    the same thing from the keyboard, which is the way back if this window is
    ever lost.
    """

    def __init__(self, x: int, y: int, on_change=None) -> None:
        self.gestures = True                     # True = the hand drives
        self.on_change = on_change
        super().__init__(x, y, "GESTURES ON", width=WIDTH, pip=True,
                         name="toggle button")

    def clicked(self) -> None:
        self.set(not self.gestures)

    def face_and_ink(self) -> tuple[int, int]:
        return (ON_FACE, ON_TEXT) if self.gestures else (OFF_FACE, OFF_TEXT)

    def text(self) -> str:
        return "GESTURES ON" if self.gestures else "MOUSE ONLY"

    def set(self, gestures: bool) -> None:
        """Turn gesture control on or off, and repaint."""
        gestures = bool(gestures)
        if gestures == self.gestures:
            return
        self.gestures = gestures
        self.repaint()
        self.sink()
        if self.on_change is not None:
            self.on_change(gestures)


class Strip:
    """The row of buttons in the corner, laid out as one thing.

    Built here rather than in the controller so that the arithmetic lives next
    to the widths it uses. The order is deliberate: the toggle is flush into
    the corner because it is the one that gets used, and QUIT is at the far
    end because it is the one you least want to hit while reaching for
    something else.
    """

    def __init__(self, screen_width: int, on_gestures=None, on_reset=None,
                 on_quit=None) -> None:
        self.notes: list[str] = []
        right = screen_width - MARGIN
        x = right - WIDTH
        self.toggle = Toggle(x, MARGIN, on_change=on_gestures)
        x -= SMALL_WIDTH + SPACING
        self.reset = Button(x, MARGIN, "RESET", on_click=on_reset,
                            name="reset button")
        x -= SMALL_WIDTH + SPACING
        self.quit = Button(x, MARGIN, "QUIT", on_click=on_quit,
                           face=QUIT_FACE, ink=QUIT_TEXT, name="quit button")
        self.left = x
        for button in self.all:
            if not button.available:
                self.notes.append(button.unavailable_reason)

    @property
    def all(self) -> list[Button]:
        return [self.toggle, self.reset, self.quit]

    @property
    def available(self) -> bool:
        """True if the toggle opened -- it is the one with a keyboard route."""
        return self.toggle.available

    @property
    def bottom(self) -> int:
        return MARGIN + HEIGHT

    def sink(self) -> None:
        for button in self.all:
            button.sink()

    def close(self) -> None:
        for button in self.all:
            button.close()
