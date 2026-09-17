"""A full-screen window you can see through, click through, and work under.

The preview used to be an ordinary OpenCV window: a rectangle of webcam sitting
on top of whatever you were trying to click.  That is backwards for something
whose whole job is to move the real cursor.  This draws the same picture over
the entire screen instead, and gets out of the way three different times:

  *see* through   Per-pixel alpha, so the camera image is a faint wash while
                  the gauntlet and the readouts stay solid.  `dim` is the
                  camera's share; at 0 the glove floats over the desktop on
                  its own.
  *click* through WS_EX_TRANSPARENT.  Hit-testing never stops here, so the
                  click the app just sent lands on the window underneath.
                  Without this the overlay would swallow every click it made.
  *stay* under    Pinned to the bottom of the z-order, so it covers the
                  wallpaper and the desktop icons and nothing else.  Open a
                  browser and the browser is in front.

Fullscreen also fixes the direct map: at 1:1 the hand in the overlay is drawn
exactly where the cursor it is driving goes, so the correspondence is something
you can see rather than something to take on trust.

Because the window can never be focused (WS_EX_NOACTIVATE -- focusing it would
steal the keystroke you meant for the window beneath), keys arrive as system
hotkeys instead.  They are all Ctrl+Alt+<letter>: a bare letter would fire
while you were typing in some other app.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

import cv2
import numpy as np

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020        # click-through
WS_EX_NOACTIVATE = 0x08000000         # never takes focus
WS_EX_TOOLWINDOW = 0x00000080         # keep it out of Alt+Tab and the taskbar

SW_SHOWNOACTIVATE = 4
HWND_BOTTOM = ctypes.c_void_p(1)
SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = 0x0002, 0x0001, 0x0010

ULW_ALPHA = 0x00000002
AC_SRC_OVER, AC_SRC_ALPHA = 0x00, 0x01
BI_RGB = 0
DIB_RGB_COLORS = 0

MOD_ALT, MOD_CONTROL, MOD_NOREPEAT = 0x0001, 0x0002, 0x4000
WM_HOTKEY = 0x0312
PM_REMOVE = 0x0001

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, ctypes.c_uint,
                             ctypes.c_size_t, ctypes.c_ssize_t)

# The keys the overlay listens for, as Ctrl+Alt+<letter>.
#   q quit   p pause   s snapping   k on-screen keyboard   d dump a capture
#   j add/relink FreeClaw   m hand the mouse back (same as the toggle button)
HOTKEYS = ("q", "p", "s", "k", "d", "j", "m")


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
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_uint]
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
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.DestroyWindow.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.UpdateLayeredWindow.argtypes = [
    wintypes.HWND, wintypes.HDC, ctypes.POINTER(wintypes.POINT),
    ctypes.POINTER(wintypes.SIZE), wintypes.HDC, ctypes.POINTER(wintypes.POINT),
    wintypes.DWORD, ctypes.POINTER(BLENDFUNCTION), wintypes.DWORD]

# How different a pixel has to be from the bare camera frame before it counts
# as something the app drew.  Camera noise moves a pixel by a few levels even
# when nothing is happening, so the ramp starts above that and is fully opaque
# well before the gauntlet's own soft edges are, which leaves those edges
# feathering into the wash instead of ending on a hard line.
INK_FLOOR, INK_FULL = 14.0, 52.0


class Overlay:
    """The screen-sized layered window.  `available` is False if it won't open."""

    def __init__(self, size: tuple[int, int], dim: float = 0.32) -> None:
        self.available = False
        self.dim = float(min(max(dim, 0.0), 1.0))
        self.width, self.height = int(size[0]), int(size[1])
        self._hwnd = None
        self._pressed: list[str] = []
        try:
            self._open()
            self.available = True
        except OSError as exc:                 # no window station, no GDI, ...
            self.unavailable_reason = f"Could not open the overlay window ({exc})."
            self.close()

    # -- setup -------------------------------------------------------------

    def _open(self) -> None:
        hinst = wintypes.HINSTANCE(0)
        # Kept on the instance: Windows calls this back, and a callback that
        # has been garbage collected crashes the process rather than raising.
        self._wndproc = WNDPROC(
            lambda h, m, w, l: user32.DefWindowProcW(h, m, ctypes.c_size_t(w),
                                                     ctypes.c_ssize_t(l)))
        cls = WNDCLASSEXW()
        cls.cbSize = ctypes.sizeof(WNDCLASSEXW)
        cls.lpfnWndProc = self._wndproc
        cls.hInstance = hinst
        cls.lpszClassName = "GestureControlOverlay"
        # A second run in the same process would fail on the duplicate class
        # name; that is harmless, the class is already there.
        user32.RegisterClassExW(ctypes.byref(cls))

        self._hwnd = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
            cls.lpszClassName, "Gesture Control", WS_POPUP,
            0, 0, self.width, self.height, None, None, hinst, None)
        if not self._hwnd:
            raise ctypes.WinError(ctypes.get_last_error())

        # A 32-bit top-down DIB, addressed straight as a numpy array: the frame
        # is written into the bitmap Windows composites from, with no copy in
        # between.  Negative height is what makes it top-down, so row 0 is the
        # top row and the array matches OpenCV's without a flip.
        head = BITMAPINFOHEADER()
        head.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        head.biWidth, head.biHeight = self.width, -self.height
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
        # Every intermediate, allocated once.  At screen size these are six
        # megabytes apiece, and letting OpenCV allocate them per frame cost
        # more than all the arithmetic done in them put together.  The mask's
        # three are sized to whatever the frame turns out to be, which is not
        # known here, so _alpha_for allocates those on its first sight of one.
        # Everything here is sized to the *frame*, not to the screen, and the
        # frame is usually a crop about a third the area -- so these are
        # allocated on first sight of one, in _buffers_for.
        self._premul = self._wide = self._small = None
        self._diff = self._ink = self._alpha = None
        base = round(self.dim * 255)
        steps = np.arange(256, dtype=np.float32)
        lift = np.clip((steps - INK_FLOOR) / (INK_FULL - INK_FLOOR), 0.0, 1.0)
        self._ramp = (base + (255 - base) * lift).astype(np.uint8)
        self._pos = wintypes.POINT(0, 0)
        self._origin = wintypes.POINT(0, 0)
        self._size = wintypes.SIZE(self.width, self.height)
        self._blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)

        for i, key in enumerate(HOTKEYS):
            user32.RegisterHotKey(None, i, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT,
                                  ord(key.upper()))
        user32.ShowWindow(self._hwnd, SW_SHOWNOACTIVATE)
        self._sink()

    @property
    def hwnd(self):
        """The window handle, for anything that must stack against it."""
        return self._hwnd

    def _sink(self) -> None:
        """Put the window back at the bottom of the z-order."""
        user32.SetWindowPos(self._hwnd, HWND_BOTTOM, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)

    # -- per frame ---------------------------------------------------------

    def show(self, frame: np.ndarray, clean: np.ndarray | None = None,
             solid: np.ndarray | None = None) -> None:
        """Put `frame` on the screen, fading back towards the bare camera.

        `clean` is the same frame before the app drew on it.  Where the two
        agree nothing was drawn there, so it is camera and fades to `dim`;
        where they differ the app put something there and it stays solid.
        Deriving the mask this way rather than having every drawing call
        maintain one means nothing can be drawn and then forgotten about.

        `solid` is the exception: a per-pixel *floor* on the alpha, for the
        parts of the interface that have to be opaque whatever the camera
        happens to be showing.  Inference is not enough for those.  Darkening
        a region only differs from the bare camera by as much as the camera
        was bright, so a panel drawn over a dim room barely registers -- and
        because the difference then tracks the camera's own noise, what does
        register is speckle.  A backdrop has to be able to say it is a
        backdrop.

        **Everything is done at the frame's own size and scaled once at the
        end.**  The frame is a crop of the camera roughly a third the area of
        the screen, so the mask, the premultiply and the channel interleave
        each cost a third of what they cost at screen size -- and the scale
        that used to be done to the colour and the mask separately is now one
        pass over one four-channel image.  It is also the more correct order:
        premultiplied alpha is exactly the representation in which linear
        interpolation of a translucent image is valid, so scaling afterwards
        is what stops edges haloing, where scaling the straight colour first
        is what causes it.
        """
        alpha = self._alpha_for(frame, clean)
        if solid is not None:
            cv2.max(alpha, solid, dst=alpha)
        shape = frame.shape[:2]
        premul, wide, small = self._buffers_for(shape)
        # UpdateLayeredWindow wants the colour premultiplied by the alpha it is
        # about to blend with; skipping this haloes every edge.
        cv2.merge((alpha, alpha, alpha), dst=wide)
        cv2.multiply(frame, wide, dst=premul, scale=1.0 / 255.0)
        cv2.mixChannels([premul, alpha], [small], (0, 0, 1, 1, 2, 2, 3, 3))
        if shape == (self.height, self.width):
            np.copyto(self.bgra, small)
        else:
            cv2.resize(small, (self.width, self.height), dst=self.bgra,
                       interpolation=cv2.INTER_LINEAR)
        self._blit()

    def _buffers_for(self, shape):
        """The three scratch images, sized to whatever the frame turns out to be."""
        if self._small is None or self._small.shape[:2] != shape:
            self._premul = np.empty((*shape, 3), np.uint8)
            self._wide = np.empty((*shape, 3), np.uint8)
            self._small = np.empty((*shape, 4), np.uint8)
        return self._premul, self._wide, self._small

    def _alpha_for(self, frame: np.ndarray,
                   clean: np.ndarray | None) -> np.ndarray:
        """One alpha level per pixel: `dim` for camera, opaque for anything drawn."""
        shape = frame.shape[:2]
        if self._ink is None or self._ink.shape != shape:
            self._diff = np.empty((*shape, 3), np.uint8)
            self._ink = np.empty(shape, np.uint8)
            self._alpha = np.empty(shape, np.uint8)
        if clean is None or self.dim >= 1.0:
            self._alpha[:] = round(self.dim * 255)
            return self._alpha
        cv2.absdiff(frame, clean, dst=self._diff)
        cv2.max(self._diff[:, :, 0], self._diff[:, :, 1], dst=self._ink)
        cv2.max(self._ink, self._diff[:, :, 2], dst=self._ink)
        # The ramp is a function of one byte, so it is a lookup rather than
        # arithmetic: subtract, scale and clamp would be three passes over two
        # million pixels every frame to compute 256 distinct answers.
        return cv2.LUT(self._ink, self._ramp, dst=self._alpha)

    def _blit(self) -> None:
        """Hand the bitmap to the compositor, and reclaim the bottom of the pile."""
        screen = user32.GetDC(None)
        user32.UpdateLayeredWindow(self._hwnd, screen, ctypes.byref(self._pos),
                                   ctypes.byref(self._size), self._dc,
                                   ctypes.byref(self._origin), 0,
                                   ctypes.byref(self._blend), ULW_ALPHA)
        user32.ReleaseDC(None, screen)
        # Every newly shown window lands above this one, so the claim to the
        # bottom has to be renewed rather than made once.
        self._sink()

    def keys(self) -> list[str]:
        """Which hotkeys were pressed since the last call."""
        msg = wintypes.MSG()
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            if msg.message == WM_HOTKEY and 0 <= msg.wParam < len(HOTKEYS):
                self._pressed.append(HOTKEYS[msg.wParam])
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        pressed, self._pressed = self._pressed, []
        return pressed

    def close(self) -> None:
        for i in range(len(HOTKEYS)):
            user32.UnregisterHotKey(None, i)
        # The array is a window into the bitmap; dropping it before the bitmap
        # goes away keeps a stray reference from outliving its memory.
        self.bgra = None
        if getattr(self, "_dc", None):
            if getattr(self, "_old", None):
                gdi32.SelectObject(self._dc, self._old)
            gdi32.DeleteDC(self._dc)
            self._dc = None
        if getattr(self, "_bitmap", None):
            gdi32.DeleteObject(self._bitmap)
            self._bitmap = None
        if self._hwnd:
            user32.DestroyWindow(self._hwnd)
            self._hwnd = None
        self.available = False
