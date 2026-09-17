"""The full-screen overlay: that it is see-through, click-through, and underneath.

This one really does open a window on the real desktop, because every property
worth checking is a property of the window Windows made, not of our arguments
to CreateWindowEx.  It puts up a flat grey "camera" field with a bright bar
"drawn" on it, so the two alpha levels are separable, and takes it down again.

A screenshot would not settle any of it: BitBlt on the screen DC does not
capture layered windows, so the compositing has to be checked in the bitmap we
hand the compositor and the placement in the z-order Windows reports.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ctypes
from ctypes import wintypes

import cv2
import numpy as np

from gesture_control.config import Settings
from gesture_control.mouse import enable_dpi_awareness, primary_screen_size
from gesture_control.overlay import (INK_FLOOR, INK_FULL,
                                     Overlay, WS_EX_LAYERED, WS_EX_NOACTIVATE,
                                     WS_EX_TRANSPARENT)

ok = True


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetTopWindow.restype = wintypes.HWND
user32.GetTopWindow.argtypes = [wintypes.HWND]
user32.GetWindow.restype = wintypes.HWND
user32.GetWindow.argtypes = [wintypes.HWND, ctypes.c_uint]
user32.FindWindowW.restype = wintypes.HWND
user32.WindowFromPoint.restype = wintypes.HWND
user32.WindowFromPoint.argtypes = [wintypes.POINT]
user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
GWL_EXSTYLE, GW_HWNDNEXT = -20, 2


def handle(hwnd):
    return ctypes.cast(hwnd, ctypes.c_void_p).value


def zorder():
    """Every top-level window, front to back."""
    out, h = [], user32.GetTopWindow(None)
    while h:
        out.append(handle(h))
        h = user32.GetWindow(h, GW_HWNDNEXT)
    return out


def classname(hwnd):
    buf = ctypes.create_unicode_buffer(128)
    user32.GetClassNameW(wintypes.HWND(hwnd), buf, 128)
    return buf.value


enable_dpi_awareness()
W, H = primary_screen_size()
DIM = 0.32
GREY, BAR = 128, (40, 190, 255)

print(f"Opening the overlay over the real desktop ({W}x{H}):")
ov = Overlay((W, H), dim=DIM)
check("it opened", ov.available)
if not ov.available:
    print(getattr(ov, "unavailable_reason", ""))
    raise SystemExit(1)

try:
    mine = handle(ov._hwnd)
    clean = np.full((H, W, 3), GREY, np.uint8)
    frame = clean.copy()
    cv2.rectangle(frame, (W // 4, H // 3), (3 * W // 4, 2 * H // 3), BAR, -1)
    ov.show(frame, clean)

    style = user32.GetWindowLongPtrW(ov._hwnd, GWL_EXSTYLE)
    check("it is layered, so it can have per-pixel alpha", style & WS_EX_LAYERED)
    check("it is transparent, so clicks fall through to what is underneath",
          style & WS_EX_TRANSPARENT)
    check("it never takes focus, so keys reach the app you are working in",
          style & WS_EX_NOACTIVATE)
    check("it is visible", bool(user32.IsWindowVisible(ov._hwnd)))

    print("\nWhere it sits in the pile:")
    order = zorder()
    desktop = handle(user32.FindWindowW("Progman", None))
    check("it is in the z-order at all", mine in order)
    check("it is in front of the desktop, so it covers the wallpaper",
          mine in order and desktop in order and order.index(mine) < order.index(desktop))
    ahead = [h for h in order[:order.index(mine)]
             if user32.IsWindowVisible(wintypes.HWND(h))
             and classname(h) != "GestureControlOverlay"]
    check(f"every other visible window is in front of it ({len(ahead)} of them)",
          len(ahead) > 0)

    # Click-through is the one that matters most: the app moves the real cursor
    # and clicks with it, so an overlay that answered to hit-testing would eat
    # every click it made.
    spot = wintypes.POINT(W // 2, H // 2)
    check("hit-testing in the middle of it lands on something else",
          handle(user32.WindowFromPoint(spot)) != mine)

    print("\nWhat it hands the compositor:")
    wash, ink = ov.bgra[H // 8, W // 8], ov.bgra[H // 2, W // 2]
    check(f"bare camera is drawn at dim ({wash[3]} of 255)",
          wash[3] == round(DIM * 255))
    check(f"what the app drew on top of it is solid ({ink[3]} of 255)",
          ink[3] == 255)
    # UpdateLayeredWindow blends colour it expects to be premultiplied already;
    # getting this wrong is invisible on a flat field and haloes every edge.
    check("the camera colour is premultiplied by its alpha",
          abs(int(wash[0]) - GREY * int(wash[3]) // 255) <= 1)
    check("the drawn colour is premultiplied by its alpha",
          all(abs(int(ink[i]) - BAR[i]) <= 1 for i in range(3)))

    print("\nA caller can insist a region is opaque:")
    # Inference alone cannot carry a backdrop: darkening a dim camera barely
    # differs from it, so the region comes out see-through and speckled.
    floor = np.zeros((H, W), np.uint8)
    floor[H // 4:H // 2, W // 4:W // 2] = 255
    flat = np.full((H, W, 3), GREY, np.uint8)
    ov.show(flat, flat, solid=floor)
    check("a declared region is fully opaque even where nothing was drawn",
          int(ov.bgra[H // 3, W // 3, 3]) == 255)
    check("and the rest of the frame is still the camera's wash",
          int(ov.bgra[H // 8, W // 8, 3]) == round(DIM * 255))
    ov.show(frame, clean)

    print("\nCost, at the size it actually runs at:")
    # Against a yardstick rather than a clock. This machine's absolute timings
    # move by 2-3x between runs -- the same code measured 8.8ms and 13.7ms an
    # hour apart -- so an absolute bound mostly tests the machine's mood. A
    # ratio against one full-frame pass over the same pixels does not, and a
    # regression means doing more passes, which shows up here whatever the
    # clock says.
    #
    # The measured ratio is ~16, which is higher than counting the operations
    # suggests (a difference, two maxes, a lookup, a multiply, a channel
    # interleave). Most of the excess is the interleave and the blit: those
    # write into the DIB section Windows composites from, which is not
    # ordinary heap memory and costs several times more per byte than the
    # arithmetic reading from it does.
    def timed(fn, n=15):
        for _ in range(3):
            fn()
        start = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - start) / n * 1000

    per = timed(lambda: ov.show(frame, clean))
    unit = timed(lambda: cv2.absdiff(frame, clean))
    check(f"a frame stays a fixed number of passes over the image "
          f"({per:.1f} ms = {per / unit:.1f}x one absdiff of {unit:.1f} ms)",
          per / unit < 24.0)
    # This test hands it the whole screen; the app hands it the smaller crop.
    # So this is the pessimistic end, and it still has to fit in a camera
    # frame -- the overlay must not become the slower half of the loop.
    check(f"and a frame still fits inside the camera's ({per:.1f} ms)", per < 30.0)

    print("\nKeys:")
    check("it collects hotkeys without blocking", ov.keys() == [])
    check("every hotkey is a Ctrl+Alt combination, not a bare letter",
          all(len(k) == 1 for k in __import__(
              "gesture_control.overlay", fromlist=["HOTKEYS"]).HOTKEYS))
finally:
    ov.close()

print("\nAfter closing:")
check("the window is gone", not user32.IsWindowVisible(wintypes.HWND(mine)))
check("it reports itself unavailable", not ov.available)
check("closing twice is harmless", ov.close() is None)

print("\nSettings:")
check("the overlay is the default", Settings().overlay is True)
check("and the old window is still reachable", Settings(overlay=False).overlay is False)
# 0 is the default, and means the camera is hidden entirely -- only the glove
# and the readouts are drawn. Anything above it is how much of the room shows
# through, which is a diagnostic setting rather than the normal one.
check(f"the camera is hidden by default ({Settings().overlay_dim})",
      Settings().overlay_dim == 0.0)
check("and can be turned back on without becoming opaque",
      0.0 < Settings(overlay_dim=0.28).overlay_dim < 0.6)

print("\nHiding the camera:")
from gesture_control.controller import GestureController

stub = GestureController.__new__(GestureController)
stub._overlay = object()
stub.settings = Settings()
check("with dim at 0 the camera is dropped", stub._hide_camera() is True)
stub.settings = Settings(overlay_dim=0.3)
check("and shown again above 0", stub._hide_camera() is False)
stub.settings = Settings()
stub._overlay = None
# A floating window with no camera in it is an empty rectangle; the overlay
# has a desktop behind it for the glove to float over, a window does not.
check("a floating window always keeps its camera", stub._hide_camera() is False)

print("\nThe hand is drawn where the cursor goes:")
# The point of a full-screen overlay, and the thing that was silently wrong:
# direct mode stretches the middle `1-2*margin` of the camera's view over the
# whole screen, so an overlay showing the *whole* view draws the glove in the
# right place only at dead centre. These two have to agree everywhere.
from gesture_control.controller import GestureController
from gesture_control.mapping import AimSample, DirectMap
from synthetic_hand import FakeTracker

settings = Settings()
ctl = GestureController.__new__(GestureController)
ctl._overlay, ctl.settings = object(), settings          # _reach_box only reads these
CAM_W, CAM_H = 1920, 1080
box = GestureController._reach_box(ctl, CAM_W, CAM_H)
check(f"direct mode crops to the reachable region {box}", box is not None)

worst = 0.0
mapping = DirectMap((W, H), mirror=settings.mirror, margin=settings.direct_margin)
for u in (0.2, 0.35, 0.5, 0.65, 0.8):
    for v in (0.2, 0.5, 0.8):
        # where the app sends the cursor
        want = mapping.predict(AimSample(tip=(u, v), direction=(0, 0), scale=0.09))
        # where the preview draws that hand: mirrored into frame pixels, moved
        # into the crop, then scaled by the overlay to fill the screen
        px = (1.0 - u) * CAM_W - box[0]
        py = v * CAM_H - box[1]
        got = (px * W / (box[2] - box[0]), py * H / (box[3] - box[1]))
        worst = max(worst, abs(got[0] - want[0]), abs(got[1] - want[1]))
check(f"the glove and the cursor land on the same pixel everywhere "
      f"(worst {worst:.1f}px)", worst <= 2.0)

# And the check that would have caught the bug in the picture: the border
# shading repaints half the frame, which the mask cannot tell from something
# deliberately drawn, so it has to be off whenever the crop is on.
src = (pathlib.Path(__file__).resolve().parent.parent
       / "gesture_control" / "controller.py").read_text(encoding="utf-8")
check("the overscan shading is skipped when the view is cropped",
      "if m > 0 and box is None:" in src)

print(chr(10) + "RESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")

print("\nIt does its arithmetic at the frame's size, not the screen's:")
# The pixels are packed at the crop's own size and scaled once at the end,
# rather than scaling the colour and the mask separately and then doing three
# full-screen passes. Two things have to hold: the picture must come out the
# same, and the buffers must be the small ones.
small = Overlay.__new__(Overlay)
small.width, small.height = 480, 270
small.dim = 0.0
small._premul = small._wide = small._small = None
small._diff = small._ink = small._alpha = None
steps = np.arange(256, dtype=np.float32)
lift = np.clip((steps - INK_FLOOR) / (INK_FULL - INK_FLOOR), 0.0, 1.0)
small._ramp = (255 * lift).astype(np.uint8)
small.bgra = np.empty((270, 480, 4), np.uint8)
small._blit = lambda: None

crop = np.zeros((135, 240, 3), np.uint8)
bare = crop.copy()
cv2.circle(crop, (120, 67), 30, (255, 200, 80), -1)
small.show(crop, bare)
check("a frame smaller than the screen is scaled up to fill it",
      small.bgra.shape[:2] == (270, 480))
check("the scratch buffers are the frame's size, not the screen's",
      small._small.shape[:2] == crop.shape[:2])
check("what was drawn is opaque", int(small.bgra[135, 240, 3]) == 255)
check("and what was not is clear", int(small.bgra[8, 8, 3]) == 0)
# Premultiplied colour is exactly the representation in which interpolating a
# translucent image is valid, so no pixel may come back brighter than its own
# alpha allows -- that is what a halo is.
haloed = int((small.bgra[:, :, :3].max(axis=2).astype(np.int32)
              - small.bgra[:, :, 3].astype(np.int32)).max())
check(f"no pixel is brighter than its alpha, so no edge haloes ({haloed})",
      haloed <= 1)

same = Overlay.__new__(Overlay)
same.width, same.height = 240, 135
same.dim = 0.0
same._premul = same._wide = same._small = None
same._diff = same._ink = same._alpha = None
same._ramp = small._ramp
same.bgra = np.empty((135, 240, 4), np.uint8)
same._blit = lambda: None
same.show(crop, bare)
check("a frame already at screen size is copied, not resized",
      int(same.bgra[67, 120, 3]) == 255 and int(same.bgra[4, 4, 3]) == 0)

raise SystemExit(0 if ok else 1)
