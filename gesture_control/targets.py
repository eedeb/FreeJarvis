"""Finding clickable things on screen, and bending the cursor towards them.

Pointing with a hand lands within roughly a dozen pixels of where you meant.
Most buttons are bigger than that, but small ones -- toolbar icons, window
close buttons, list rows -- are not, and that is where hand control stops
feeling usable.  So the cursor is given a little gravity: near a clickable
element the cursor latches to it and holds still.

Two halves, deliberately separated:

* :func:`snap` is pure geometry over a list of rectangles.  It is what runs
  every frame, it costs microseconds, and it is fully testable without a
  desktop.
* :class:`TargetFinder` is the messy half: it asks Windows UI Automation what
  is clickable in whichever window the cursor is over.  That query costs
  30-200 ms depending on the app, which is far too slow for a 30 fps loop, so
  it runs on its own thread and publishes a snapshot the control loop reads.

If UI Automation is unavailable for any reason the finder simply reports no
targets and the cursor behaves exactly as it did before.
"""

from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np

# UIA control types worth aiming at.  Panes, windows, groups and other
# containers are deliberately absent: they are big, they are everywhere, and
# snapping to them would fight the user rather than help.
CLICKABLE_TYPES: dict[int, str] = {
    50000: "button",
    50002: "checkbox",
    50003: "combobox",
    50004: "edit",
    50005: "link",
    50006: "image",
    50007: "item",
    50011: "menuitem",
    50013: "radiobutton",
    50018: "tab",
    50021: "slider",
    50023: "treeitem",
    50024: "custom",
    50031: "splitbutton",
}
# Toolbars, panes, groups, static text, scrollbars and progress bars are
# deliberately absent.  Several of them are technically interactive, but they
# are *containers*: they enclose the controls you actually want, so they sit at
# distance zero from the cursor and would win every contest against the small
# button inside them -- exactly backwards.


@dataclass(frozen=True)
class Target:
    """A clickable rectangle in screen pixels."""

    left: int
    top: int
    right: int
    bottom: int
    kind: str

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.left + self.right) / 2.0,
                         (self.top + self.bottom) / 2.0])

    def contains(self, point) -> bool:
        return (self.left <= point[0] <= self.right
                and self.top <= point[1] <= self.bottom)

    def locked_axes(self, long_ratio: float = 2.5,
                    long_px: float = 70.0) -> tuple[bool, bool]:
        """Which axes to pin, as (x, y).

        A control that is much longer than it is wide is a track, not a point:
        a window's title bar, a scrollbar, a menu row.  Along its length the
        position matters and should follow the hand; across its width it does
        not, and pinning that axis is what makes a 30-pixel-tall title bar easy
        to hold on to.  A compact control has no such axis and pins both.
        """
        wide = self.width >= long_px and self.width >= self.height * long_ratio
        tall = self.height >= long_px and self.height >= self.width * long_ratio
        return (not wide, not tall)

    def nearest_point(self, point, inset: float = 0.0) -> np.ndarray:
        """Closest point inside the rectangle, pulled `inset` off the edge."""
        ix = min(inset, max(self.width / 2.0 - 1.0, 0.0))
        iy = min(inset, max(self.height / 2.0 - 1.0, 0.0))
        return np.array([
            min(max(point[0], self.left + ix), self.right - ix),
            min(max(point[1], self.top + iy), self.bottom - iy),
        ])

    def distance(self, point) -> float:
        dx = max(self.left - point[0], 0.0, point[0] - self.right)
        dy = max(self.top - point[1], 0.0, point[1] - self.bottom)
        return float(np.hypot(dx, dy))


def snap(point, targets, radius: float, sticky: Target | None = None,
         sticky_margin: float = 24.0, inset: float = 4.0, size_bias: float = 8.0,
         unlock_radius: float | None = None
         ) -> tuple[np.ndarray, Target | None]:
    """Lock the cursor onto the nearest control, and hold it there.

    Not a gentle pull but a latch: once a control is acquired the cursor sits
    at its centre and stays put while the hand wobbles, which is the whole
    point -- the hand is never still, and a control small enough to be worth
    helping with is smaller than the hand's own jitter.

    Along a long control the lock applies to one axis only, so a title bar
    holds the cursor vertically while letting it run left and right.

    Letting go takes more than acquiring: the *hand's* position, not the
    latched one, has to leave a wider radius. Without that gap the cursor
    would sit exactly on the boundary and rattle in and out of the lock.
    """
    point = np.asarray(point, dtype=float)
    if not targets:
        return point, None
    release_at = unlock_radius if unlock_radius is not None else radius * 1.8

    # A held lock competes with everything else rather than merely waiting to
    # expire. Letting it hold unconditionally until the hand is past the wider
    # release radius means that radius has to be smaller than the gap between
    # neighbouring controls, or the cursor sails straight over the one in
    # between: locked to the first, still not free by the time it reaches the
    # second, and landing on the third. So the lock only survives while nothing
    # nearer has turned up, and the wider radius decides one thing only --
    # whether to keep holding when there is no other candidate at all.
    candidates = []
    for target in targets:
        d = target.distance(point)
        held = sticky is not None and target == sticky
        if d > (release_at if held else radius):
            continue
        # Size enters the ranking, not just distance. Controls nest: sitting
        # inside a big pane puts it at distance zero, which would beat every
        # small button beside it even though the button is obviously the
        # intended target.
        score = d + np.sqrt(target.width * target.height) / size_bias
        if held:
            # Enough to stop two equidistant neighbours trading the lock back
            # and forth, not enough to hold on past one that is plainly nearer.
            score -= sticky_margin
        candidates.append((score, target))

    if not candidates:
        return point, None
    best = min(candidates, key=lambda c: c[0])[1]

    lock_x, lock_y = best.locked_axes()
    centre = best.center
    x = centre[0] if lock_x else min(max(point[0], best.left + inset),
                                     best.right - inset)
    y = centre[1] if lock_y else min(max(point[1], best.top + inset),
                                     best.bottom - inset)
    return np.array([x, y]), best


# --------------------------------------------------------------------------
# The Windows half
# --------------------------------------------------------------------------

user32 = ctypes.WinDLL("user32", use_last_error=True)
_GA_ROOT = 2


def _window_under(x: int, y: int) -> int:
    """Top-level window at a screen point."""
    hwnd = user32.WindowFromPoint(wintypes.POINT(int(x), int(y)))
    if not hwnd:
        return 0
    return user32.GetAncestor(hwnd, _GA_ROOT) or hwnd


class TargetFinder:
    """Publishes the clickable rectangles near the cursor, refreshed off-thread."""

    def __init__(self, screen: tuple[int, int], refresh_s: float = 0.9,
                 max_elements: int = 1500):
        self.screen = screen
        self.refresh_s = refresh_s
        self.max_elements = max_elements

        self._targets: tuple[Target, ...] = ()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cursor = (screen[0] // 2, screen[1] // 2)
        self._enabled = False
        self.unavailable_reason: str | None = None
        self.last_query_ms = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "TargetFinder":
        try:
            import comtypes  # noqa: F401  (import test only)
        except ImportError:
            self.unavailable_reason = (
                "comtypes is not installed, so snapping is off "
                "(pip install comtypes)")
            return self
        self._enabled = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="targets", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def available(self) -> bool:
        return self._enabled and self.unavailable_reason is None

    # -- readers -----------------------------------------------------------

    def note_cursor(self, x: float, y: float) -> None:
        """Tell the worker where to look next.  Called from the control loop."""
        self._cursor = (int(x), int(y))

    def targets(self) -> tuple[Target, ...]:
        with self._lock:
            return self._targets

    # -- worker ------------------------------------------------------------

    def _loop(self) -> None:
        try:
            import comtypes
            import comtypes.client
            comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
            comtypes.client.GetModule("UIAutomationCore.dll")
            from comtypes.gen import UIAutomationClient as uia_mod

            automation = comtypes.client.CreateObject(
                uia_mod.CUIAutomation, interface=uia_mod.IUIAutomation)
            cache = automation.CreateCacheRequest()
            for prop in (uia_mod.UIA_BoundingRectanglePropertyId,
                         uia_mod.UIA_ControlTypePropertyId,
                         uia_mod.UIA_IsEnabledPropertyId,
                         uia_mod.UIA_IsOffscreenPropertyId):
                cache.AddProperty(prop)
            condition = automation.CreateTrueCondition()
            scope = uia_mod.TreeScope_Descendants
        except Exception as exc:  # noqa: BLE001 - any COM failure disables snapping
            self.unavailable_reason = f"UI Automation unavailable ({exc})"
            self._enabled = False
            return

        last_hwnd = 0
        next_refresh = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            hwnd = _window_under(*self._cursor)
            # Refresh immediately when the cursor crosses into another window,
            # otherwise on a timer so a changing UI does not go stale.
            if hwnd == last_hwnd and now < next_refresh:
                time.sleep(0.03)
                continue

            started = time.perf_counter()
            found = self._query(automation, cache, condition, scope, hwnd)
            elapsed = time.perf_counter() - started
            self.last_query_ms = elapsed * 1000.0

            with self._lock:
                self._targets = found
            last_hwnd = hwnd
            # Expensive windows (big browser trees) get polled less often, so
            # the worker can never eat a core no matter what is on screen.
            next_refresh = now + max(self.refresh_s, elapsed * 8.0)

    def _query(self, automation, cache, condition, scope, hwnd) -> tuple[Target, ...]:
        if not hwnd:
            return ()
        try:
            element = automation.ElementFromHandle(hwnd)
            found = element.FindAllBuildCache(scope, condition, cache)
            count = min(found.Length, self.max_elements)
        except Exception:  # noqa: BLE001 - windows close mid-query; just skip
            return ()

        out: list[Target] = []
        for i in range(count):
            try:
                el = found.GetElement(i)
                if not el.CachedIsEnabled or el.CachedIsOffscreen:
                    continue
                kind = CLICKABLE_TYPES.get(el.CachedControlType)
                if kind is None:
                    continue
                rect = el.CachedBoundingRectangle
                target = Target(int(rect.left), int(rect.top),
                                int(rect.right), int(rect.bottom), kind)
            except Exception:  # noqa: BLE001 - stale element in the cache
                continue
            if self._plausible(target):
                out.append(target)
        return tuple(out)

    def _plausible(self, t: Target) -> bool:
        """Reject rectangles that are not really aim-able things."""
        if t.width < 6 or t.height < 6:
            return False
        # Anything spanning much of the screen is a container, not a control.
        if t.width > self.screen[0] * 0.6 or t.height > self.screen[1] * 0.6:
            return False
        # Only the primary monitor is calibrated, so ignore anything elsewhere.
        if t.right < 0 or t.bottom < 0:
            return False
        return t.left < self.screen[0] and t.top < self.screen[1]
