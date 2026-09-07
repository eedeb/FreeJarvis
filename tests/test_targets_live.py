"""Live check of the UI Automation half of snapping.

Needs a real desktop with windows open. It only reads element geometry -- it
never clicks anything and never moves the cursor.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import collections
import ctypes
import time

from gesture_control.mouse import enable_dpi_awareness, primary_screen_size
from gesture_control.targets import TargetFinder, snap

enable_dpi_awareness()
screen = primary_screen_size()
print(f"primary screen: {screen[0]}x{screen[1]}")

ok = True


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


finder = TargetFinder(screen, refresh_s=0.4).start()
if not finder.available:
    print(f"  SKIP: {finder.unavailable_reason}")
    raise SystemExit(0)

user32 = ctypes.WinDLL("user32")
best = ()
try:
    # Sweep the cursor hint across the screen (without moving the real cursor)
    # so the finder visits whatever windows happen to be open.
    for fx in (0.25, 0.5, 0.75):
        for fy in (0.25, 0.5, 0.75):
            finder.note_cursor(screen[0] * fx, screen[1] * fy)
            deadline = time.monotonic() + 2.5
            while time.monotonic() < deadline:
                found = finder.targets()
                if found:
                    break
                time.sleep(0.05)
            if len(found) > len(best):
                best = found
finally:
    finder.stop()

print(f"  slowest query seen: {finder.last_query_ms:.0f} ms")
check("UI Automation returned clickable targets", len(best) > 0)
if not best:
    print("\nRESULT: SOME CHECKS FAILED (no windows with controls on screen?)")
    raise SystemExit(1)

kinds = collections.Counter(t.kind for t in best)
print(f"  {len(best)} targets: " + ", ".join(f"{k} x{v}" for k, v in kinds.most_common(6)))

widths = sorted(t.width for t in best)
heights = sorted(t.height for t in best)
print(f"  sizes: width {widths[0]}-{widths[-1]}px, height {heights[0]}-{heights[-1]}px")

check("every target is on the primary screen",
      all(t.left < screen[0] and t.top < screen[1] and t.right > 0 and t.bottom > 0
          for t in best))
check("no target is a whole-window container",
      all(t.width <= screen[0] * 0.6 and t.height <= screen[1] * 0.6 for t in best))
check("no degenerate rectangles", all(t.width >= 6 and t.height >= 6 for t in best))

# Snapping against the real list must behave the same as against synthetic ones.
sample = min(best, key=lambda t: t.width * t.height)   # a small, fiddly control
just_outside = (sample.right + 15.0, (sample.top + sample.bottom) / 2.0)
moved, chosen = snap(just_outside, best, 45.0)
check(f"a point 15px off a real {sample.kind} ({sample.width}x{sample.height}) "
      f"snaps to something", chosen is not None)
before = chosen.distance(just_outside) if chosen else 0.0
after = chosen.distance(moved) if chosen else 0.0
print(f"  chose a {chosen.kind} ({chosen.width}x{chosen.height}); "
      f"distance {before:.1f} -> {after:.1f}px")
# If the point already sits inside the chosen control there is nothing to do;
# otherwise the snap has to have closed the gap.
check("snapping never moves the point away from its target", after <= before + 1e-9)
check("snapping closes the gap when there is one", before == 0.0 or after < before)

# A container enclosing the cursor must not beat a small control beside it.
from gesture_control.targets import Target
big = Target(0, 0, 900, 500, "custom")
small = Target(400, 300, 440, 320, "button")
_, picked = snap((450, 310), [big, small], 45.0)
check("a small control beside the cursor beats the pane under it",
      picked is small)

far_corner = (screen[0] - 1.0, screen[1] - 1.0)
_, none_chosen = snap(far_corner, [t for t in best if t.distance(far_corner) > 100], 45.0)
check("a point far from every control is left alone", none_chosen is None)

print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
