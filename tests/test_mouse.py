"""Check the SendInput absolute-coordinate math against the real cursor.

Moves the pointer a few pixels, reads it back, and restores it.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import time
from gesture_control.mouse import (Mouse, cursor_position, primary_screen_size,
                                   virtual_screen_rect)

m = Mouse()
print("primary screen :", primary_screen_size())
print("virtual desktop:", virtual_screen_rect())

start = cursor_position()
print("cursor now     :", start)

ok = True
try:
    for dx, dy in [(40, 25), (-60, 40), (0, 0)]:
        target = (min(max(start[0] + dx, 5), m.width - 5),
                  min(max(start[1] + dy, 5), m.height - 5))
        m.move_to(*target)
        time.sleep(0.06)
        got = cursor_position()
        err = max(abs(got[0] - target[0]), abs(got[1] - target[1]))
        good = err <= 2          # 1 px of rounding through the 0..65535 grid
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] asked {target} -> got {got} (off by {err} px)")
finally:
    m.move_to(*start)
    time.sleep(0.05)

back = cursor_position()
restored = max(abs(back[0] - start[0]), abs(back[1] - start[1])) <= 2
ok &= restored
print(f"  [{'PASS' if restored else 'FAIL'}] cursor restored to {back}")
print("RESULT:", "mouse control works" if ok else "MOUSE CHECKS FAILED")
raise SystemExit(0 if ok else 1)
