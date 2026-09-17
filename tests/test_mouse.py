"""Check the SendInput absolute-coordinate math against the real cursor.

Moves the pointer a few pixels, reads it back, and restores it.

This is the one test that competes for a piece of hardware someone else may be
holding.  It asks Windows to put the cursor somewhere and then asks where the
cursor is, and *anything* that moves the pointer in between -- a hand on the
mouse, another copy of this app driving it, a window opening under it -- makes
the readback disagree with the request.

That is not the coordinate math being wrong, which is the only thing this test
exists to check, so the two are told apart rather than both reported as
failure.  Before each attempt the pointer is watched briefly; if it is moving
on its own, the attempt is not evidence either way and is retried.  If every
attempt is contaminated the answer is "inconclusive, the pointer was in use" --
not "broken".  Measured on this machine, a parked pointer passes every run and
a pointer someone is using fails about half of them, which is exactly the
confusion this is here to remove.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import time
from gesture_control.mouse import (Mouse, cursor_position, primary_screen_size,
                                   virtual_screen_rect)

TRIES = 5
SLACK = 2                    # 1px of rounding through the 0..65535 grid
SETTLE = 0.06

m = Mouse()
print("primary screen :", primary_screen_size())
print("virtual desktop:", virtual_screen_rect())

start = cursor_position()
print("cursor now     :", start)

ok = True
blocked = 0


def busy() -> bool:
    """Whether something else is moving the pointer right now."""
    first = cursor_position()
    time.sleep(0.03)
    return cursor_position() != first


def land(target):
    """Put the cursor on `target`. Returns (verdict, error, attempts).

    verdict is True (the math is right), False (it is wrong), or None (the
    pointer was in use every time, so nothing was measured).
    """
    worst = None
    for attempt in range(1, TRIES + 1):
        if busy():
            continue
        m.move_to(*target)
        time.sleep(SETTLE)
        got = cursor_position()
        err = max(abs(got[0] - target[0]), abs(got[1] - target[1]))
        if err <= SLACK:
            return True, err, attempt
        # A miss only counts if the pointer held still through the check.
        if not busy():
            worst = err
    return (None if worst is None else False), worst, TRIES


try:
    for dx, dy in [(40, 25), (-60, 40), (0, 0)]:
        target = (min(max(start[0] + dx, 5), m.width - 5),
                  min(max(start[1] + dy, 5), m.height - 5))
        verdict, err, tries = land(target)
        if verdict is None:
            blocked += 1
            print(f"  [SKIP] asked {target} -- the pointer was in use")
            continue
        ok &= verdict
        note = "" if tries == 1 else f", on try {tries}"
        print(f"  [{'PASS' if verdict else 'FAIL'}] asked {target} -> "
              f"off by {err} px{note}")
finally:
    m.move_to(*start)
    time.sleep(0.05)

if blocked == 3:
    print("RESULT: inconclusive -- something else was moving the pointer "
          "throughout. Run it again with the mouse left alone.")
    raise SystemExit(0)

back = cursor_position()
restored = max(abs(back[0] - start[0]), abs(back[1] - start[1])) <= SLACK
if not restored and busy():
    print(f"  [SKIP] cursor is at {back}, but it is being moved by something "
          f"else, so this proves nothing")
else:
    ok &= restored
    print(f"  [{'PASS' if restored else 'FAIL'}] cursor restored to {back}")
print("RESULT:", "mouse control works" if ok else "MOUSE CHECKS FAILED")
raise SystemExit(0 if ok else 1)
