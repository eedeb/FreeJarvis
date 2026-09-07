"""Showing and hiding the Windows on-screen keyboard.

Typing is the one thing a hand in the air cannot do directly, so it is handed
to the keyboard Windows already ships: `osk.exe`, the accessibility on-screen
keyboard. Its keys are ordinary clickable controls, which means the cursor and
the pinch-click already work on them, and snapping treats them as targets like
anything else.
"""

from __future__ import annotations

import ctypes
import subprocess

user32 = ctypes.WinDLL("user32", use_last_error=True)

# The on-screen keyboard's top-level window class.
_OSK_CLASS = "OSKMainClass"
_WM_CLOSE = 0x0010


def is_showing() -> bool:
    return bool(user32.FindWindowW(_OSK_CLASS, None))


def show() -> bool:
    """Launch the on-screen keyboard. True if it is up afterwards."""
    if is_showing():
        return True
    try:
        # DETACHED_PROCESS so closing this app does not take the keyboard with
        # it, and so no console window flashes up.
        subprocess.Popen(["osk.exe"], creationflags=0x00000008,
                         close_fds=True)
    except (OSError, ValueError):
        return False
    return True


def hide() -> bool:
    """Ask the on-screen keyboard to close.  True if it is gone afterwards."""
    handle = user32.FindWindowW(_OSK_CLASS, None)
    if not handle:
        return True
    user32.PostMessageW(handle, _WM_CLOSE, 0, 0)
    return True


def toggle() -> bool:
    """Flip it, returning whether it should now be showing."""
    if is_showing():
        hide()
        return False
    return show()
