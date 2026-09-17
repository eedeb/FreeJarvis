"""Start FreeJarvis with no console window, and keep a log of what it said.

The Start Menu shortcut runs this with `pythonw.exe`, which is the only way to
launch a Python app on Windows without a black console window sitting behind
the overlay for as long as it runs.

That has one sharp edge worth stating, because it is a crash and not a
cosmetic problem: under `pythonw` there is no standard output at all --
`sys.stdout` and `sys.stderr` are `None`, not a sink -- so the first `print()`
anywhere in the app raises `AttributeError: 'NoneType' object has no attribute
'write'`.  This app prints plenty: which parts of Jarvis started, which could
not and why, the frame rate warnings.  So the very first thing that happens
here, before anything is imported, is giving those two names somewhere to go.

A file rather than a null sink, because the printed lines are the only
diagnosis anyone gets when an installed copy will not start -- there is no
console to read them in.  The log is truncated at each launch: what is wanted
is why *this* run failed, and an append-only file of every run since install
would be worse at answering that.
"""

import datetime
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG = LOG_DIR / "freejarvis.log"


def _open_log():
    """The log file, or None if even that is not possible."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return open(LOG, "w", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        return None


def main() -> int:
    stream = _open_log()
    if stream is not None:
        sys.stdout = sys.stderr = stream
    else:
        # No log and no console. Anything written has to go somewhere that
        # exists, or the app dies on its first print.
        sys.stdout = sys.stderr = open(os.devnull, "w")

    print(f"FreeJarvis starting {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  python {sys.version.split()[0]}")
    print(f"  from   {ROOT}")
    print()

    # Imported here, not at the top: the app prints while it is being imported,
    # and until the two lines above have run there is nowhere for that to go.
    sys.path.insert(0, str(ROOT))
    try:
        from gesture_control.camera import CameraError
        from gesture_control.cli import main as run
        from gesture_control.tracker import ModelError
    except Exception:                                            # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("\nFreeJarvis could not start. The traceback above is the reason.")
        return 1

    try:
        return int(run() or 0)
    except KeyboardInterrupt:
        return 0
    except (CameraError, ModelError) as exc:
        # The two failures that are expected and actionable -- no camera, no
        # model. Their message is the whole story, and a traceback would only
        # bury it. cli.entry() does the same thing for the console launch.
        print(f"\n{exc}\n")
        return 1
    except Exception:                                            # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("\nFreeJarvis stopped because of the error above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
