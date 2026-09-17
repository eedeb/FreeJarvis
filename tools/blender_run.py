"""Run a script inside Blender, headless.

    python tools/blender_run.py tools/bl_gauntlet.py models/gauntlet.glb

Blender ships its own Python with `bpy` in it, so anything that touches a model
has to run *inside* Blender rather than in this project's virtualenv. This is
the seam between the two: it finds the executable and hands the arguments on.

Everything under tools/bl_*.py is on the Blender side of that seam. Those files
import `bpy` and will not run here; everything else in tools/ is the other way
round.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

WINGET = pathlib.Path(r"C:\Program Files\Blender Foundation")


def find_blender() -> pathlib.Path | None:
    """Where Blender is, or None."""
    env = os.environ.get("BLENDER")
    if env and pathlib.Path(env).exists():
        return pathlib.Path(env)
    # Newest first: "Blender 5.2" sorts after "Blender 4.5" for our purposes,
    # and a version-sorted string compare is close enough with one digit.
    if WINGET.exists():
        found = sorted(WINGET.glob("Blender */blender.exe"), reverse=True)
        if found:
            return found[0]
    for name in ("blender", "blender.exe"):
        for folder in os.environ.get("PATH", "").split(os.pathsep):
            candidate = pathlib.Path(folder) / name
            if candidate.exists():
                return candidate
    steam = pathlib.Path(r"C:\Program Files (x86)\Steam\steamapps\common"
                         r"\Blender\blender.exe")
    return steam if steam.exists() else None


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    blender = find_blender()
    if blender is None:
        print("Blender not found. Install it:\n"
              "  winget install --id BlenderFoundation.Blender --exact\n"
              "or set BLENDER to the path of blender.exe.")
        return 1

    script = pathlib.Path(argv[1]).resolve()
    if not script.exists():
        print(f"No such script: {script}")
        return 1
    rest = [str(pathlib.Path(a).resolve()) if pathlib.Path(a).exists() else a
            for a in argv[2:]]

    # --factory-startup so a stray addon or a saved preference cannot change
    # what an import does; the results have to be the same on any machine.
    cmd = [str(blender), "--background", "--factory-startup", "-noaudio",
           "--python-exit-code", "1", "--python", str(script), "--"] + rest
    print(f"  {blender}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
