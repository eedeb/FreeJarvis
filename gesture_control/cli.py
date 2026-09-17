"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
import time

from .config import CALIBRATION_PATH, Settings


def _parse_grid(text: str) -> tuple[int, int]:
    try:
        cols, rows = text.lower().split("x")
        c, r = int(cols), int(rows)
    except ValueError:
        raise argparse.ArgumentTypeError("grid must look like 4x3") from None
    if c < 2 or r < 2 or c * r < 4:
        raise argparse.ArgumentTypeError("grid needs at least 2x2 points")
    return c, r


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gesture-control",
        description="Control the mouse by pointing at the screen with your hand.")
    p.add_argument("command", nargs="?", default="run",
                   choices=["run", "calibrate", "check", "tune"],
                   help="run: control the mouse (calibrating first if needed). "
                        "calibrate: recalibrate. tune: measure your hand and fix "
                        "clicking. check: test the camera and tracking.")
    p.add_argument("--camera", type=int, help="camera index (default 0)")
    p.add_argument("--grid", type=_parse_grid,
                   help="calibration grid, e.g. 4x3 (more dots = more accurate)")
    p.add_argument("--no-preview", action="store_true",
                   help="hide the webcam preview window while controlling")
    p.add_argument("--no-gauntlet", action="store_true",
                   help="draw the plain tracking skeleton instead of the gauntlet")
    p.add_argument("--window", action="store_true",
                   help="show the preview in a floating window instead of as a "
                        "full-screen overlay")
    p.add_argument("--overlay-dim", type=float, metavar="F",
                   help="how visible the camera image is in the overlay, 0 to 1 "
                        "(0, the default, hides it and shows only the glove)")
    p.add_argument("--gauntlet-style", choices=("holo", "solid"),
                   help="draw the glove as projected light or as painted metal")
    p.add_argument("--gauntlet-model", choices=("rig", "drawn"),
                   help="which gauntlet to draw: the rigged generated glove, "
                        "or flat vectors")
    p.add_argument("--recalibrate", action="store_true",
                   help="force calibration even if a saved one exists")
    return p


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> None:
    if args.camera is not None:
        settings.camera_index = args.camera
    if args.grid is not None:
        settings.grid_cols, settings.grid_rows = args.grid
    if args.no_preview:
        settings.show_preview = False
    if args.no_gauntlet:
        settings.gauntlet = False
    if args.window:
        settings.overlay = False
    if args.overlay_dim is not None:
        settings.overlay_dim = args.overlay_dim
    if args.gauntlet_style:
        settings.gauntlet_style = args.gauntlet_style
    if args.gauntlet_model:
        settings.gauntlet_model = args.gauntlet_model


def cmd_check(settings: Settings) -> int:
    from .camera import permission_hint, probe
    from .tracker import HandTracker, ensure_model

    print("Cameras:")
    found = probe()
    if not found:
        hint = permission_hint()
        if hint:
            print(f"  none could be opened.\n\n  {hint}")
            return 1
        print("  none found.")
        print("  - close other apps using the webcam (Teams, Zoom, Camera)")
        print("  - Settings > Privacy & security > Camera > let desktop apps access")
        return 1
    for index, backend, size in found:
        mark = "*" if index == settings.camera_index else " "
        print(f"  {mark} index {index}  {backend:5s}  {size[0]}x{size[1]}")

    from .camera import benchmark
    print("\nWhat each resolution actually delivers:")
    modes = benchmark(settings.camera_index)
    best = max(modes, key=lambda m: (m[2] >= 24, m[0] * m[1]), default=None)
    for w, h, fps, fourcc in modes:
        mark = "*" if (w, h) == (settings.camera_width, settings.camera_height) else " "
        note = ""
        if best is not None and (w, h) == (best[0], best[1]):
            note = "  <- best: most detail at a usable frame rate"
        print(f"  {mark} {w}x{h:<5} {fps:5.1f} fps  {fourcc}{note}")
    if best is not None and (best[0], best[1]) != (settings.camera_width,
                                                   settings.camera_height):
        print(f"  Set camera_width {best[0]} and camera_height {best[1]} "
              f"in settings.json.")

    print(f"\nModel: {ensure_model()}")
    print(f"\nTracking on camera {settings.camera_index} for 5 seconds - "
          "hold your hand up...")

    with HandTracker(settings) as tracker:
        if not tracker.wait_until_ready(15.0):
            print("  no frames reached the tracker.")
            return 1
        seen = 0
        total = 0
        end = time.monotonic() + 5.0
        last = -1.0
        while time.monotonic() < end:
            tracker.raise_if_failed()
            hand = tracker.latest()
            if hand is not None and hand.stamp != last:
                last = hand.stamp
                seen += 1
            total += 1
            time.sleep(0.01)
        print(f"  {tracker.fps:.0f} fps through MediaPipe, "
              f"hand detected in {seen} frames.")
        if seen == 0:
            print("  No hand detected. Check lighting and that your hand is in frame.")
            return 1
        hand = tracker.latest()
        if hand is not None:
            fingers = ", ".join(k for k, v in hand.extended().items() if v) or "none"
            print(f"  last pose: {hand.handedness or 'hand'} "
                  f"({hand.confidence:.0%}), extended: {fingers}, "
                  f"pinch {hand.pinch():.2f}")
    print("\nAll good. Run `run.bat` to calibrate and start.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.load()
    _apply_overrides(settings, args)

    from .mouse import enable_dpi_awareness
    # Before any window exists, so screen sizes come back in real pixels.
    enable_dpi_awareness()

    if args.command == "check":
        return cmd_check(settings)

    from .calibrate import calibrate
    from .controller import GestureController
    from .mapping import PointerMap
    from .mouse import primary_screen_size
    from .tracker import HandTracker

    if args.command == "tune":
        from .tune import run_tuning
        with HandTracker(settings) as tracker:
            if not tracker.wait_until_ready(20.0):
                print("The tracker never produced a frame. Try `run.bat check`.")
                return 1
            return 0 if run_tuning(tracker, settings) else 1

    if settings.aim_mode == "direct" and args.command != "calibrate":
        # Nothing to fit: the camera frame maps straight onto the screen. Skip
        # every calibration check below and start controlling immediately.
        from .mapping import DirectMap
        model = DirectMap(primary_screen_size(), settings.mirror,
                          settings.direct_margin)
        with HandTracker(settings) as tracker:
            print("Camera ready. Warming up the tracker...")
            if not tracker.wait_until_ready(20.0):
                print("The tracker never produced a frame. Try `run.bat check`.")
                return 1
            GestureController(tracker, model, settings).run()
        print("Stopped. Your mouse is yours again.")
        return 0

    need_calibration = args.command == "calibrate" or args.recalibrate
    model = None
    if not need_calibration:
        if CALIBRATION_PATH.exists():
            try:
                model = PointerMap.load(CALIBRATION_PATH)
            except (OSError, ValueError, KeyError) as exc:
                print(f"Could not read {CALIBRATION_PATH} ({exc}); recalibrating.")
                need_calibration = True
        else:
            print("No calibration found - starting with calibration.")
            need_calibration = True

    if model is not None and model.stale:
        print("Your saved calibration was recorded before hand pose moved to 3D.\n"
              "Back then a pinch could register while your hand was still moving,\n"
              "so those calibration points cannot be trusted. Recalibrating.")
        need_calibration = True
        model = None

    if model is not None:
        saved_mode = model.meta.get("aim_mode", "finger")
        if saved_mode != settings.aim_mode:
            print(f"Your calibration was made for {saved_mode} aiming but you are "
                  f"set to {settings.aim_mode}. The two map different points on "
                  f"your hand, so recalibrating.")
            need_calibration = True
            model = None

    if model is not None:
        # A calibration this far out is not worth running with; it was likely
        # captured through mis-fired pinches, and the cursor will miss by inches.
        accuracy = model.meta.get("loocv_rms_px")
        if accuracy is not None and accuracy > model.screen[1] * 0.03:
            print(f"Your saved calibration is only accurate to +/-{accuracy:.0f} px "
                  f"({accuracy / model.screen[1] * 100:.0f}% of screen height), "
                  f"which is unusable. Recalibrating.")
            need_calibration = True
            model = None

    if model is not None and tuple(model.screen) != primary_screen_size():
        print(f"Screen is now {primary_screen_size()[0]}x{primary_screen_size()[1]}, "
              f"but calibration was done at {model.screen[0]}x{model.screen[1]}. "
              "Recalibrating.")
        need_calibration = True
        model = None

    with HandTracker(settings) as tracker:
        print("Camera ready. Warming up the tracker...")
        if not tracker.wait_until_ready(20.0):
            print("The tracker never produced a frame. Try `run.bat check`.")
            return 1

        if need_calibration:
            model = calibrate(tracker, settings)
            if model is None:
                print("Calibration cancelled.")
                return 1
            if args.command == "calibrate":
                return 0

        assert model is not None
        GestureController(tracker, model, settings).run()
    print("Stopped. Your mouse is yours again.")
    return 0


def entry() -> None:
    from .camera import CameraError
    from .tracker import ModelError

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (CameraError, ModelError) as exc:
        # Expected, actionable failures: the message is the whole story, and a
        # traceback in a double-clicked .bat window only buries it.
        print(f"\n{exc}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - unexpected: show everything
        print(f"\nUnexpected error: {exc}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
