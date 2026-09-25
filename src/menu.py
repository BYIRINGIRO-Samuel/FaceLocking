# src/menu.py
"""
Single entry point — pick what to run without remembering modules.

    python -m src.menu
    python -m src.menu --cam 2
"""
from __future__ import annotations

import argparse
import runpy
import sys


ACTIONS = [
    ("1", "Camera test", "src.camera"),
    ("2", "Enroll a face (save to DB)", "src.enroll"),
    ("3", "Live recognition (names / Unknown)", "src.recognize"),
    ("4", "Full demo — lock + expression + multi-face + nose", "src.identity_lock_track"),
    ("5", "Expression calibration (raw numbers)", "src.calibrate_expression"),
    ("6", "Servo MQTT test (needs ESP board)", "src.servo_link"),
    ("7", "Track & follow servo (needs ESP board)", "src.track_and_follow"),
]


def _print_menu(cam: int):
    print()
    print("=" * 56)
    print("  Falcon Eye — choose an action")
    print(f"  Default camera index: {cam}  (change with --cam N)")
    print("=" * 56)
    for key, title, _mod in ACTIONS:
        print(f"  [{key}]  {title}")
    print("  [c]  Change camera index")
    print("  [q]  Quit")
    print("=" * 56)


def _run_module(mod: str, cam: int):
    # Pass --cam through for modules that accept it
    argv_backup = sys.argv[:]
    try:
        if mod in {
            "src.camera",
            "src.enroll",
            "src.recognize",
            "src.identity_lock_track",
            "src.calibrate_expression",
        }:
            sys.argv = [mod, "--cam", str(cam)]
        else:
            sys.argv = [mod]
        print(f"\n>>> Starting {mod}  (press q in the video window to return here)\n")
        runpy.run_module(mod, run_name="__main__")
    except SystemExit:
        # argparse / scripts may call sys.exit — ignore and return to menu
        pass
    except Exception as e:
        print(f"\n[menu] {mod} failed: {type(e).__name__}: {e}\n")
    finally:
        sys.argv = argv_backup
        # Make sure OpenCV windows are gone before next action
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass
    print("\n<<< Back to menu.\n")


def main():
    parser = argparse.ArgumentParser(description="Falcon Eye launcher menu")
    parser.add_argument("--cam", type=int, default=2, help="Camera index for demos")
    args, _unknown = parser.parse_known_args()

    while True:
        _print_menu(args.cam)
        choice = input("Select option: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Bye.")
            break

        if choice == "c":
            raw = input(f"New camera index [{args.cam}]: ").strip()
            if raw.isdigit():
                args.cam = int(raw)
                print(f"Camera set to {args.cam}")
            continue

        match = next((a for a in ACTIONS if a[0] == choice), None)
        if not match:
            print("Invalid choice. Pick a number or q.")
            continue

        _key, _title, mod = match
        _run_module(mod, args.cam)


if __name__ == "__main__":
    main()
