#!/usr/bin/env python3
"""One command to do everything: train, then drive and build the viewer.

    python3 run_all.py            # full pipeline
    python3 run_all.py --quick    # fast, lower-quality demo

When it finishes, open viewer.html in your browser.
"""
import subprocess
import sys

def main():
    quick = "--quick" in sys.argv[1:]
    py = sys.executable
    train = [py, "train.py"] + (["--quick"] if quick else [])
    print(">>> " + " ".join(train))
    if subprocess.call(train) != 0:
        raise SystemExit("training failed")
    drive = [py, "drive.py", "--viewer"]
    print(">>> " + " ".join(drive))
    if subprocess.call(drive) != 0:
        raise SystemExit("drive failed")
    print("\nAll done. Open viewer.html in your browser to watch it drive.")

if __name__ == "__main__":
    main()
