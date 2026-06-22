#!/usr/bin/env python3
"""Drive with the trained network and check that it works.

For each scenario it runs the network in closed loop, prints a PASS/FAIL
report, saves the full replay to runs/run_<track>.json, and (with --viewer)
builds a single self-contained viewer.html you can open in any browser.

Usage:
    python3 drive.py --viewer            # drive highway+city, build viewer
    python3 drive.py --track city        # just the city scenario
    python3 drive.py --model my.json
"""
import argparse
import json
import os

from selfdrive.evaluate import run_and_record
from selfdrive.nn import MLP
from selfdrive.track import get_track
from selfdrive.viewer import write_viewer
from selfdrive import config

TRACKS = ["highway", "city"]


def verdict(m):
    return "PASS" if (m["finished"] and m["offroad_steps"] == 0) else "NEEDS WORK"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=config.MODEL_PATH)
    ap.add_argument("--track", choices=TRACKS + ["both"], default="both")
    ap.add_argument("--viewer", action="store_true",
                    help="also build a self-contained viewer.html")
    ap.add_argument("--viewer-out", default="viewer.html")
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit("Model '%s' not found. Run:  python3 train.py" % args.model)

    net = MLP.load(args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    chosen = TRACKS if args.track == "both" else [args.track]

    runs = {}
    print("=" * 64)
    all_pass = True
    for name in chosen:
        tr = get_track(name)
        run, m = run_and_record(tr, net)
        runs[name] = run
        path = os.path.join(args.out_dir, "run_%s.json" % name)
        with open(path, "w") as f:
            json.dump(run, f, separators=(",", ":"))
        v = verdict(m)
        all_pass = all_pass and (v == "PASS")
        print("%-8s | %-9s | completed %5.1f%% (%6.1fm) | lane dev mean %.2fm "
              "max %.2fm | off-road %d | %d steps"
              % (name, v, m["completed_pct"], m["completed_m"], m["mean_dev"],
                 m["max_dev"], m["offroad_steps"], m["steps"]))
        print("         | saved replay -> %s" % path)
    print("=" * 64)
    print("RESULT: %s" % ("ALL SCENARIOS PASS ✓" if all_pass
                          else "some scenarios need more training"))

    if args.viewer:
        write_viewer(runs, args.viewer_out)
        print("\nViewer written -> %s" % args.viewer_out)
        print("Open it in a browser:  file://%s" % os.path.abspath(args.viewer_out))


if __name__ == "__main__":
    main()
