#!/usr/bin/env python3
"""Fast smoke test of the whole pipeline (no full training required).

Checks that the simulator, expert, cameras, network and viewer all work end to
end. Runs in a few seconds. Exits non-zero on failure.

    python3 selftest.py
"""
import json
import os
import tempfile

from selfdrive.track import get_track
from selfdrive.car import Car
from selfdrive.expert import ExpertDriver
from selfdrive.world import World
from selfdrive.cameras import input_size, render_features
from selfdrive.nn import MLP
from selfdrive.evaluate import run_and_record
from selfdrive.viewer import build_html
from selfdrive import config


def check(name, cond):
    print(("  PASS " if cond else "  FAIL ") + name)
    if not cond:
        raise SystemExit("selftest failed: " + name)


def main():
    print("1) Expert can complete both tracks")
    for name in ("highway", "city"):
        tr = get_track(name)
        car = Car(); car.reset_on_track(tr, 0)
        ex = ExpertDriver(tr); ex.reset()
        offroad = 0; finished = False
        for _ in range(config.MAX_STEPS):
            st, th, lat, s = ex.act(car); car.step(st, th)
            if abs(lat) > tr.half_width:
                offroad += 1
            i, _, _, _ = tr.localize(car.x, car.y, ex.hint)
            if i >= tr.n - 3:
                finished = True; break
        check("%s: expert finished, no off-road" % name, finished and offroad == 0)

    print("2) Cameras + ego produce the expected input size")
    tr = get_track("highway")
    w = World(tr); w.reset(0)
    feats = w.observe_features()
    check("observation length == input_size()", len(feats) == input_size())

    print("3) Network backprop reduces loss")
    import random
    rng = random.Random(0)
    X = [[rng.uniform(-1, 1) for _ in range(input_size())] for _ in range(120)]
    Y = [[0.5 * x[0] - 0.3 * x[1], -0.4 * x[2]] for x in X]
    net = MLP([input_size(), 16, 2], seed=1)
    hist = net.train(X, Y, epochs=20, batch=16, lr=0.05, seed=2)
    check("loss decreased", hist[-1] < hist[0] * 0.6)

    print("4) Closed-loop run records frames and viewer builds")
    run, metrics = run_and_record(tr, net)
    check("run has frames", len(run["frames"]) > 5)
    html = build_html({"highway": run})
    # map + minimap are static; the 4 camera canvases are created in JS.
    check("viewer html well-formed",
          '"frames"' in html and '<canvas id="map"' in html
          and "cam_" in html)
    # The embedded payload should be valid JSON.
    payload = html.split("const RUNS = ", 1)[1].split(";\n", 1)[0]
    parsed = json.loads(payload)
    check("embedded RUNS JSON parses", "highway" in parsed)

    print("5) Model save/load round-trips")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.json")
        net.save(p)
        net2 = MLP.load(p)
        a = net.predict(X[0]); b = net2.predict(X[0])
        check("reloaded model matches", abs(a[0] - b[0]) < 1e-9)

    print("\nALL SELFTESTS PASSED ✓")


if __name__ == "__main__":
    main()
