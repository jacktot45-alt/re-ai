#!/usr/bin/env python3
"""Train the 4-camera neural-net driver on the highway and city tracks.

Pipeline:
  1. Collect a balanced set of expert demonstrations on both tracks
     (with recovery starts + noise injection so the net learns to correct).
  2. Behavioural cloning: fit the network to imitate the expert.
  3. DAgger rounds: let the network drive, have the expert label the states it
     actually visits, then retrain on the balanced aggregate. This is what
     makes the closed-loop driving robust.

The trained weights are written to a JSON file (default: selfdrive_model.json).

Usage:
    python3 train.py                 # full training (a few minutes)
    python3 train.py --quick         # faster, lower quality (for a quick look)
    python3 train.py --out my.json --rounds 3
"""
import argparse
import time

from selfdrive import config
from selfdrive.cameras import feature_size, input_size
from selfdrive.data import collect_balanced, collect_dagger
from selfdrive.evaluate import evaluate
from selfdrive.nn import MLP
from selfdrive.track import get_track

TRACKS = ["highway", "city"]


def _report(tag, net, tracks):
    """Evaluate both tracks; print a line; return (score, all_pass).

    Score rewards completing the tracks and penalises off-road steps, so the
    best checkpoint across training rounds can be kept even if rounds oscillate.
    """
    parts = []
    ok = True
    score = 0.0
    for name, tr in tracks.items():
        m = evaluate(tr, net)
        passed = m["finished"] and m["offroad_steps"] == 0
        ok = ok and passed
        score += m["completed_pct"] - 0.1 * m["offroad_steps"]
        parts.append("%s %3.0f%% dev%.2f off%d%s" % (
            name[:4].upper(), m["completed_pct"], m["mean_dev"],
            m["offroad_steps"], " OK" if passed else ""))
    print("  [%s] %s  (score %.1f)" % (tag, " | ".join(parts), score))
    return score, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=config.MODEL_PATH, help="output model path")
    ap.add_argument("--samples", type=int, default=2400,
                    help="expert samples per track for behavioural cloning")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS,
                    help="epochs for the initial behavioural-cloning fit")
    ap.add_argument("--rounds", type=int, default=5, help="DAgger rounds")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--quick", action="store_true",
                    help="smaller/faster run for a quick demo")
    args = ap.parse_args()

    if args.quick:
        args.samples, args.epochs, args.rounds = 1200, 12, 1

    t0 = time.time()
    tracks = {name: get_track(name, args.seed) for name in TRACKS}
    print("Tracks: " + ", ".join("%s (%.0fm)" % (n, t.length)
                                 for n, t in tracks.items()))
    print("Network input = %d  (4 cameras x %d px  +  %d speed)" % (
        input_size(), feature_size() // 4, config.EGO_REPLICAS))

    # 1) Balanced expert data -------------------------------------------------
    print("\n[1/3] Collecting expert demonstrations ...")
    data = {}
    for i, (name, tr) in enumerate(tracks.items()):
        X, Y = collect_balanced(tr, args.samples, seed=args.seed + i)
        data[name] = [X, Y]
        print("  %-8s %d samples" % (name, len(X)))

    # 2) Behavioural cloning --------------------------------------------------
    print("\n[2/3] Behavioural cloning (%d epochs) ..." % args.epochs)
    net = MLP([input_size()] + config.HIDDEN_LAYERS + [2], seed=args.seed)
    X = data["highway"][0] + data["city"][0]
    Y = data["highway"][1] + data["city"][1]

    def log(ep, mse):
        if ep % 4 == 0 or ep == args.epochs - 1:
            print("  epoch %2d/%d  mse=%.4f" % (ep + 1, args.epochs, mse))
    net.train(X, Y, epochs=args.epochs, batch=config.BATCH_SIZE,
              lr=config.LEARNING_RATE, momentum=config.MOMENTUM,
              weight_decay=config.WEIGHT_DECAY, lr_decay=config.LR_DECAY,
              seed=args.seed, log=log)

    # Keep the BEST checkpoint across all rounds. Closed-loop driving can
    # oscillate between rounds, so we save whenever a checkpoint improves and
    # ship that one -- never just the last (possibly worse) round.
    best = {"score": -1e9, "tag": None}

    def keep_best(tag):
        score, ok = _report(tag, net, tracks)
        improved = score > best["score"]
        if improved:
            best["score"] = score
            best["tag"] = tag
            net.save(args.out)
            print("    -> new best, saved to %s" % args.out)
        return improved

    keep_best("after BC")

    # 3) DAgger ---------------------------------------------------------------
    # The city's sharp turns are the hard part, so DAgger is weighted heavily
    # toward city episodes. The clean initial expert demos are ALWAYS kept;
    # capped on-policy DAgger data is added on top. Closed-loop driving can
    # oscillate between rounds, hence the keep-best selection above.
    print("\n[3/3] DAgger refinement (%d rounds) ..." % args.rounds)
    dagger_eps = {"highway": 3, "city": 12}
    per_track = 4500            # training samples kept per track each round
    dag = {name: [[], []] for name in tracks}

    def build_set(name):
        """Clean base expert demos (always kept) + most recent DAgger data."""
        bX, bY = data[name]
        dX, dY = dag[name]
        room = max(0, per_track - len(bX))
        return bX + dX[-room:], bY + dY[-room:]

    for rnd in range(args.rounds):
        for i, (name, tr) in enumerate(tracks.items()):
            steps = 500 if name == "highway" else 600
            dX, dY = collect_dagger(tr, net, episodes=dagger_eps[name],
                                    steps_per_ep=steps,
                                    seed=args.seed + 50 + rnd * 10 + i)
            dag[name][0] += dX
            dag[name][1] += dY
        hX, hY = build_set("highway")
        cX, cY = build_set("city")
        XX, YY = hX + cX, hY + cY
        net.train(XX, YY, epochs=10, batch=config.BATCH_SIZE, lr=0.02,
                  momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY,
                  lr_decay=0.9, seed=args.seed + 200 + rnd)
        improved = keep_best("round %d" % (rnd + 1))
        if not improved:
            # Closed-loop driving oscillates; don't let a regressed round
            # poison the next one -- rewind to the best policy so far and let
            # the next DAgger round build on it instead.
            bestnet = MLP.load(args.out)
            net.W, net.b = bestnet.W, bestnet.b
            print("    (rewound to best: %s)" % best["tag"])

    print("\nSaved BEST model (%s, score %.1f) -> %s  (%.0fs total)"
          % (best["tag"], best["score"], args.out, time.time() - t0))
    print("Next:  python3 drive.py --viewer   then open viewer.html")


if __name__ == "__main__":
    main()
