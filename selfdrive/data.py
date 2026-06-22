"""Collect training data by letting the expert drive (behavioural cloning).

To make the learned policy robust in closed loop we do two important things:

  * Recovery starts: episodes begin from a range of lateral offsets and
    heading errors, so the network sees how to steer back to the centre.
  * Noise injection: the executed command is occasionally perturbed so the car
    drifts off-centre, but the *label* is always the expert's correct command
    at the visited state. This is the classic fix for compounding error.
"""
import random
from .world import World


def collect_expert(track, episodes=10, steps_per_ep=260, seed=0,
                   noise=0.20, perturb=True):
    """Return (X, Y): features and [steer, throttle] labels."""
    rng = random.Random(seed)
    world = World(track)
    X, Y = [], []
    for ep in range(episodes):
        if perturb:
            # Spread starts across the WHOLE track so every turn is covered.
            start = rng.randint(0, max(1, track.n - 5))
            lateral = rng.uniform(-track.half_width * 0.6, track.half_width * 0.6)
            head_err = rng.uniform(-0.25, 0.25)
            speed = rng.uniform(0.2, 0.8) * track.speed_limit
        else:
            start, lateral, head_err, speed = 0, 0.0, 0.0, None
        world.reset(start, lateral, head_err, speed)

        for _ in range(steps_per_ep):
            feats = world.observe_features()
            steer, throttle = world.expert_action()      # the label
            X.append(feats)
            Y.append([steer, throttle])

            # Execute with occasional exploration noise to visit off-centre states.
            ex_steer = steer + (rng.uniform(-noise, noise) if rng.random() < 0.5 else 0.0)
            ex_throttle = throttle + rng.uniform(-0.1, 0.1)
            info = world.step(ex_steer, ex_throttle)
            if info["done"]:
                break
    return X, Y


def collect_balanced(track, target_samples, seed=0, steps_per_ep=320):
    """Collect roughly `target_samples` expert demonstrations on one track."""
    X, Y = [], []
    k = 0
    while len(X) < target_samples:
        Xi, Yi = collect_expert(track, episodes=2, steps_per_ep=steps_per_ep,
                                seed=seed * 1000 + k)
        X += Xi
        Y += Yi
        k += 1
    return X[:target_samples], Y[:target_samples]


def collect_dagger(track, net, episodes=4, steps_per_ep=400, seed=0):
    """DAgger aggregation: the *network* drives, the expert labels each state.

    This teaches the network to correct the specific mistakes it makes in
    closed loop -- the single biggest win for stable driving.
    """
    rng = random.Random(seed)
    world = World(track)
    X, Y = [], []
    span = max(1, track.n - 5)
    for ep in range(episodes):
        # Spread starts evenly across the whole track (plus jitter) so every
        # turn the network might fail on gets corrective labels.
        start = int((ep + rng.random()) / episodes * span)
        world.reset(start, lateral=rng.uniform(-1.5, 1.5),
                    heading_err=rng.uniform(-0.1, 0.1))
        for _ in range(steps_per_ep):
            feats = world.observe_features()
            steer, throttle = world.expert_action()       # expert label
            X.append(feats)
            Y.append([steer, throttle])
            out = net.predict(feats)                       # network drives
            info = world.step(out[0], out[1])
            if info["done"]:
                break
    return X, Y
