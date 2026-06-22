"""Closed-loop evaluation: let the trained network drive and measure how well.

`evaluate` returns metrics only (used to track training progress).
`run_and_record` additionally captures every frame (car pose + 4 camera images)
so the HTML viewer can replay the drive.
"""
from . import config
from .world import World


def _quant_image(img):
    """Encode an HxW float image as a list of H strings of digits 0-9."""
    rows = []
    for row in img:
        rows.append("".join(str(min(9, int(v * 9.999))) for v in row))
    return rows


def evaluate(track, net, start_lateral=0.0, start_heading=0.0):
    """Drive from the start with the network and return metrics."""
    world = World(track)
    world.reset(0, lateral=start_lateral, heading_err=start_heading)
    sum_dev = 0.0
    max_dev = 0.0
    offroad = 0
    n = 0
    finished = False
    last_s = 0.0
    while True:
        feats = world.observe_features()
        out = net.predict(feats)
        info = world.step(out[0], out[1])
        ad = abs(info["lateral"])
        sum_dev += ad
        max_dev = max(max_dev, ad)
        if not info["on_road"]:
            offroad += 1
        last_s = info["progress_s"]
        n += 1
        if info["finished"]:
            finished = True
            break
        if info["done"]:
            break
    return {
        "track": track.name,
        "completed_m": round(last_s, 1),
        "completed_pct": round(100.0 * last_s / track.length, 1),
        "mean_dev": round(sum_dev / max(1, n), 3),
        "max_dev": round(max_dev, 3),
        "offroad_steps": offroad,
        "steps": n,
        "finished": finished,
        "half_width": track.half_width,
    }


def run_and_record(track, net, stride=2):
    """Drive with the network and capture a replay for the viewer.

    Physics runs every step; only every `stride`-th step is recorded (with its
    full-resolution camera images) to keep the viewer file small. Metrics are
    measured over every step.
    """
    from .cameras import render_display
    world = World(track)
    world.reset(0)
    frames = []
    metrics_dev = 0.0
    max_dev = 0.0
    offroad = 0
    total_steps = 0
    finished = False
    last_s = 0.0
    while True:
        feats = world.observe_features()
        out = net.predict(feats)
        steer, throttle = out[0], out[1]
        # Record (with cameras) only on stride boundaries.
        if total_steps % stride == 0:
            images = render_display(world.car, track)
            frames.append({
                "x": round(world.car.x, 2),
                "y": round(world.car.y, 2),
                "th": round(world.car.theta, 4),
                "v": round(world.car.v, 2),
                "steer": round(steer, 3),
                "thr": round(throttle, 3),
                "cams": {name: _quant_image(images[name]) for name in images},
            })
        info = world.step(steer, throttle)
        # Attach post-step lane offset / on-road flag to the last recorded frame.
        if frames and total_steps % stride == 0:
            frames[-1]["lat"] = round(info["lateral"], 3)
            frames[-1]["on"] = 1 if info["on_road"] else 0
        ad = abs(info["lateral"])
        metrics_dev += ad
        max_dev = max(max_dev, ad)
        if not info["on_road"]:
            offroad += 1
        last_s = info["progress_s"]
        total_steps += 1
        if info["finished"]:
            finished = True
            break
        if info["done"]:
            break

    metrics = {
        "track": track.name,
        "completed_m": round(last_s, 1),
        "completed_pct": round(100.0 * last_s / track.length, 1),
        "mean_dev": round(metrics_dev / max(1, total_steps), 3),
        "max_dev": round(max_dev, 3),
        "offroad_steps": offroad,
        "steps": total_steps,
        "finished": finished,
        "half_width": track.half_width,
    }
    # Downsample the centreline for drawing (keep file small).
    step = max(1, track.n // 600)
    poly = [[round(track.pts[i][0], 1), round(track.pts[i][1], 1)]
            for i in range(0, track.n, step)]
    run = {
        "track": {
            "name": track.name,
            "pts": poly,
            "half_width": track.half_width,
            "lane_width": track.lane_width,
            "length": round(track.length, 1),
            "speed_limit": track.speed_limit,
        },
        "cam_layout": [{"name": nm, "w": config.CAM_W, "h": config.CAM_H}
                       for (nm, _y, _f, _r) in config.CAMERAS],
        "dt": round(config.DT * stride, 4),
        "frames": frames,
        "metrics": metrics,
    }
    return run, metrics
