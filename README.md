# 🚗 selfdrive — train a 4‑camera neural‑net driver (highway + city)

Train a small neural network to drive a **simulated** car that has **4 cameras**
(front / left / right / rear) on a **highway** and a **city** course, then watch
it drive in a browser‑based viewer that shows the cameras, a top‑down map and
live telemetry — so you can *see* that it works.

> ⚠️ **This is a driving simulator for learning.** It trains a neural network in
> a toy 2‑D simulation. It is **not** software for a real vehicle and must
> **never** be connected to a real car. Self‑driving a real Tesla (or any car)
> requires safety‑critical engineering, validation and hardware that are far
> outside the scope of an educational demo.

It uses **only the Python standard library** — no numpy, no PyTorch, no
internet. If you can run `python3`, you can run this.

---

## Quick start

```bash
# 1) Train (a few minutes) and build the viewer in one go:
python3 run_all.py

# 2) Open the viewer that was created:
#    open viewer.html        (macOS)
#    xdg-open viewer.html    (Linux)
#    start viewer.html       (Windows)
```

Want a fast first look? `python3 run_all.py --quick` trains a rougher model in
about a minute.

Prefer to run the steps yourself:

```bash
python3 train.py            # learns to drive, writes selfdrive_model.json
python3 drive.py --viewer   # tests it, writes viewer.html + runs/*.json
```

---

## What you’ll see

`drive.py` prints a report like this and marks each scenario **PASS** when the
car finishes the course without leaving the road:

```
================================================================
highway  | PASS      | completed 100.0% ( 909.0m) | lane dev mean 0.10m max 0.45m | off-road 0 |  640 steps
city     | PASS      | completed 100.0% (1073.0m) | lane dev mean 0.50m max 1.40m | off-road 0 | 2150 steps
================================================================
RESULT: ALL SCENARIOS PASS ✓
```

(Exact numbers vary a little with the random seed; both scenarios should reach
**PASS** — 100% completed, 0 off-road — after training.)

`viewer.html` is a self‑contained dashboard (no server needed). It shows:

* a **follow‑cam top‑down view** of the car driving, with its trail,
* a **minimap** of the whole route,
* the **4 camera feeds** — exactly the low‑resolution images the network drives
  from,
* **telemetry**: speed, steering, throttle/brake, lane offset, on/off‑road,
* **play / pause / scrub** controls and a **PASS / FAIL** verdict.

---

## How it works

```
 4 cameras ─┐
 (images)   ├─► neural network ──► steering + throttle ──► car moves ──► repeat
 speed ─────┘     (MLP)            (both in [-1, 1])
```

The cameras show *where* the road goes; the speedometer tells the net *how
fast* it is going (needed to decide throttle vs brake). Together they are the
network's input.

1. **Simulator** (`selfdrive/`) — a car (kinematic bicycle model) on a
   procedurally generated **highway** (long gentle curves, ~29 m/s) and **city**
   route (sweeping 90° turns, ~10 m/s). Each of the 4 cameras renders a small
   grayscale image by projecting the road ahead of it.
2. **Expert driver** (`expert.py`) — a classic pure‑pursuit + speed controller
   that follows the lane. It is only used to *generate training labels*; the
   network never sees it.
3. **Behavioural cloning** (`train.py`) — the network learns to map the 4 camera
   images (plus its speed) to the steering/throttle the expert used. Training
   data is collected **across the whole track** and from **recovery situations**
   (starting off‑centre) so the net learns to steer back to the lane.
4. **DAgger** — the network then drives on its own, the expert labels the states
   it actually visits, and we retrain. This fixes the small mistakes that
   otherwise snowball, and is what makes the closed‑loop driving solid.
5. **Viewer** (`drive.py` → `viewer.py`) — runs the trained net in closed loop,
   records the drive, and writes the HTML dashboard.

The network (`nn.py`) is a tiny multi‑layer perceptron with hand‑written
back‑propagation — small on purpose, since the images are low‑resolution and the
task is smooth.

---

## Project layout

```
re-ai/
├── run_all.py            one command: train + drive + viewer
├── train.py              train the network (behavioural cloning + DAgger)
├── drive.py              closed-loop test + PASS/FAIL report + viewer
├── selfdrive/
│   ├── config.py         all the knobs (camera size, network, learning rate…)
│   ├── geometry.py       small math helpers
│   ├── track.py          highway & city track generation
│   ├── car.py            bicycle-model vehicle dynamics
│   ├── cameras.py        the 4 synthetic cameras
│   ├── expert.py         pure-pursuit expert (label source)
│   ├── world.py          steppable environment + metrics
│   ├── data.py           dataset collection (recovery + DAgger)
│   ├── nn.py             pure-Python MLP with backprop, save/load
│   ├── evaluate.py       closed-loop scoring + replay recording
│   └── viewer.py         builds the self-contained viewer.html
├── runs/                 recorded drives (JSON) used by the viewer
└── selfdrive_model.json  the trained weights (created by train.py)
```

---

## Tuning

Open `selfdrive/config.py`. Useful knobs:

| Setting | Meaning |
|---|---|
| `CAM_W`, `CAM_H` | camera resolution (bigger = more detail, slower) |
| `CAMERAS` | the 4 mounts (name, yaw, field of view, range) |
| `HIDDEN_LAYERS` | network size, e.g. `[64, 32]` |
| `EPOCHS`, `LEARNING_RATE`, `LR_DECAY` | training schedule |
| highway / city speed & width | in `track.py` |

`python3 train.py --help` and `python3 drive.py --help` list the CLI options
(samples, DAgger rounds, output paths, which scenario, etc.).

---

## FAQ

**Can I use this to make my real Tesla drive itself?** No. This is a simulation
for learning how end‑to‑end (camera → steering) driving networks are trained and
evaluated. Real autonomy is a safety‑critical engineering problem.

**Why no GPU / PyTorch?** So it runs anywhere with zero setup. The ideas
(behavioural cloning, DAgger, closed‑loop evaluation) are exactly the same ones
used in the real research — just at a tiny, readable scale.

**It went off the road on city.** Train longer: `python3 train.py --rounds 4`,
or raise `--samples`. The DAgger rounds are what make the hard turns reliable.
