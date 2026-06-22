"""Four synthetic cameras (front / left / right / rear).

Each camera renders a small grayscale image of the road by projecting a grid
of ground points out in front of it and asking the track "is there road here,
and is it a lane marking?".

The renderer is resolution-generic:
  * the NETWORK sees a low-resolution render (fast, what it learns from),
  * the VIEWER gets a higher-resolution render of the same projection (pretty).

Pixel intensity convention (0 = dark, 1 = bright):
    ~0.18  road surface (asphalt)
    ~0.85  off-road / background / sky
    ~0.70  lane boundary line
    ~0.95  solid road edge line
    ~1.00  dashed centre line
"""
import math
from . import config

# Network input resolution (per camera) derived from the display resolution.
NET_W = config.CAM_W // config.NET_POOL
NET_H = config.CAM_H // config.NET_POOL

# Cache of precomputed projection tables, keyed by (W, H, fov, far).
_TABLES = {}


def _build_table(W, H, fov, far):
    near = config.CAM_NEAR
    table = []
    for r in range(H):
        t = (H - 1 - r) / float(H - 1)         # 0 near (bottom) .. 1 far (top)
        d = near + (far - near) * (t ** 1.7)    # perspective-ish row spacing
        for c in range(W):
            a = (0.5 - c / float(W - 1)) * fov  # col 0 = left edge -> +bearing
            table.append((d * math.cos(a), d * math.sin(a)))
    return table


def _table_for(W, H, fov, far):
    key = (W, H, round(fov, 4), round(far, 2))
    tbl = _TABLES.get(key)
    if tbl is None:
        tbl = _build_table(W, H, fov, far)
        _TABLES[key] = tbl
    return tbl


def _pixel_value(track, lateral, s, on_path):
    if not on_path:
        return 0.85                             # background / sky
    al = lateral if lateral >= 0 else -lateral
    if al > track.half_width:
        return 0.85                             # off-road
    if track.half_width - al < 0.30:
        return 0.95                             # solid road edge
    if al < 0.22 and int(s / 3.0) % 2 == 0:
        return 1.00                             # dashed centre line
    lw = track.lane_width
    k = round(lateral / lw)
    if k != 0 and abs(lateral - k * lw) < 0.20 and abs(k * lw) < track.half_width:
        return 0.70                             # lane boundary line
    return 0.18                                 # asphalt


def render_camera(car, track, yaw_offset, fov, far, W, H):
    """Render one camera as a flat list of W*H floats (row-major, top row first)."""
    table = _table_for(W, H, fov, far)
    yaw = car.theta + yaw_offset
    cx, cy = car.x, car.y
    cyaw = math.cos(yaw)
    syaw = math.sin(yaw)
    ng = track.nearest_global
    pv = _pixel_value
    out = []
    for (fwd, left) in table:
        X = cx + fwd * cyaw - left * syaw
        Y = cy + fwd * syaw + left * cyaw
        lateral, s, on_path = ng(X, Y)
        out.append(pv(track, lateral, s, on_path))
    return out


def render_features(car, track):
    """Low-resolution render of all four cameras -> flat network input vector.

    Values are centred to roughly [-0.5, 0.5]. Order follows config.CAMERAS.
    """
    feats = []
    for (name, yaw, fov, far) in config.CAMERAS:
        flat = render_camera(car, track, yaw, fov, far, NET_W, NET_H)
        for v in flat:
            feats.append(v - 0.5)
    return feats


def render_display(car, track):
    """Full-resolution render of all four cameras for the viewer.

    Returns dict name -> 2D image (list of H rows of W floats).
    """
    out = {}
    for (name, yaw, fov, far) in config.CAMERAS:
        flat = render_camera(car, track, yaw, fov, far, config.CAM_W, config.CAM_H)
        img = [flat[r * config.CAM_W:(r + 1) * config.CAM_W]
               for r in range(config.CAM_H)]
        out[name] = img
    return out


def feature_size():
    """Number of camera pixels fed to the network (all 4 cameras)."""
    return NET_W * NET_H * len(config.CAMERAS)


# Extra (non-camera) inputs appended to the observation: the ego speed, fed as
# several copies (config.EGO_REPLICAS) so it is not drowned out by the pixels.
EGO_FEATURES = config.EGO_REPLICAS


def input_size():
    """Total network input length: camera pixels + ego state."""
    return feature_size() + EGO_FEATURES
