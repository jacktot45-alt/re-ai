"""The 'expert' driver used to generate training labels.

It is a classic pure-pursuit steering controller plus a curvature-aware speed
controller. The neural network never sees this code -- it only sees the camera
images and tries to imitate the steer/throttle commands the expert produced.
"""
import math
from . import config
from .geometry import clamp, wrap_angle


def _curvature_ahead(track, i, span=25):
    """Estimate path curvature a little ahead, to slow down for bends."""
    j = track.index_at_arc(i, span)
    dth = abs(wrap_angle(track.th[j] - track.th[i]))
    ds = max(1e-3, track.s[j] - track.s[i])
    return dth / ds


class ExpertDriver:
    def __init__(self, track):
        self.track = track
        self.hint = 0

    def reset(self):
        self.hint = 0

    def act(self, car):
        tr = self.track
        i, lateral, th, s = tr.localize(car.x, car.y, self.hint)
        self.hint = i

        # --- Pure-pursuit steering -------------------------------------
        lookahead = clamp(0.9 * car.v + 5.0, 6.0, 30.0)
        ti = tr.index_at_arc(i, lookahead)
        tx, ty = tr.pts[ti]
        # Angle to the target point in the car's frame.
        ang_to_target = math.atan2(ty - car.y, tx - car.x)
        alpha = wrap_angle(ang_to_target - car.theta)
        ld = max(1.0, math.hypot(tx - car.x, ty - car.y))
        steer = math.atan2(2.0 * config.WHEELBASE * math.sin(alpha), ld)
        steer_cmd = clamp(steer / config.MAX_STEER, -1.0, 1.0)

        # --- Speed control ---------------------------------------------
        kappa = _curvature_ahead(tr, i)
        a_lat_max = 3.0                       # comfortable lateral accel
        v_curve = math.sqrt(a_lat_max / kappa) if kappa > 1e-4 else 1e3
        v_target = min(tr.speed_limit, v_curve)
        accel = 0.6 * (v_target - car.v)
        if accel >= 0.0:
            throttle_cmd = clamp(accel / config.ACCEL_MAX, 0.0, 1.0)
        else:
            throttle_cmd = clamp(accel / config.BRAKE_MAX, -1.0, 0.0)

        return steer_cmd, throttle_cmd, lateral, s
