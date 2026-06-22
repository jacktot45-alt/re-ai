"""Kinematic bicycle model of the car.

State: position (x, y), heading theta, speed v.
Controls (network / expert outputs, both in [-1, 1]):
    steer_cmd  -> front wheel angle = steer_cmd * MAX_STEER
    throttle   -> accelerate (>0) or brake (<0)
"""
import math
from . import config
from .geometry import clamp


class Car:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0

    def reset_on_track(self, track, progress_i=0, lateral=0.0, heading_err=0.0,
                       speed=None):
        """Place the car on the track, optionally offset for recovery data."""
        px, py = track.pts[progress_i]
        th = track.th[progress_i]
        # Offset sideways by `lateral` along the left-normal.
        self.x = px + (-math.sin(th)) * lateral
        self.y = py + (math.cos(th)) * lateral
        self.theta = th + heading_err
        self.v = track.speed_limit * 0.5 if speed is None else speed

    def step(self, steer_cmd, throttle_cmd, dt=config.DT):
        steer = clamp(steer_cmd, -1.0, 1.0) * config.MAX_STEER
        if throttle_cmd >= 0.0:
            accel = clamp(throttle_cmd, 0.0, 1.0) * config.ACCEL_MAX
        else:
            accel = clamp(throttle_cmd, -1.0, 0.0) * config.BRAKE_MAX

        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += self.v / config.WHEELBASE * math.tan(steer) * dt
        self.v = max(0.0, self.v + accel * dt)
