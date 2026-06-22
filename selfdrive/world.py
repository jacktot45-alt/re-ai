"""The driving world: steps the car, renders the 4 cameras and tracks metrics.

This is the environment both the data collector and the closed-loop viewer use.
"""
from . import config
from .car import Car
from .cameras import render_features, render_display
from .expert import ExpertDriver


class World:
    def __init__(self, track):
        self.track = track
        self.car = Car()
        self.expert = ExpertDriver(track)
        self.hint = 0
        self.steps = 0

    def reset(self, progress_i=0, lateral=0.0, heading_err=0.0, speed=None):
        self.car.reset_on_track(self.track, progress_i, lateral, heading_err, speed)
        self.expert.reset()
        self.expert.hint = progress_i
        self.hint = progress_i
        self.steps = 0

    # ------------------------------------------------------------- perception
    def _ego(self):
        """Ego-state features appended to the camera pixels (normalised speed,
        repeated EGO_REPLICAS times so it carries weight against the pixels)."""
        return [self.car.v / config.EGO_SPEED_REF - 0.5] * config.EGO_REPLICAS

    def observe_features(self):
        """4-camera pixels + ego speed -> the full network input vector."""
        return render_features(self.car, self.track) + self._ego()

    def observe_full(self):
        """Both display images (full-res) and the matching net input vector."""
        images = render_display(self.car, self.track)
        feats = render_features(self.car, self.track) + self._ego()
        return images, feats

    # ----------------------------------------------------------------- expert
    def expert_action(self):
        steer, throttle, lateral, s = self.expert.act(self.car)
        return steer, throttle

    # -------------------------------------------------------------- dynamics
    def step(self, steer, throttle):
        self.car.step(steer, throttle)
        self.steps += 1
        i, lateral, th, s = self.track.localize(self.car.x, self.car.y, self.hint)
        self.hint = i
        on_road = abs(lateral) <= self.track.half_width
        finished = i >= self.track.n - 3
        done = finished or (self.steps >= config.MAX_STEPS) or \
            (abs(lateral) > self.track.half_width + 3.0)   # fully off the road
        info = {
            "index": i,
            "lateral": lateral,
            "progress_s": s,
            "on_road": on_road,
            "finished": finished,
            "done": done,
            "speed": self.car.v,
        }
        return info
