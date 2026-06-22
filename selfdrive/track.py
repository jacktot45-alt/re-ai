"""Procedural track generation: a HIGHWAY and a CITY route.

A Track is a smooth centreline (list of points) plus per-point heading and
cumulative arc length, a road half-width, lane width and a speed limit.
The `localize` method projects a world position onto the centreline, which is
used by the expert driver, the cameras and the metrics.
"""
import math
from .geometry import catmull_rom, wrap_angle


class Track:
    def __init__(self, name, pts, half_width, lane_width, speed_limit):
        self.name = name
        self.pts = pts                      # list of (x, y)
        self.half_width = half_width        # m, drivable distance from centre
        self.lane_width = lane_width        # m, used for lane-line rendering
        self.speed_limit = speed_limit      # m/s
        self.n = len(pts)

        # Per-point heading (tangent direction) and cumulative arc length.
        self.th = [0.0] * self.n
        self.s = [0.0] * self.n
        for i in range(self.n):
            a = pts[max(0, i - 1)]
            b = pts[min(self.n - 1, i + 1)]
            self.th[i] = math.atan2(b[1] - a[1], b[0] - a[0])
        for i in range(1, self.n):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            self.s[i] = self.s[i - 1] + math.hypot(dx, dy)
        self.length = self.s[-1]

        # Spatial hash grid for O(1) nearest-point queries (used by cameras).
        self._cell = 5.0
        self._grid = {}
        for i, (px, py) in enumerate(pts):
            key = (int(px // self._cell), int(py // self._cell))
            self._grid.setdefault(key, []).append(i)

    def nearest_global(self, x, y):
        """Fast global nearest-centreline query for arbitrary points.

        Returns (lateral, progress_s, on_path) where `on_path` is False when
        the point falls before the start or beyond the end of the route
        (i.e. there is no road there -- render it as sky/background).
        """
        cx = int(x // self._cell)
        cy = int(y // self._cell)
        best_i = -1
        best_d = float("inf")
        ring = 1
        while best_i < 0 and ring <= 6:
            for gx in range(cx - ring, cx + ring + 1):
                for gy in range(cy - ring, cy + ring + 1):
                    for i in self._grid.get((gx, gy), ()):  # candidates
                        dx = x - self.pts[i][0]
                        dy = y - self.pts[i][1]
                        d = dx * dx + dy * dy
                        if d < best_d:
                            best_d = d
                            best_i = i
            ring += 1
        if best_i < 0:
            return 0.0, 0.0, False
        th = self.th[best_i]
        dx = x - self.pts[best_i][0]
        dy = y - self.pts[best_i][1]
        along = math.cos(th) * dx + math.sin(th) * dy
        lateral = -math.sin(th) * dx + math.cos(th) * dy
        # Off the ends of the route -> no road here.
        if (best_i == 0 and along < 0.0) or (best_i == self.n - 1 and along > 0.0):
            return lateral, self.s[best_i], False
        return lateral, self.s[best_i] + along, True

    def localize(self, x, y, hint=0):
        """Find the nearest centreline point to (x, y).

        Returns (index, lateral, heading, progress_s) where `lateral` is the
        signed perpendicular offset (positive = left of travel direction).
        A `hint` index keeps the search local and fast in the control loop.
        """
        lo = max(0, hint - 8)
        hi = min(self.n, hint + 120)
        best_i = lo
        best_d = float("inf")
        for i in range(lo, hi):
            dx = x - self.pts[i][0]
            dy = y - self.pts[i][1]
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best_i = i
        th = self.th[best_i]
        dx = x - self.pts[best_i][0]
        dy = y - self.pts[best_i][1]
        # Signed lateral offset using the left-normal of the tangent.
        lateral = -math.sin(th) * dx + math.cos(th) * dy
        return best_i, lateral, th, self.s[best_i]

    def index_at_arc(self, start_i, ahead):
        """Index of the point roughly `ahead` metres past point `start_i`."""
        target = self.s[start_i] + ahead
        i = start_i
        while i < self.n - 1 and self.s[i] < target:
            i += 1
        return i

    def heading_error(self, i, car_theta):
        return wrap_angle(self.th[i] - car_theta)


def build_highway(seed=0):
    """A long, gently curving multi-lane highway."""
    pts = []
    s = 0.0
    ds = 1.0
    length = 900.0
    while s <= length:
        # Gentle, long-wavelength curves -> easy, high-speed driving.
        y = (18.0 * math.sin(s / 130.0) +
             7.0 * math.sin(s / 47.0 + 1.2))
        pts.append((s, y))
        s += ds
    return Track("highway", pts, half_width=6.0, lane_width=3.7, speed_limit=29.0)


def build_city(seed=0):
    """A winding city route through a grid with sweeping 90-degree turns.

    Legs are ~100 m so the Catmull-Rom corners have a moderate radius the
    low-resolution camera + small network can actually learn to take.
    """
    waypoints = [
        (0, 0), (120, 0), (120, 105), (235, 105), (235, -15),
        (350, -15), (350, 110), (245, 110), (245, 215), (370, 215),
    ]
    pts = catmull_rom(waypoints, samples_per_segment=46)
    return Track("city", pts, half_width=5.0, lane_width=3.5, speed_limit=10.0)


def get_track(name, seed=0):
    if name == "highway":
        return build_highway(seed)
    if name == "city":
        return build_city(seed)
    raise ValueError("unknown track: %s (use 'highway' or 'city')" % name)
