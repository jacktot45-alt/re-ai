"""Small geometry / math helpers (pure standard library)."""
import math

TAU = 2.0 * math.pi


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def lerp(a, b, t):
    return a + (b - a) * t


def wrap_angle(a):
    """Wrap an angle to [-pi, pi]."""
    while a > math.pi:
        a -= TAU
    while a < -math.pi:
        a += TAU
    return a


def rotate(x, y, ang):
    """Rotate point (x, y) by ang radians about the origin."""
    c = math.cos(ang)
    s = math.sin(ang)
    return (c * x - s * y, s * x + c * y)


def catmull_rom(points, samples_per_segment=24):
    """Return a smooth densely-sampled polyline through the given control
    points using a centripetal-ish Catmull-Rom spline. Endpoints are
    duplicated so the curve passes through the first and last points."""
    if len(points) < 2:
        return list(points)
    pts = [points[0]] + list(points) + [points[-1]]
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for j in range(samples_per_segment):
            t = j / float(samples_per_segment)
            t2 = t * t
            t3 = t2 * t
            # Standard Catmull-Rom basis (tension 0.5)
            x = 0.5 * ((2 * p1[0]) +
                       (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) +
                       (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    out.append(points[-1])
    return out
