"""
Driving simulation environment.
Renders a 2D top-down world and provides 4 camera views around the Tesla car.
"""

import pygame
import numpy as np
import math
import random

# Camera angle offsets relative to car heading (degrees)
# 0 = front, 90 = right, 180 = rear, -90 = left
CAMERA_OFFSETS = [0, -90, 90, 180]
CAMERA_NAMES   = ["front", "left", "right", "rear"]

CAM_W = 128
CAM_H = 128


# ─────────────────────────────────────────────
# Camera capture
# ─────────────────────────────────────────────

def capture_camera(world_surf, car_x, car_y, car_heading, cam_offset):
    """Return (CAM_H, CAM_W, 3) uint8 array for one camera.

    Fast path: crop a small region around the car first, then rotate only
    that patch — instead of rotating the whole (600x4000) world every call.
    The car ends up at the centre of the patch, so no projection math needed.
    """
    R = 130                                        # half-size of patch to crop
    cxi, cyi = int(car_x), int(car_y)

    region = pygame.Surface((2 * R, 2 * R))
    region.fill((30, 30, 30))                      # off-world fill (rarely seen)
    region.blit(world_surf, (R - cxi, R - cyi))    # car now centred at (R, R)

    theta   = -(car_heading + cam_offset)          # pygame rotates CCW
    rotated = pygame.transform.rotate(region, theta)
    rW, rH  = rotated.get_size()                   # car stays at the centre

    # crop: car appears in bottom 20 % of view
    cx = int(rW / 2 - CAM_W / 2)
    cy = int(rH / 2 - CAM_H * 0.8)

    tmp = pygame.Surface((CAM_W, CAM_H))
    tmp.blit(rotated, (0, 0), (cx, cy, CAM_W, CAM_H))
    arr = pygame.surfarray.array3d(tmp)
    return arr.transpose(1, 0, 2)   # (H, W, 3)


# ─────────────────────────────────────────────
# World: Highway
# ─────────────────────────────────────────────

class HighwayWorld:
    W, H      = 600, 4000
    ROAD_X    = 100
    ROAD_W    = 320
    LANE_W    = 80
    NUM_LANES = 4

    def __init__(self):
        self.surf = pygame.Surface((self.W, self.H))
        self.npcs = []
        self._draw_base()
        self._spawn_npcs()

    def _draw_base(self):
        s = self.surf
        s.fill((34, 110, 34))                                          # grass
        pygame.draw.rect(s, (65, 65, 65),
                         (self.ROAD_X, 0, self.ROAD_W, self.H))       # road
        for i in range(1, self.NUM_LANES):
            x = self.ROAD_X + i * self.LANE_W
            for y in range(0, self.H, 80):
                pygame.draw.rect(s, (230, 230, 140), (x - 2, y, 4, 40))
        # edges
        pygame.draw.rect(s, (255,255,255), (self.ROAD_X,              0, 4, self.H))
        pygame.draw.rect(s, (255,255,255), (self.ROAD_X+self.ROAD_W-4, 0, 4, self.H))
        # trees
        for y in range(0, self.H, 160):
            pygame.draw.circle(s, (0, 80, 0), (55, y), 22)
            pygame.draw.circle(s, (0, 80, 0), (self.W - 55, y), 22)

    def _spawn_npcs(self):
        self.npcs = []
        for _ in range(20):
            lane  = random.randint(0, self.NUM_LANES - 1)
            x     = self.ROAD_X + lane * self.LANE_W + self.LANE_W // 2
            y     = random.randint(300, self.H - 300)
            speed = random.uniform(2.0, 4.5)
            color = (random.randint(140, 255), random.randint(40, 120), random.randint(40, 120))
            self.npcs.append(dict(x=x, y=y, speed=speed, w=18, h=36, color=color))

    def lane_center(self, lane):
        return self.ROAD_X + lane * self.LANE_W + self.LANE_W // 2

    def nearest_lane_dist(self, x):
        lane = int((x - self.ROAD_X) / self.LANE_W)
        lane = max(0, min(self.NUM_LANES - 1, lane))
        return abs(x - self.lane_center(lane))

    def is_on_road(self, x):
        return self.ROAD_X < x < self.ROAD_X + self.ROAD_W

    def update(self):
        for npc in self.npcs:
            npc["y"] -= npc["speed"]
            if npc["y"] < 50:
                npc["y"] = self.H - 100

    def check_collision(self, x, y, w=22, h=44):
        r = pygame.Rect(x - w//2, y - h//2, w, h)
        for npc in self.npcs:
            if r.colliderect(pygame.Rect(npc["x"]-npc["w"]//2,
                                         npc["y"]-npc["h"]//2,
                                         npc["w"], npc["h"])):
                return True
        return False

    def nearest_car_ahead(self, x, y, look_ahead=200):
        """Distance in pixels to the closest NPC directly ahead (lower y = further up)."""
        min_dist = float("inf")
        for npc in self.npcs:
            dy = y - npc["y"]           # positive → NPC is above car (ahead)
            dx = abs(x - npc["x"])
            if 0 < dy < look_ahead and dx < self.LANE_W * 0.8:
                min_dist = min(min_dist, dy)
        return min_dist

    def get_surface(self, tx=None, ty=None, ta=None):
        s = self.surf.copy()
        for npc in self.npcs:
            pygame.draw.rect(s, npc["color"],
                             (npc["x"]-npc["w"]//2, npc["y"]-npc["h"]//2,
                              npc["w"], npc["h"]))
            pygame.draw.rect(s, (150, 200, 255),
                             (npc["x"]-npc["w"]//2+2, npc["y"]-npc["h"]//4,
                              npc["w"]-4, npc["h"]//5))
        if tx is not None:
            _draw_tesla(s, tx, ty, ta)
        return s

    def start_pos(self):
        lane = self.NUM_LANES // 2
        return self.lane_center(lane), self.H - 300


# ─────────────────────────────────────────────
# World: City
# ─────────────────────────────────────────────

class CityWorld:
    W, H      = 600, 4000
    ROAD_X    = 180
    ROAD_W    = 200
    LANE_W    = 50
    NUM_LANES = 4

    def __init__(self):
        self.surf = pygame.Surface((self.W, self.H))
        self.npcs = []
        self._draw_base()
        self._spawn_npcs()

    def _draw_base(self):
        s = self.surf
        s.fill((170, 155, 135))                                       # sidewalk
        pygame.draw.rect(s, (65, 65, 65),
                         (self.ROAD_X, 0, self.ROAD_W, self.H))      # main road
        # lane markings
        cx = self.ROAD_X + self.ROAD_W // 2
        for y in range(0, self.H, 60):
            pygame.draw.rect(s, (255, 220, 0), (cx-2, y, 4, 30))    # center line
        for i in range(1, self.NUM_LANES):
            x = self.ROAD_X + i * self.LANE_W
            for y in range(0, self.H, 60):
                pygame.draw.rect(s, (200, 200, 140), (x-1, y, 2, 25))
        # cross streets every 350 px
        for y in range(350, self.H, 350):
            pygame.draw.rect(s, (65, 65, 65), (0, y-45, self.W, 90))
            for x in range(0, self.W, 24):
                pygame.draw.rect(s, (200, 200, 200), (x, y-45, 12, 90))
        # buildings (random each reset)
        rng = random.Random(42)
        for i in range(0, self.H, 350):
            for side_x, side_w in [(5, self.ROAD_X-10), (self.ROAD_X+self.ROAD_W+5, 120)]:
                bw = rng.randint(50, min(side_w, 130))
                bh = rng.randint(80, 230)
                c  = (rng.randint(90,170), rng.randint(80,150), rng.randint(80,140))
                pygame.draw.rect(s, c, (side_x, i+60, bw, bh))
                pygame.draw.rect(s, (20,20,20), (side_x, i+60, bw, bh), 2)
        # road edges
        pygame.draw.rect(s, (255,255,255), (self.ROAD_X,              0, 3, self.H))
        pygame.draw.rect(s, (255,255,255), (self.ROAD_X+self.ROAD_W-3, 0, 3, self.H))

    def _spawn_npcs(self):
        self.npcs = []
        for _ in range(12):
            lane  = random.randint(0, self.NUM_LANES - 1)
            x     = self.ROAD_X + lane * self.LANE_W + self.LANE_W // 2
            y     = random.randint(300, self.H - 300)
            speed = random.uniform(1.5, 3.5)
            color = (random.randint(140, 255), random.randint(40, 120), random.randint(40, 120))
            self.npcs.append(dict(x=x, y=y, speed=speed, w=16, h=32, color=color))

    def lane_center(self, lane):
        return self.ROAD_X + lane * self.LANE_W + self.LANE_W // 2

    def nearest_lane_dist(self, x):
        lane = int((x - self.ROAD_X) / self.LANE_W)
        lane = max(0, min(self.NUM_LANES - 1, lane))
        return abs(x - self.lane_center(lane))

    def is_on_road(self, x):
        return self.ROAD_X < x < self.ROAD_X + self.ROAD_W

    def update(self):
        for npc in self.npcs:
            npc["y"] -= npc["speed"]
            if npc["y"] < 50:
                npc["y"] = self.H - 100

    def check_collision(self, x, y, w=22, h=44):
        r = pygame.Rect(x - w//2, y - h//2, w, h)
        for npc in self.npcs:
            if r.colliderect(pygame.Rect(npc["x"]-npc["w"]//2,
                                         npc["y"]-npc["h"]//2,
                                         npc["w"], npc["h"])):
                return True
        return False

    def nearest_car_ahead(self, x, y, look_ahead=200):
        """Distance in pixels to the closest NPC directly ahead."""
        min_dist = float("inf")
        for npc in self.npcs:
            dy = y - npc["y"]
            dx = abs(x - npc["x"])
            if 0 < dy < look_ahead and dx < self.LANE_W * 0.8:
                min_dist = min(min_dist, dy)
        return min_dist

    def get_surface(self, tx=None, ty=None, ta=None):
        s = self.surf.copy()
        for npc in self.npcs:
            pygame.draw.rect(s, npc["color"],
                             (npc["x"]-npc["w"]//2, npc["y"]-npc["h"]//2,
                              npc["w"], npc["h"]))
        if tx is not None:
            _draw_tesla(s, tx, ty, ta)
        return s

    def start_pos(self):
        lane = self.NUM_LANES // 2
        return self.lane_center(lane), self.H - 300


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _draw_tesla(surf, x, y, angle):
    W, L = 22, 44
    car = pygame.Surface((W, L), pygame.SRCALPHA)
    car.fill((180, 210, 255))
    pygame.draw.rect(car, (150, 220, 255), (2, 4, W-4, L//5))   # windshield
    pygame.draw.rect(car, (255, 255, 180), (2, 2, 5, 4))         # headlight L
    pygame.draw.rect(car, (255, 255, 180), (W-7, 2, 5, 4))       # headlight R
    pygame.draw.rect(car, (255, 60, 60),   (2, L-6, 5, 4))       # tail L
    pygame.draw.rect(car, (255, 60, 60),   (W-7, L-6, 5, 4))     # tail R
    rot = pygame.transform.rotate(car, -angle)
    surf.blit(rot, rot.get_rect(center=(int(x), int(y))))


# ─────────────────────────────────────────────
# Tesla vehicle
# ─────────────────────────────────────────────

class Tesla:
    MAX_SPEED = 8.0

    def __init__(self, x, y):
        self.x     = float(x)
        self.y     = float(y)
        self.angle = 0.0
        self.speed = 2.0

    def step(self, steering, throttle):
        if throttle >= 0:
            self.speed = float(np.clip(self.speed + throttle * 0.4, 0, self.MAX_SPEED))
        else:
            # Braking is stronger than accelerating
            self.speed = float(np.clip(self.speed + throttle * 1.2, 0, self.MAX_SPEED))
        if self.speed > 0.1:
            self.angle += steering * 3.0 * (self.speed / self.MAX_SPEED)
        rad     = math.radians(self.angle)
        self.x += self.speed * math.sin(rad)
        self.y -= self.speed * math.cos(rad)

    def reset(self, x, y):
        self.x, self.y, self.angle, self.speed = float(x), float(y), 0.0, 2.0


# ─────────────────────────────────────────────
# Main environment (gym-style)
# ─────────────────────────────────────────────

class DrivingEnv:
    """
    obs  : np.float32 (4, 3, CAM_H, CAM_W) — four camera images in [0,1]
    action: [steering ∈ [-1,1], throttle ∈ [-1,1]]
    """

    MAX_STEPS = 2000

    def __init__(self, mode="highway", render_mode=False):
        assert mode in ("highway", "city")
        self.mode        = mode
        self.render_mode = render_mode
        self._font       = None
        self.world       = None
        self.tesla       = None
        self.step_n      = 0
        self.ep_reward   = 0.0

        if not pygame.get_init():
            pygame.init()
        if render_mode:
            self.screen = pygame.display.set_mode((960, 620))
            pygame.display.set_caption(f"Tesla AI — {mode.capitalize()}")
            self._font = pygame.font.Font(None, 22)

    # ── public API ───────────────────────────

    def reset(self):
        WorldCls   = HighwayWorld if self.mode == "highway" else CityWorld
        self.world = WorldCls()
        sx, sy     = self.world.start_pos()
        self.tesla = Tesla(sx, sy)
        self.step_n    = 0
        self.ep_reward = 0.0
        self._cached_obs  = None
        self._cached_surf = None
        return self._obs()

    def step(self, action):
        s = float(np.clip(action[0], -1, 1))
        t = float(np.clip(action[1], -1, 1))
        self.tesla.step(s, t)
        self.world.update()
        self.step_n += 1

        reward, done = self._reward()
        self.ep_reward += reward
        # cache so render() reuses without recomputing
        self._cached_obs  = None
        self._cached_surf = None
        obs = self._obs()
        return obs, reward, done, {"step": self.step_n, "ep_reward": self.ep_reward}

    def render(self, action=None):
        if not self.render_mode:
            return
        scr = self.screen
        scr.fill((15, 15, 25))

        # ── bird's-eye view — reuse cached surface ──
        world_s = self._cached_surf
        vw, vh  = 340, 460
        bx = int(np.clip(self.tesla.x - vw//2, 0, self.world.W - vw))
        by = int(np.clip(self.tesla.y - vh//2, 0, self.world.H - vh))
        bird = pygame.Surface((vw, vh))
        bird.blit(world_s, (0, 0), (bx, by, vw, vh))
        bird = pygame.transform.scale(bird, (340, 460))
        scr.blit(bird, (5, 80))

        # ── four camera views — reuse cached obs ──
        obs = self._cached_obs
        for i, name in enumerate(CAMERA_NAMES):
            col, row = i % 2, i // 2
            px, py   = 360 + col * 300, 10 + row * 305
            img = (obs[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            surf = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
            surf = pygame.transform.scale(surf, (280, 280))
            scr.blit(surf, (px, py))
            pygame.draw.rect(scr, (80, 120, 200), (px, py, 280, 280), 2)
            lbl = self._font.render(f"Cam {i+1} · {name.upper()}", True, (180, 210, 255))
            scr.blit(lbl, (px+4, py+4))

        # ── HUD ──────────────────────────────
        texts = [
            (f"Mode : {self.mode.upper()}", (255, 210, 60)),
            (f"Speed: {self.tesla.speed:.1f} m/s", (180, 255, 180)),
            (f"Step : {self.step_n}", (180, 220, 255)),
            (f"Reward: {self.ep_reward:.1f}", (255, 180, 180)),
        ]
        for i, (txt, col) in enumerate(texts):
            scr.blit(self._font.render(txt, True, col), (10, 10 + i * 22))

        if action is not None:
            a_txt = self._font.render(
                f"Steer {action[0]:+.2f}  Throttle {action[1]:+.2f}", True, (200, 200, 255))
            scr.blit(a_txt, (10, 100))

        pygame.display.flip()

    def close(self):
        if self.render_mode:
            pygame.quit()

    # ── internals ────────────────────────────

    def _obs(self):
        if self._cached_obs is not None:
            return self._cached_obs
        ws = self.world.get_surface(self.tesla.x, self.tesla.y, self.tesla.angle)
        self._cached_surf = ws
        imgs = [capture_camera(ws, self.tesla.x, self.tesla.y,
                               self.tesla.angle, off)
                for off in CAMERA_OFFSETS]
        obs = np.stack(imgs, axis=0)          # (4, H, W, 3)
        obs = obs.transpose(0, 3, 1, 2)       # (4, 3, H, W)
        self._cached_obs = obs.astype(np.float32) / 255.0
        return self._cached_obs

    def _reward(self):
        x, y = self.tesla.x, self.tesla.y
        done = False

        if y < 50 or y > self.world.H - 50:
            return -50.0, True

        if self.world.check_collision(x, y):
            return -100.0, True

        if not self.world.is_on_road(x):
            reward = -5.0
        else:
            # Lane-keeping reward
            lane_r = max(0.0, 1.0 - self.world.nearest_lane_dist(x) / (self.world.LANE_W * 0.5))

            # Adaptive speed reward based on distance to car ahead
            dist_ahead = self.world.nearest_car_ahead(x, y, look_ahead=200)
            SAFE_DIST  = 80   # pixels — comfortable following distance
            DANGER_DIST = 40  # pixels — must brake immediately

            if dist_ahead < DANGER_DIST:
                # Too close: penalise speed, reward being slow
                speed_r = -self.tesla.speed / self.tesla.MAX_SPEED * 1.5
            elif dist_ahead < SAFE_DIST:
                # Closing in: scale target speed down with distance
                t = (dist_ahead - DANGER_DIST) / (SAFE_DIST - DANGER_DIST)
                target_speed = t * self.tesla.MAX_SPEED
                speed_r = -max(0.0, self.tesla.speed - target_speed) * 0.5
            else:
                # Clear road: reward going fast
                speed_r = self.tesla.speed / self.tesla.MAX_SPEED * 0.5

            reward = lane_r + speed_r

        if self.step_n >= self.MAX_STEPS:
            done = True

        return reward, done
