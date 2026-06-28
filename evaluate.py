"""
Watch the trained Tesla drive and plot training curves.

Usage:
    python evaluate.py checkpoints/best_highway.pth --mode highway
    python evaluate.py checkpoints/best_city.pth    --mode city
    python evaluate.py --plot logs/train_highway.json
"""

import argparse
import sys

import pygame
import torch
import numpy as np

from src.env   import DrivingEnv
from src.model import DrivingPolicy


def run(ckpt, mode, episodes, fps):
    env    = DrivingEnv(mode=mode, render_mode=True)
    policy = DrivingPolicy()
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
    policy.eval()

    clock = pygame.time.Clock()
    print(f"Running {episodes} episode(s)  [ESC or close window to quit, R to reset]\n")

    for ep in range(episodes):
        obs    = env.reset()
        done   = False
        total  = 0.0
        steps  = 0

        while not done:
            action, _ = policy.act(obs, deterministic=True)
            obs, rew, done, _ = env.step(action)
            total += rew
            steps += 1

            env.render(action=action)
            clock.tick(fps)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    env.close(); return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        env.close(); return
                    if ev.key == pygame.K_r:
                        done = True

        print(f"  Episode {ep+1}: reward={total:.1f}  steps={steps}")

    env.close()


def plot(log_file):
    import json, matplotlib.pyplot as plt
    with open(log_file) as f:
        log = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(log["iter"], log["mean_rew"], color="tab:green")
    ax1.set_title("Mean Reward per Iteration")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)

    ax2.plot(log["iter"], log["loss"], color="tab:red")
    ax2.set_title("PPO Loss")
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Tesla AI Training — {log.get('mode','').upper()}")
    plt.tight_layout()
    out = "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", help="Path to .pth checkpoint")
    p.add_argument("--mode",     choices=["highway","city"], default="highway")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--fps",      type=int, default=30)
    p.add_argument("--plot",     type=str, help="Plot a training log JSON")
    a = p.parse_args()

    if a.plot:
        plot(a.plot)
    elif a.checkpoint:
        run(a.checkpoint, a.mode, a.episodes, a.fps)
    else:
        p.print_help()
