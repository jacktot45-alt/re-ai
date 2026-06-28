"""
Train the Tesla autonomous driving policy.

Usage:
    python train.py --mode highway --iters 300
    python train.py --mode city    --iters 300 --render
    python train.py --mode highway --resume checkpoints/best_highway.pth
"""

import argparse
import json
import os

import torch

from src.env   import DrivingEnv
from src.model import DrivingPolicy
from src.ppo   import PPOTrainer

try:
    import pygame
except ImportError:
    pygame = None


def get_device():
    try:
        import torch_directml
        dev = torch_directml.device()
        print(f"GPU gevonden: AMD via DirectML ({dev})")
        return dev
    except ImportError:
        pass
    if torch.cuda.is_available():
        print(f"GPU gevonden: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    print("Geen GPU gevonden, training op CPU.")
    return torch.device("cpu")


def train(mode, iters, steps, render, resume, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = get_device()
    env    = DrivingEnv(mode=mode, render_mode=render)
    policy = DrivingPolicy().to(device)

    if resume:
        print(f"Resuming from {resume}")
        policy.load_state_dict(torch.load(resume, map_location="cpu"))

    trainer = PPOTrainer(policy, device=device, lr=3e-4, epochs=10, batch=64)

    print(f"\nTraining Tesla AI  |  mode={mode}  iters={iters}  steps/iter={steps}")
    print(f"Total env steps: {iters * steps:,}\n")

    log  = {"mode": mode, "iter": [], "loss": [], "mean_rew": []}
    best = -1e9

    for it in range(iters):
        last_obs  = trainer.collect(env, n_steps=steps)
        mean_rew  = float(sum(trainer.buf["rew"]) / max(len(trainer.buf["rew"]), 1))
        loss      = trainer.update(last_obs)

        log["iter"].append(it)
        log["loss"].append(loss)
        log["mean_rew"].append(mean_rew)

        print(f"  iter {it+1:4d}/{iters}  loss={loss:.4f}  mean_rew={mean_rew:.3f}")

        if mean_rew > best:
            best = mean_rew
            torch.save(policy.to("cpu").state_dict(), f"{save_dir}/best_{mode}.pth")
            policy.to(device)

        if (it + 1) % 50 == 0:
            path = f"{save_dir}/{mode}_iter{it+1}.pth"
            torch.save(policy.to("cpu").state_dict(), path)
            policy.to(device)
            with open(f"logs/train_{mode}.json", "w") as f:
                json.dump(log, f)
            print(f"    → saved {path}")

        if render and pygame:
            env.render()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    env.close(); return

    torch.save(policy.to("cpu").state_dict(), f"{save_dir}/{mode}_final.pth")
    with open(f"logs/train_{mode}.json", "w") as f:
        json.dump(log, f)

    print(f"\nDone. Best reward: {best:.3f}")
    print(f"Checkpoints → {save_dir}/")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",   choices=["highway", "city"], default="highway")
    p.add_argument("--iters",  type=int, default=300)
    p.add_argument("--steps",  type=int, default=512)
    p.add_argument("--render", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save",   type=str, default="checkpoints")
    a = p.parse_args()
    train(a.mode, a.iters, a.steps, a.render, a.resume, a.save)
