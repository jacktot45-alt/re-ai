"""
Neural network: 4-camera CNN encoder → shared policy + value heads (PPO).
"""

import torch
import torch.nn as nn
import numpy as np


class CamEncoder(nn.Module):
    """Encodes one (3, 128, 128) camera image → feature vector."""

    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),   # → 64×64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → 32×32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # → 16×16
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),# → 8×8
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.net(torch.zeros(1, 3, 128, 128)).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, out_dim), nn.ReLU())

    def forward(self, x):           # x: (B, 3, H, W)
        return self.fc(self.net(x)) # → (B, out_dim)


class DrivingPolicy(nn.Module):
    """
    Input : (B, 4, 3, 128, 128)  — 4 camera images
    Output: action_mean (B, 2), action_log_std (2,), value (B,)
    """

    def __init__(self, n_cams=4, cam_dim=256, hidden=512):
        super().__init__()
        self.n_cams  = n_cams
        self.encoder = CamEncoder(cam_dim)          # shared weights across cameras

        fused = n_cams * cam_dim
        self.shared = nn.Sequential(
            nn.Linear(fused, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.action_mean    = nn.Linear(hidden // 2, 2)
        self.action_log_std = nn.Parameter(torch.full((2,), -0.5))
        self.value_head     = nn.Sequential(
            nn.Linear(fused, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _fuse(self, obs):
        B = obs.shape[0]
        flat = obs.view(B * self.n_cams, *obs.shape[2:])   # (B*4, 3, H, W)
        enc  = self.encoder(flat)                           # (B*4, cam_dim)
        return enc.view(B, -1)                              # (B, 4*cam_dim)

    def forward(self, obs):
        fused = self._fuse(obs)
        h     = self.shared(fused)
        mean  = torch.tanh(self.action_mean(h))
        log_std = self.action_log_std.clamp(-4, 0)
        value   = self.value_head(fused).squeeze(-1)
        return mean, log_std, value

    # ── inference helper ─────────────────────

    @torch.no_grad()
    def act(self, obs_np, deterministic=False):
        """obs_np: (4, 3, H, W) float32.  Returns (action np (2,), value float)."""
        obs = torch.FloatTensor(obs_np).unsqueeze(0)
        mean, log_std, value = self(obs)
        if deterministic:
            action = mean[0]
        else:
            action = torch.distributions.Normal(mean, log_std.exp()).sample()[0].clamp(-1, 1)
        return action.numpy(), value[0].item()

    # ── training helper ──────────────────────

    def evaluate(self, obs, actions):
        """Returns log_prob (B,), entropy (B,), value (B,)."""
        mean, log_std, value = self(obs)
        dist     = torch.distributions.Normal(mean, log_std.exp())
        log_prob = dist.log_prob(actions).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy, value
