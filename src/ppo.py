"""
PPO (Proximal Policy Optimization) trainer.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam


class PPOTrainer:
    def __init__(self, policy, device=None,
                 lr=3e-4, clip=0.2, ent_coef=0.01, val_coef=0.5,
                 max_grad=0.5, batch=64, epochs=10,
                 gamma=0.99, lam=0.95):
        self.policy    = policy
        self.device    = device or torch.device("cpu")
        self.opt       = Adam(policy.parameters(), lr=lr)
        self.clip      = clip
        self.ent_coef  = ent_coef
        self.val_coef  = val_coef
        self.max_grad  = max_grad
        self.batch     = batch
        self.epochs    = epochs
        self.gamma     = gamma
        self.lam       = lam
        self._clear()

    # ── buffer ───────────────────────────────

    def _clear(self):
        self.buf = dict(obs=[], act=[], rew=[], done=[], val=[], lp=[])

    def _push(self, obs, act, rew, done, val, lp):
        self.buf["obs"].append(obs)
        self.buf["act"].append(act)
        self.buf["rew"].append(rew)
        self.buf["done"].append(done)
        self.buf["val"].append(val)
        self.buf["lp"].append(lp)

    # ── rollout ──────────────────────────────

    def collect(self, env, n_steps=512):
        """Gather n_steps of experience; returns last obs."""
        self._clear()
        obs  = env.reset()
        done = False

        for _ in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                mean, log_std, val = self.policy(obs_t)
                dist = torch.distributions.Normal(mean, log_std.exp())
                act  = dist.sample()[0].clamp(-1, 1)
                lp   = dist.log_prob(act).sum().item()

            next_obs, rew, done, _ = env.step(act.cpu().numpy())
            self._push(obs, act.numpy(), rew, done, val[0].item(), lp)

            obs = next_obs if not done else env.reset()

        return obs

    # ── update ───────────────────────────────

    def update(self, last_obs):
        """Run PPO update; returns mean loss."""
        obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, last_val = self.policy(obs_t)

        adv, ret = self._gae(last_val[0].item())

        obs_T  = torch.FloatTensor(np.array(self.buf["obs"])).to(self.device)
        act_T  = torch.FloatTensor(np.array(self.buf["act"])).to(self.device)
        lp_T   = torch.FloatTensor(np.array(self.buf["lp"])).to(self.device)
        adv_T  = torch.FloatTensor(adv).to(self.device)
        ret_T  = torch.FloatTensor(ret).to(self.device)
        adv_T  = (adv_T - adv_T.mean()) / (adv_T.std() + 1e-8)

        n, total_loss, n_up = len(self.buf["obs"]), 0.0, 0
        for _ in range(self.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch):
                b    = idx[start : start + self.batch]
                lp, ent, val = self.policy.evaluate(obs_T[b], act_T[b])

                ratio = (lp - lp_T[b]).exp()
                a     = adv_T[b]
                pol_loss = -torch.min(ratio * a,
                                      ratio.clamp(1-self.clip, 1+self.clip) * a).mean()
                val_loss = F.mse_loss(val, ret_T[b])
                loss     = pol_loss + self.val_coef * val_loss - self.ent_coef * ent.mean()

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad)
                self.opt.step()
                total_loss += loss.item(); n_up += 1

        return total_loss / max(n_up, 1)

    def _gae(self, last_val):
        rews  = np.array(self.buf["rew"],  dtype=np.float32)
        dones = np.array(self.buf["done"], dtype=np.float32)
        vals  = np.array(self.buf["val"]  + [last_val], dtype=np.float32)
        n     = len(rews)
        adv   = np.zeros(n, dtype=np.float32)
        g     = 0.0
        for t in reversed(range(n)):
            nt  = 1.0 - dones[t]
            d   = rews[t] + self.gamma * vals[t+1] * nt - vals[t]
            g   = d + self.gamma * self.lam * nt * g
            adv[t] = g
        return adv, adv + vals[:-1]
