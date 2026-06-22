"""A tiny multi-layer perceptron implemented in pure Python.

No numpy, no torch -- just lists and loops, with hand-written backpropagation.
It is small on purpose: the camera images are low-resolution and the task
(imitate the expert's steering/throttle) is smooth, so a compact MLP learns it
in well under a minute.

Inputs : flattened, pooled 4-camera pixels (see cameras.pool_flat)
Outputs: [steer_cmd, throttle_cmd], both squashed to [-1, 1] with tanh.
"""
import json
import math
import random


def _tanh(z):
    # math.tanh is fine and stable
    return math.tanh(z)


class MLP:
    def __init__(self, sizes, seed=0):
        """sizes = [n_in, h1, h2, ..., n_out]."""
        self.sizes = list(sizes)
        rng = random.Random(seed)
        self.W = []   # weights[layer][out][in]
        self.b = []   # biases[layer][out]
        self.vW = []  # momentum buffers
        self.vb = []
        for li in range(len(sizes) - 1):
            n_in, n_out = sizes[li], sizes[li + 1]
            limit = math.sqrt(6.0 / (n_in + n_out))   # Xavier/Glorot init
            self.W.append([[rng.uniform(-limit, limit) for _ in range(n_in)]
                           for _ in range(n_out)])
            self.b.append([0.0 for _ in range(n_out)])
            self.vW.append([[0.0] * n_in for _ in range(n_out)])
            self.vb.append([0.0] * n_out)

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        """Return (output, activations) where activations includes the input."""
        acts = [x]
        a = x
        for li in range(len(self.W)):
            W, b = self.W[li], self.b[li]
            z = []
            for o in range(len(W)):
                row = W[o]
                s = b[o]
                for i in range(len(row)):
                    s += row[i] * a[i]
                z.append(_tanh(s))
            a = z
            acts.append(a)
        return a, acts

    def predict(self, x):
        out, _ = self.forward(x)
        return out

    # --------------------------------------------------------------- training
    def train(self, X, Y, epochs=10, batch=32, lr=0.05, momentum=0.9,
              weight_decay=1e-5, lr_decay=1.0, seed=0, log=None):
        rng = random.Random(seed)
        n = len(X)
        idx = list(range(n))
        history = []
        cur_lr = lr
        # Reset momentum so each fit starts clean (no stale velocity carrying
        # over from a previous training phase, which can destabilise updates).
        for li in range(len(self.vW)):
            for o in range(len(self.vW[li])):
                row = self.vW[li][o]
                for i in range(len(row)):
                    row[i] = 0.0
                self.vb[li][o] = 0.0
        for ep in range(epochs):
            rng.shuffle(idx)
            total = 0.0
            for start in range(0, n, batch):
                bi = idx[start:start + batch]
                total += self._train_batch(X, Y, bi, cur_lr, momentum, weight_decay)
            mse = total / n
            history.append(mse)
            if log:
                log(ep, mse)
            cur_lr *= lr_decay
        return history

    def _train_batch(self, X, Y, batch_idx, lr, momentum, wd):
        nL = len(self.W)
        # Gradient accumulators
        gW = [[[0.0] * len(self.W[li][o]) for o in range(len(self.W[li]))]
              for li in range(nL)]
        gb = [[0.0] * len(self.b[li]) for li in range(nL)]
        batch_loss = 0.0

        for s in batch_idx:
            x, y = X[s], Y[s]
            out, acts = self.forward(x)

            # MSE loss gradient at the output (tanh already applied).
            delta = []
            for o in range(len(out)):
                diff = out[o] - y[o]
                batch_loss += diff * diff
                # d/dz of 0.5*tanh? loss=(a-y)^2; da=2(a-y); dz=da*(1-a^2)
                delta.append(2.0 * diff * (1.0 - out[o] * out[o]))

            # Backpropagate layer by layer.
            for li in range(nL - 1, -1, -1):
                a_in = acts[li]
                W = self.W[li]
                gWl = gW[li]
                gbl = gb[li]
                for o in range(len(W)):
                    d = delta[o]
                    gbl[o] += d
                    gWlo = gWl[o]
                    for i in range(len(a_in)):
                        gWlo[i] += d * a_in[i]
                if li > 0:
                    # Propagate delta to the previous layer's activations.
                    new_delta = [0.0] * len(a_in)
                    for o in range(len(W)):
                        d = delta[o]
                        Wo = W[o]
                        for i in range(len(a_in)):
                            new_delta[i] += Wo[i] * d
                    a_prev = acts[li]            # activation of prev layer (=a_in)
                    delta = [new_delta[i] * (1.0 - a_prev[i] * a_prev[i])
                             for i in range(len(a_in))]

        # Apply the averaged gradient with momentum + weight decay.
        m = len(batch_idx)
        for li in range(nL):
            W, b = self.W[li], self.b[li]
            vW, vb = self.vW[li], self.vb[li]
            gWl, gbl = gW[li], gb[li]
            for o in range(len(W)):
                Wo, vWo, gWo = W[o], vW[o], gWl[o]
                for i in range(len(Wo)):
                    grad = gWo[i] / m + wd * Wo[i]
                    vWo[i] = momentum * vWo[i] - lr * grad
                    Wo[i] += vWo[i]
                vb[o] = momentum * vb[o] - lr * (gbl[o] / m)
                b[o] += vb[o]
        return batch_loss

    # ------------------------------------------------------------- save / load
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"sizes": self.sizes, "W": self.W, "b": self.b}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        net = cls(d["sizes"])
        net.W = d["W"]
        net.b = d["b"]
        return net
