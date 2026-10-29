"""Agent-side feature option for quick_test (config key "features": "base" | "demand"). The env is NOT edited.

"base"   : the 6 adapter features (experiments/dinos_simv2/adapter.py), unchanged.
"demand" : base + 3 features computed from the offered request rate of the step that was JUST executed (info["lambda"],
           the arrival rate at the front door during that control step; only current and past values are used):
  6 offered_norm   lambda_t / (mu * max_pods)                           in [0, ~1]
  7 offered_delta  0.5 + 0.5 * clip((lambda_t - lambda_{t-K}) / (DELTA_FRAC * mu * max_pods), -1, 1)   in [0, 1]
                   K = DELTA_STEPS env steps back (history inside the wrapper; before K steps exist the oldest value)
  8 offered_util   min(lambda_t / (ready_t * mu), UTIL_CAP) / UTIL_CAP  (ready_t = ready pods in that step; >1/UTIL_CAP
                   means offered load above the nominal capacity of the ready pods)
Pending pods (slot 5) and failed requests/s (slot 3) are already in the base features.
At reset no request has arrived yet: the three demand features are 0 (offered_delta 0.5), nothing from the future.
Observability: lambda_t is the request arrival rate at the proxy / load balancer / Redis client side (e.g. a request
counter rate in Prometheus), something a real autoscaler can read; the sim value is noise-free, a measured rate is not.
"""
from collections import deque

import numpy as np

import adapter
from adapter import DinosSimV2
from env import ObsWindow

FEATURES = ("base", "demand")
DELTA_STEPS = 4
DELTA_FRAC = 0.25
UTIL_CAP = 2.0
N_BASE = adapter.N_FEATURES
N_DEMAND = 3


def n_features(features):
    if features not in FEATURES:
        raise ValueError(f"features must be one of {FEATURES}, got {features!r}")
    return N_BASE + (N_DEMAND if features == "demand" else 0)


def demand_row(lam, hist, ready, P):
    """3 demand features from the current arrival rate `lam`, past rates `hist` (oldest first, may be empty) and the
    ready pods of the executed step."""
    big = P.mu_ops_per_pod * P.max_pods
    ref = hist[0] if len(hist) else lam
    d = float(np.clip((lam - ref) / (DELTA_FRAC * big), -1.0, 1.0))
    util = min(lam / max(ready * P.mu_ops_per_pod, 1e-9), UTIL_CAP) / UTIL_CAP
    return np.array([lam / big, 0.5 + 0.5 * d, util])


class DemandSim(DinosSimV2):
    """DinosSimV2 whose window has 6 + 3 columns (see module docstring)."""

    def __init__(self, workload="W0", params=None, **kw):
        super().__init__(workload, params, **kw)
        self.win = ObsWindow(adapter.WINDOW, N_BASE + N_DEMAND)
        self.hist = deque(maxlen=DELTA_STEPS)

    def reset(self, seed, init_pods=None):
        self.raw, _ = self.env.reset(seed, init_pods=init_pods)
        self.hist.clear()
        row = np.concatenate([adapter.normalise(self.raw, self.P), [0.0, 0.5, 0.0]])
        return self.win.reset(row)

    def step(self, action):
        self.raw, r, term, trunc, info = self.env.step(int(action))
        lam = float(info["lambda"])
        d = demand_row(lam, self.hist, info["ready"], self.P)
        self.hist.append(lam)
        row = np.concatenate([adapter.normalise(self.raw, self.P), d])
        return self.win.push(row), r, bool(term or trunc), info


def make_sim(features, workload="W0", params=None):
    n_features(features)
    return DemandSim(workload, params) if features == "demand" else DinosSimV2(workload, params)
