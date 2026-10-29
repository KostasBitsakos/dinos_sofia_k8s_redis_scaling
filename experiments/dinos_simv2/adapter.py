"""Adapter between Sim v2 and the DInos agents (10 x 6 window, actions 0/1/2).

Sim v2 observation slots (env.OBS_NAMES) are written into the 6 input slots of the DInos LSTM in this order:
    0 completed_ops   1 cpu_pct   2 p95_ms   3 failed_per_s   4 ready   5 pending
The old DInos environment had (load, capacity, cpu, memory, latency, vms) in these slots, so the meaning of every
slot differs: loading modelCPU_*.pth is a weight initialisation only, not a feature-compatible transfer.

Normalisation uses FIXED constants (no statistics from data, so nothing is learned from held-out episodes):
    completed_ops : x / (mu * max_pods)            -> 0..1  (30000 ops/s at the default parameters)
    cpu_pct       : x / 100                        -> 0..1
    p95_ms        : log10(1 + x) / 4               -> 0..~1.2 (10^4 ms = 10 s maps to 1; timeout 60 s -> 1.2)
    failed_per_s  : log1p(x) / log1p(mu*max_pods)  -> 0..1 (failures are rare and span orders of magnitude)
    ready         : x / max_pods                   -> 0..1
    pending       : x / PENDING_SCALE (=4)         -> ~0..0.5 (at most max_delta * startup_delay_steps in flight)
The window is initialised with the first observation repeated (ObsWindow.reset), not zeros, because a zero
row is a valid 'no traffic' observation in Sim v2.
Rewards are multiplied by REWARD_SCALE for learning only (Q-values of raw returns of -20 per step are hard for a
lr=5e-5 Huber-loss learner); all reported returns are raw Sim v2 rewards.
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SIM_V2 = os.path.normpath(os.path.join(_HERE, "..", "..", "sim_v2"))
if SIM_V2 not in sys.path:
    sys.path.insert(0, SIM_V2)

from env import OBS_NAMES, ObsWindow, SimV2  # noqa: E402
from params import SimParams  # noqa: E402

WINDOW = 10
N_FEATURES = len(OBS_NAMES)
N_ACTIONS = 3
PENDING_SCALE = 4.0
REWARD_SCALE = 0.02


def normalise(obs, P):
    """Raw Sim v2 observation (6,) -> normalised (6,) float64."""
    big = P.mu_ops_per_pod * P.max_pods
    o = np.asarray(obs, dtype=np.float64)[:N_FEATURES]
    return np.array([
        o[0] / big,
        o[1] / 100.0,
        math.log10(1.0 + max(o[2], 0.0)) / 4.0,
        math.log1p(max(o[3], 0.0)) / math.log1p(big),
        o[4] / P.max_pods,
        o[5] / PENDING_SCALE,
    ])


class DinosSimV2:
    """Wraps SimV2: reset(seed, init_pods) -> window (10, 6); step(a) -> window, reward, done, info.
    env_kwargs go to SimV2 (only the K8sHPA comparator uses them: replica_actions / util_obs). The window keeps the
    6 default slots, so an agent never sees the extra cpu_util_pct field (it is only in `raw`)."""

    def __init__(self, workload="W0", params=None, **env_kwargs):
        self.P = params or SimParams()
        if self.P.max_delta != 1:
            raise ValueError("DInos agents have 3 actions; use max_delta=1")
        self.env = SimV2(self.P, workload=workload, **env_kwargs)
        self.win = ObsWindow(WINDOW, N_FEATURES)
        self.raw = None

    def reset(self, seed, init_pods=None):
        self.raw, _ = self.env.reset(seed, init_pods=init_pods)
        return self.win.reset(normalise(self.raw, self.P))

    def step(self, action):
        self.raw, r, term, trunc, info = self.env.step(int(action))
        return self.win.push(normalise(self.raw, self.P)), r, bool(term or trunc), info
