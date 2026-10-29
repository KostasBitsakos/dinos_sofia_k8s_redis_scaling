#!/usr/bin/env python3
"""Determinism check (R-20261017-017): same seed twice, the second while N busy-loop processes load the CPU;
compares sha256 of the full trajectory (reward, obs, info floats) for policies on a dev seed (default 1000).
Usage: python3 determinism_check.py [--seed 1000] [--busy 8] [--steps 1000]. Exit 0 iff all hashes agree."""
import argparse
import hashlib
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adapter  # noqa: E402,F401
import numpy as np  # noqa: E402
from env import SimV2  # noqa: E402
from params import SimParams  # noqa: E402
from policies import K8sHPA, Threshold  # noqa: E402


def traj_hash(seed, steps, which, reward_mode):
    pol = Threshold() if which == "threshold" else K8sHPA(40)
    P = SimParams().with_(reward_mode=reward_mode, episode_steps=steps)
    env = SimV2(P, workload="W0", **getattr(pol, "env_kwargs", {}))
    obs, _ = env.reset(seed)
    pol.reset()
    h = hashlib.sha256()
    while True:
        obs, r, term, _, info = env.step(pol.act(obs))
        h.update(np.float64([r]).tobytes())
        h.update(obs.tobytes())
        h.update(np.float64([v for v in info.values() if isinstance(v, (int, float, bool))]).tobytes())
        if "dinos" in info:
            h.update(np.float64(list(info["dinos"].values())).tobytes())
        if term:
            return h.hexdigest()


def _busy(stop):
    x = 0
    while not stop.is_set():
        x = (x * 1103515245 + 12345) % 2**31


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--busy", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    a = ap.parse_args()
    assert 1000 <= a.seed <= 1004
    ok = True
    for rm in ("cost", "dinos"):
        for which in ("threshold", "hpa"):
            h1 = traj_hash(a.seed, a.steps, which, rm)
            stop = mp.Event()
            ps = [mp.Process(target=_busy, args=(stop,)) for _ in range(a.busy)]
            [p.start() for p in ps]
            try:
                h2 = traj_hash(a.seed, a.steps, which, rm)
            finally:
                stop.set()
                [p.join() for p in ps]
            ok &= h1 == h2
            print(f"{rm:6s} {which:10s} idle={h1[:16]} busy={h2[:16]} {'SAME' if h1 == h2 else 'DIFFERENT'}")
    print("DETERMINISM:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
