#!/usr/bin/env python3
"""Replay-coverage diagnostic (seed 0 only, sandbox dinos-tuning): build the agent of a quick_test config, optionally
train it for --train-steps, then roll out 2000 env steps twice from the same sim seed, once greedy (epsilon 0) and once
exploratory (epsilon = --epsilon, with the config's explore_repeat_max if any), and write a histogram of visited pod
counts (ready+pending, 1..30) for each. Learning is off during the rollouts.

  diagnose.py --config tuning/configs/w3/derp_w3.json --train-steps 0 --epsilon 0.5 --out diag.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments", "dinos_simv2"))
sys.path.insert(0, os.path.dirname(HERE))
from adapter import DinosSimV2, SimParams  # noqa: E402
from tuning import quick_test, transfer_map, tuned_agents  # noqa: E402

MAX_PODS = 30


def rollout(agent, state_of, name, epsilon, steps, sim_seed):
    P = SimParams().with_(reward_mode=quick_test.REWARD_MODE)
    sim = DinosSimV2("W0", P)
    agent.epsilon = epsilon
    agent.reset_exploration()
    window, hist = sim.reset(sim_seed), np.zeros(MAX_PODS + 1, dtype=int)
    for _ in range(steps):
        a = agent.act(state_of(window))
        window, r, done, info = sim.step(a)
        hist[min(int(info["ready"] + info["pending"]), MAX_PODS)] += 1
        if done:
            window = sim.reset(sim_seed)
            agent.reset_exploration()
    return hist


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-steps", type=int, default=0)
    p.add_argument("--rollout-steps", type=int, default=2000)
    p.add_argument("--epsilon", type=float, default=0.5)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.seed != 0:
        raise SystemExit("refusing: diagnose.py is seed 0 only")
    import train_eval as te
    cfg = json.load(open(a.config))
    state, got = {}, []
    if a.train_steps:
        res = quick_test.run(cfg, 0, a.train_steps, eval_seeds=(1000,), agent_out=got)
        state["train_eval_return"] = res["eval_mean_return"]
        agent = got[0]
    else:
        te.seed_everything(0)
        agent = quick_test.build(cfg)
    name, imap = cfg["agent"], cfg.get("input_map", "none")

    def state_of(w):
        w = transfer_map.map_window(np.asarray(w), imap)
        return w[-1] if name == "derp" else w
    out = dict(config=cfg, rollout_steps=a.rollout_steps, epsilon=a.epsilon, **state,
               agent="trained" if a.train_steps else "untrained")
    for label, eps in (("greedy", 0.0), ("exploratory", a.epsilon)):
        h = rollout(agent, state_of, name, eps, a.rollout_steps, 1000)
        out[label] = dict(hist_by_pods={str(i): int(h[i]) for i in range(MAX_PODS + 1) if h[i]},
                          max_pods_visited=int(np.nonzero(h)[0].max()), frac_ge10=float(h[10:].sum() / h.sum()))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    main()
