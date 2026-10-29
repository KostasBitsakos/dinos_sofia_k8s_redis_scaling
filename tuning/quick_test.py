#!/usr/bin/env python3
"""Cheap sandbox test: train ONE config on ONE training seed for N env steps, then evaluate greedily on the dev seeds
1000-1004 under reward_mode="dinos". Writes a result JSON (config, per-training-episode and per-eval-episode
diagnostics). Seeds outside [0,999] (training) / [1000,1004] (evaluation) are refused. Sandbox dinos-tuning only.

  quick_test.py --config tuning/configs/derp_tuned.json --seed 0 --steps 20000 --out result.json
Config JSON: {"agent": "derp"|"dinos_t"|"lstm_scratch", "pretrained": 5, "input_map": "none"|"pretrain"|"adapter",
              "batch_size": 16, "replay_every": 1, "learn_scale": 0.02, "agent_options": {tuned_agents.DEFAULT_CFG keys}}
"load_def": "offered" (default) | "served" (cycle 2, L from served requests; result JSON records it).
"reward_variant": "v2" (default) | "d_sec" (tuning/reward_variants.py: D = mean latency in s / 1 s). "pretrained_path": a
state_dict saved by tuning/pretrain_source.py (dinos_t; trunk trained on V2 features, so the input adapter starts as identity;
"reset_head" in agent_options still applies). Eval seeds: dev 1000-1004 or confirmation 1100-1109. Result JSON has
eval_mean_return and eval_sum_return (SUM over the eval episodes) in the units of the chosen reward variant; per-episode
records have sum_3L/sum_neg5D/sum_neg10N/sum_neg05U/sum_neg05M, failed_requests, mean_latency_s (mean over steps of the
per-step mean latency) and pods.
"features": "base" (default) | "demand" (+ offered rate, its 4-step change, offered load per ready pod; see
features.py; dinos_t then needs input_map "adapter" = Linear(9 -> 6)).
"adapter": raw sim v2 features into a trainable Linear(6,6) (initialised to the pretrain map) in front of the trunk.
learn_scale is the learner-only linear reward scale (e.g. 1e-3); other learner options are agent_options (n_step,
explore_repeat_max, huber_beta, grad_clip ...). Eval episodes also report pods at t=0/100/500/end.
Telemetry (PLAN 7): metrics.jsonl and progress.json under $QRL_RUN_DIR when set.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments", "dinos_simv2"))
sys.path.insert(0, os.path.dirname(HERE))
import adapter  # noqa: E402
from adapter import DinosSimV2, SimParams  # noqa: E402
from tuning import diagnostics, features as featmod, reward_variants, transfer_map, tuned_agents  # noqa: E402

TRAIN_RANGE = range(0, 1000)
DEV_SEEDS = tuple(range(1000, 1005))
CONFIRM_SEEDS = tuple(range(1100, 1110))
EVAL_ALLOWED = DEV_SEEDS + CONFIRM_SEEDS
REWARD_MODE = "dinos"
AGENTS = ("derp", "dinos_t", "lstm_scratch")
CFG_KEYS = {"agent", "pretrained", "input_map", "batch_size", "replay_every", "learn_scale", "agent_options", "features",
            "reward_variant", "pretrained_path", "load_def"}


def check_seeds(train_seed, eval_seeds):
    if int(train_seed) not in TRAIN_RANGE:
        raise SystemExit(f"refusing: training seed {train_seed} outside 0-999")
    bad = [s for s in eval_seeds if int(s) not in EVAL_ALLOWED]
    if bad:
        raise SystemExit(f"refusing: evaluation seeds {bad} outside the dev block 1000-1004 / confirmation block 1100-1109")


def build(cfg):
    unknown = set(cfg) - CFG_KEYS
    if unknown:
        raise SystemExit(f"unknown config keys {sorted(unknown)}")
    name = cfg["agent"]
    if cfg.get("reward_variant", "v2") not in reward_variants.VARIANTS:
        raise SystemExit(f"reward_variant must be one of {reward_variants.VARIANTS}")
    if cfg.get("load_def", "offered") not in reward_variants.LOAD_DEFS:
        raise SystemExit(f"load_def must be one of {reward_variants.LOAD_DEFS}")
    if name not in AGENTS:
        raise SystemExit(f"agent must be one of {AGENTS}")
    opts = cfg.get("agent_options", {})
    n = featmod.n_features(cfg.get("features", "base"))
    if name == "derp":
        return tuned_agents.make_tuned_derp(opts, n)
    if name == "dinos_t":
        if cfg.get("pretrained_path"):
            # V2-compatible source-task trunk (tuning/pretrain_source.py): identity input adapter, head reset optional
            if cfg.get("input_map") not in ("adapter", "none", None):
                raise SystemExit("pretrained_path needs input_map 'adapter' or 'none'")
            return tuned_agents.make_tuned_lstm(cfg["pretrained_path"], opts, cfg.get("input_map") == "adapter", n)
        if cfg.get("pretrained") is None:
            raise SystemExit("dinos_t needs 'pretrained' or 'pretrained_path'")
        if n != featmod.N_BASE and cfg.get("input_map") != "adapter":
            raise SystemExit("features 'demand' with dinos_t needs input_map 'adapter'")
        return tuned_agents.make_tuned_lstm(cfg["pretrained"], opts, cfg.get("input_map") == "adapter", n)
    return tuned_agents.make_tuned_lstm(None, opts, cfg.get("input_map") == "adapter", n)


def run(cfg, seed, steps, eval_seeds=DEV_SEEDS, out=None, tel_dir=None, diag_every=1000, agent_out=None,
        workload="W0", train_sim_seeds=None):
    import torch
    import train_eval as te
    if train_sim_seeds is None:   # pretrain_source.py passes its own source-task seeds and no eval
        check_seeds(seed, eval_seeds)
    te.seed_everything(seed)
    agent = build(cfg)
    if agent_out is not None:
        agent_out.append(agent)
    name, scale = cfg["agent"], float(cfg.get("learn_scale", adapter.REWARD_SCALE))
    bs, every = int(cfg.get("batch_size", 16)), int(cfg.get("replay_every", 1))
    imap = cfg.get("input_map", "none")
    if imap not in transfer_map.MODES:
        raise SystemExit(f"input_map must be one of {transfer_map.MODES}")
    if imap != "none" and name == "derp":
        raise SystemExit("input_map applies to the LSTM agents only")

    def state_of(w):
        w = transfer_map.map_window(np.asarray(w), imap)
        return w[-1] if name == "derp" else w

    tel = te.Telemetry(tel_dir)
    P = SimParams().with_(reward_mode=REWARD_MODE)
    variant = cfg.get("reward_variant", "v2")
    load_def = cfg.get("load_def", "offered")
    sim = reward_variants.wrap(featmod.make_sim(cfg.get("features", "base"), workload, P), variant, load_def)
    n_ep = -(-steps // P.episode_steps)
    seeds = te.train_seeds(seed, max(n_ep, 1)) if train_sim_seeds is None else \
        [int(train_sim_seeds[i % len(train_sim_seeds)]) for i in range(max(n_ep, 1))]
    rng = np.random.default_rng([int(seed), 0xD1A6])
    t_wall, t_cpu = time.time(), time.process_time()
    step, ep, train_eps = 0, 0, []
    while step < steps:
        st = diagnostics.EpisodeStats()
        loss0 = len(agent.loss_history)
        rwd = []
        window, done = sim.reset(seeds[ep]), False
        agent.reset_exploration()
        agent.ctl.reset()
        n_dec = 0

        def on_step(a, r, info):
            st.add(a, r, info)
            rwd.append(r)
        while not done and step < steps:
            tuned_agents.set_epsilon(agent, step)
            window, done, used = tuned_agents.macro_step(agent, sim, window, state_of, scale, steps - step, on_step,
                                                         True, n_dec, bs, every)
            n_dec += 1
            for _ in range(used):
                step += 1
                if step % 1000 == 0:
                    tel.log(step=step, epsilon=float(agent.epsilon), train_mean_return_ep=st.ret / max(st.steps, 1))
                    tel.progress(step, steps)
        rec = st.summary()
        losses = agent.loss_history[loss0:]
        rec.update(reward_min=float(min(rwd)), reward_max=float(max(rwd)), reward_mean=float(np.mean(rwd)),
                   reward_p01=float(np.percentile(rwd, 1)), reward_p99=float(np.percentile(rwd, 99)),
                   episode=ep, sim_seed=int(seeds[ep]), complete=bool(done), end_step=step,
                   epsilon_end=float(agent.epsilon), td_loss=float(np.mean(losses)) if losses else None,
                   **tuned_agents.q_stats(agent, bs, rng))
        train_eps.append(rec)
        tel.log(step=step, **{k: v for k, v in rec.items() if isinstance(v, (int, float)) and v is not None})
        ep += 1
    train_cpu, train_wall = time.process_time() - t_cpu, time.time() - t_wall
    saved, agent.epsilon = agent.epsilon, 0.0
    evals = []
    for s in eval_seeds:
        st = diagnostics.EpisodeStats()
        window, done = sim.reset(int(s)), False
        agent.ctl.reset()
        while not done:
            window, done, _ = tuned_agents.macro_step(agent, sim, window, state_of, 1.0, 10 ** 9, st.add, False)
        evals.append(dict(seed=int(s), **st.summary()))
    agent.epsilon = saved
    res = dict(config=cfg, train_seed=int(seed), train_steps=int(steps), reward_mode=REWARD_MODE, reward_variant=variant, load_def=load_def,
               eval_seeds=[int(s) for s in eval_seeds], eval=evals,
               eval_mean_return=float(np.mean([e["ret"] for e in evals])),
               eval_sum_return=float(np.sum([e["ret"] for e in evals])),   # SUM over the eval episodes (n = len(eval_seeds))
               eval_n_episodes=len(evals),
               eval_mean_pods=float(np.mean([e["mean_pods"] for e in evals])),
               eval_mean_pod_changes=float(np.mean([e["pod_changes"] for e in evals])),
               train_episodes=train_eps, train_wall_s=train_wall, train_cpu_s=train_cpu,
               cpu_s_per_1k_steps=train_cpu / steps * 1000.0, grad_updates=int(agent.grad_updates))
    tel.log(step=steps, eval_mean_return=res["eval_mean_return"], eval_mean_pods=res["eval_mean_pods"])
    tel.progress(steps, steps)
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=int(os.environ.get("QRL_RUN_SEED", 0)))
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--eval-seeds", default=",".join(map(str, DEV_SEEDS)))
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    with open(a.config) as f:
        cfg = json.load(f)
    seeds = [int(x) for x in a.eval_seeds.split(",") if x]
    res = run(cfg, a.seed, a.steps, seeds, a.out, os.environ.get("QRL_RUN_DIR"))
    print(json.dumps({k: res[k] for k in ("eval_mean_return", "eval_sum_return", "eval_mean_pods", "eval_mean_pod_changes",
                                          "train_wall_s", "train_cpu_s", "cpu_s_per_1k_steps")}))
    return res


if __name__ == "__main__":
    main()
