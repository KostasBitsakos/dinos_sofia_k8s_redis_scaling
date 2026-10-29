#!/usr/bin/env python3
"""Train and evaluate DInos-style agents on Sim v2 (provisional parameters), or evaluate the baselines.

  train_eval.py --agent dinos_t|lstm_scratch|derp --pretrained 5 --seed S --train-steps N --workload W0
  train_eval.py --agent baselines --seed S          (threshold tuned on tune seeds, fixed-best, always_min/max, oracle)

Telemetry (PLAN section 7) under $QRL_RUN_DIR: metrics.jsonl, progress.json, checkpoints/, plots/.
  train_eval.py --agent preflight --freeze spec_preflight_v21.json   (validation-only sanity study, PLAN 8.1 gate)
  train_eval.py --agent diagnose --preflight-json PATH --freeze spec_diagnose_v21.json   (no re-tuning, see run_diagnose)
  train_eval.py --agent preflight_rev1 --preflight-json PATH --freeze spec_preflight_rev1.json   (protocol rev 1, see run_preflight_rev1)
  train_eval.py --agent dinos_t|lstm_scratch|derp|explore_baselines --protocol explore_w0 --amendment-id R-20261017-014
                --freeze spec_explore_w0.json   (exploratory W0 after the failed gates, PREREG_explore_w0.md)

Training uses train seeds (0-999), baselines are tuned on tune (validation) seeds 1000-1004, every method is
evaluated on the same FRESH test seeds 20000... (greedy, epsilon = 0); 10000-10999 are burned (seen by tests, gates
and smokes, see README). Every run first verifies the frozen file hashes and git state (`--freeze SPEC`, freeze.py);
`--unfrozen` (or QRL_ALLOW_UNFROZEN=1) is for tests and smokes only and is never in a spec.
Nothing here is a paper result: Sim v2 is provisional.
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapter  # noqa: E402  (also puts sim_v2 on sys.path)
from adapter import REWARD_SCALE, DinosSimV2, SimParams  # noqa: E402

REWARD_MODES = ("cost", "dinos")
COST_ONLY_AGENTS = ("baselines", "preflight", "diagnose", "preflight_rev1", "explore_baselines")

AGENTS = ("dinos_t", "lstm_scratch", "derp", "baselines", "preflight", "diagnose", "preflight_rev1", "explore_baselines")
# Sim v2.1 threshold grid, widened so the W0 validation optimum (tune seeds 1000-1004) is interior: with the
# Sim v2 grid it sat on the edge (up=40, the lowest value), and a grid of 40..90 stays edge-bound. Cooldown 0
# (no cooldown) is the natural floor of that axis, so only its upper edge counts as an edge.
THRESHOLD_GRID_V21 = {"up": (25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 95),
                      "down": (5, 10, 15, 20, 25, 30, 35, 40, 50, 60),
                      "cooldown": (0, 1, 2, 4, 8)}
NATURAL_FLOOR = {"cooldown": 0}
TEST0 = 20000          # fresh evaluation seeds TEST0 + i (never used by any test, gate or smoke)
HELDOUT0 = TEST0       # historical name
TUNE_SEEDS = tuple(range(1000, 1005))
BURNED = range(10000, 11000)   # seen by sim_v2 tests / gates / early smokes: never evaluate on these


def eval_seeds(n):
    from workloads import check_seed
    return [check_seed("test", TEST0 + i) for i in range(n)]


def seed_sets(n_eval=1000):
    """Train / validation / test seed sets as used by this module (for the disjointness test)."""
    from workloads import SEED_RANGES
    return dict(train=set(SEED_RANGES["train"]), validation=set(TUNE_SEEDS), test=set(eval_seeds(n_eval)))


# PLAN 8.1 sanity gate: the tuned threshold must beat the trivial/static controls (and the oracle must not lose).
CONTROLS = ("fixed_best", "always_min", "always_max")


def gate_check(returns):
    """returns: {policy: [per-episode return, same episodes in the same order]} (paired). Never passes silently:
    reports mean, paired difference and wins against each control; `passed` needs thr > every control (mean),
    `oracle_sane` needs oracle >= thr (mean)."""
    thr = np.asarray(returns["threshold"], dtype=float)
    out = dict(controls={}, passed=True)
    for c in CONTROLS:
        d = thr - np.asarray(returns[c], dtype=float)
        out["controls"][c] = dict(mean_diff=float(d.mean()), wins=int((d > 0).sum()), n=int(d.size), beats=bool(d.mean() > 0))
        out["passed"] = out["passed"] and bool(d.mean() > 0)
    out["oracle_sane"] = bool(np.mean(returns["oracle"]) >= thr.mean() - 1e-9) if "oracle" in returns else None
    return out


def threshold_edge_axes(thr, grid=None, floors=None):
    """Axes on which the selected Threshold sits on the edge of the search grid (empty list = interior)."""
    grid = grid or THRESHOLD_GRID_V21
    floors = NATURAL_FLOOR if floors is None else floors
    sel = {"up": thr.up, "down": thr.down, "cooldown": thr.cooldown}
    bad = []
    for ax, vals in grid.items():
        lo, hi = min(vals), max(vals)
        if sel[ax] == hi or (sel[ax] == lo and floors.get(ax) != lo):
            bad.append(ax)
    return bad


def assert_threshold_interior(thr, grid=None):
    bad = threshold_edge_axes(thr, grid)
    if bad:
        raise RuntimeError(f"tuned threshold ({thr.up:g}, {thr.down:g}, {thr.cooldown}) is on the grid edge "
                           f"along {bad}: widen THRESHOLD_GRID_V21 and re-freeze the spec")


def sensitivity_names(arg):
    import sensitivity
    return sensitivity.names() if arg == "all" else [x for x in arg.split(",") if x]


# ---------------------------------------------------------------- reward mode / learning scale
def reward_mode_of(args):
    """The reward mode of this run. Old specs do not pass --reward-mode and keep 'cost'; new specs pass it explicitly
    (make_spec refuses to generate a spec without it)."""
    m = getattr(args, "reward_mode", None) or "cost"
    if m not in REWARD_MODES:
        raise SystemExit(f"--reward-mode must be one of {REWARD_MODES}")
    return m


def learn_scale_of(args):
    """Positive scale applied identically to the reward of EVERY learned agent in the replay buffer (learning only;
    all reported returns are raw rewards). Default = adapter.REWARD_SCALE = 0.02 (unchanged)."""
    v = getattr(args, "learn_scale", None)
    v = REWARD_SCALE if v is None else float(v)
    if not v > 0.0 or not np.isfinite(v):
        raise SystemExit("--learn-scale must be a positive finite number")
    return v


def dinos_variant_of(args):
    """The reward_dinos.json variant of this run ('default' = V2 when absent). Validated against the frozen file."""
    v = getattr(args, "dinos_variant", None) or "default"
    import env as _env
    k, _ = _env.load_reward_dinos()
    names = sorted(k.get("variants", {}))
    if v not in names:
        raise SystemExit(f"--dinos-variant must be one of {names}")
    return v


def base_params(args):
    """SimParams() of the run (default config) with the run's reward mode and dinos variant; sensitivity configs are
    applied on it."""
    return SimParams().with_(reward_mode=reward_mode_of(args), dinos_variant=dinos_variant_of(args))


def sens_params(args, cname):
    import sensitivity
    return sensitivity.params(cname, base_params(args))


def run_config(args):
    """What the run used for reward and learning, written to run_config.json and (numeric) to metrics.jsonl step 0."""
    import env as _env
    mode = reward_mode_of(args)
    out = dict(reward_mode=mode, learn_scale=learn_scale_of(args), agent=args.agent, seed=args.seed,
               dinos_variant=dinos_variant_of(args))
    if mode == "dinos":
        k, sha = _env.load_reward_dinos()
        out.update(reward_dinos_sha256=sha, reward_dinos=k)
    return out


# ---------------------------------------------------------------- TD / Q diagnostics (learning-scale instrumentation)
def td_diagnostics(agent, agent_name, batch_size, rng):
    """Distribution of the TD error, Q / target magnitudes and the Huber saturation on a batch drawn from the replay
    memory with a PRIVATE generator (the global random / numpy / torch streams, hence training, are untouched).
    Same target as the agent's replay(): LSTM = Double DQN, DERP = max over the target network. Saturation = share of
    samples with |TD| > 1 (the SmoothL1 / Huber transition point, beta = 1). Values are in LEARNING units (scaled)."""
    import torch
    n = len(agent.memory)
    if n < max(batch_size, 2):
        return {}
    idx = rng.choice(n, size=batch_size, replace=False)
    mem = [agent.memory[int(i)] for i in idx]
    was = agent.online_model.training, agent.target_model.training
    agent.online_model.eval()
    agent.target_model.eval()
    try:
        with torch.no_grad():
            st = torch.tensor(np.array([m[0] for m in mem]), dtype=torch.float32)
            nx = torch.tensor(np.array([m[3] for m in mem]), dtype=torch.float32)
            if agent_name != "derp":
                shape = (batch_size, agent.sequence_length, agent.state_size)
                st, nx = st.reshape(shape), nx.reshape(shape)
            a = torch.tensor([m[1] for m in mem], dtype=torch.int64).unsqueeze(1)
            rw = torch.tensor([m[2] for m in mem], dtype=torch.float32)
            dn = torch.tensor([float(m[4]) for m in mem], dtype=torch.float32)
            qall = agent.online_model(st)
            q = qall.gather(1, a).squeeze(1)
            if agent_name != "derp":
                y = agent.compute_targets(rw, nx, dn)
            else:
                y = rw + agent.gamma * (1.0 - dn) * agent.target_model(nx).max(dim=1).values
            td = (y - q).abs().numpy()
    finally:
        agent.online_model.train(was[0])
        agent.target_model.train(was[1])
    qa, ya = np.abs(q.numpy()), np.abs(y.numpy())
    return dict(diag_td_abs_mean=float(td.mean()), diag_td_abs_p50=float(np.percentile(td, 50)),
                diag_td_abs_p90=float(np.percentile(td, 90)), diag_td_abs_p99=float(np.percentile(td, 99)),
                diag_td_abs_max=float(td.max()), diag_q_abs_mean=float(qa.mean()), diag_q_abs_max=float(qa.max()),
                diag_target_abs_mean=float(ya.mean()), diag_target_abs_max=float(ya.max()),
                diag_reward_abs_mean=float(np.abs(rw.numpy()).mean()), diag_huber_saturation=float((td > 1.0).mean()))


# ---------------------------------------------------------------- telemetry
class Telemetry:
    def __init__(self, run_dir):
        self.dir = run_dir
        if run_dir:
            os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    def log(self, **m):
        if self.dir:
            clean = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}
            with open(os.path.join(self.dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(clean) + "\n")

    def progress(self, step, total):
        if self.dir:
            with open(os.path.join(self.dir, "progress.json"), "w") as f:
                json.dump({"step": int(step), "total": int(total)}, f)


def check_modules(args):
    """With --freeze: every imported source file under code/ must be in the frozen manifest (before training)."""
    if getattr(args, "freeze", None):
        import freeze
        with open(args.freeze) as f:
            man = json.load(f)["frozen_sha256"]
        freeze.check_loaded_modules(man, os.path.dirname(os.path.abspath(args.freeze)))


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


# ---------------------------------------------------------------- rollout
def rollout(sim, act, seed, init_pods=None, record=False):
    """One episode with `act(window)->action`. Returns (summary dict, traces or None). Raw (unscaled) rewards."""
    win = sim.reset(seed, init_pods=init_pods)
    s = dict(ret=0.0, cost=0.0, sla=0.0, churn=0.0, bad=0.0, arrived=0.0, pod_steps=0.0, steps=0)
    p95s = []
    tr = {k: [] for k in ("load", "cpu", "p95_ms", "failed_per_s", "pods")} if record else None
    done = False
    while not done:
        win, r, done, info = sim.step(act(win))
        s["ret"] += r
        s["cost"] += info["reward_cost"]
        s["sla"] += info["reward_sla"]
        s["churn"] += info["reward_churn"]
        s["bad"] += info["bad_requests"]
        s["arrived"] += info["arrived"]
        s["pod_steps"] += info["ready"] + info["pending"]
        s["steps"] += 1
        if info["p95_ms"] == info["p95_ms"]:
            p95s.append(info["p95_ms"])
        if record:
            tr["load"].append(info["lambda"])
            tr["cpu"].append(sim.raw[1])
            tr["p95_ms"].append(info["p95_ms"])
            tr["failed_per_s"].append(sim.raw[3])
            tr["pods"].append(info["ready"] + info["pending"])
    out = dict(ret=s["ret"], cost=s["cost"], sla=s["sla"], churn=s["churn"],
               mean_pods=s["pod_steps"] / s["steps"], sla_violation_rate=s["bad"] / max(s["arrived"], 1e-12),
               mean_p95_ms=float(np.mean(p95s)), p95_of_p95_ms=float(np.percentile(p95s, 95)))
    return out, tr


def log_eval(tel, step, episode, seed, res, **extra):
    tel.log(step=step, eval_episode=episode, eval_seed=seed, eval_return=res["ret"], eval_cost=res["cost"],
            eval_sla=res["sla"], eval_churn=res["churn"], eval_mean_pods=res["mean_pods"],
            eval_sla_violation_rate=res["sla_violation_rate"], eval_mean_p95_ms=res["mean_p95_ms"],
            eval_p95_of_p95_ms=res["p95_of_p95_ms"], **extra)


# ---------------------------------------------------------------- plots
def plot_eval(traces, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("load", "Load (ops/s)", "blue", "Load Over Time"), ("cpu", "CPU (%)", "orange", "CPU Usage Over Time"),
              ("p95_ms", "p95 latency (ms)", "purple", "Latency Over Time"),
              ("failed_per_s", "Failures (/s)", "green", "Failures Over Time")]
    plt.figure(figsize=(12, 10))
    for i, (key, ylabel, color, ttl) in enumerate(panels, 1):
        ax = plt.subplot(2, 2, i)
        for tr in traces:
            ax.plot(tr[key], color=color, alpha=0.5)
        ax.set_title(ttl)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel, color=color)
        ax2 = ax.twinx()
        for tr in traces:
            ax2.scatter(range(len(tr["pods"])), tr["pods"], color="red", marker="o", s=3, alpha=0.3)
        ax2.set_ylabel("Number of pods", color="red")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def plot_training(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not rows:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([r["step"] for r in rows], [r["train_mean_reward"] for r in rows])
    ax[0].set_xlabel("env steps")
    ax[0].set_ylabel("mean raw reward / step (train)")
    ax[1].plot([r["step"] for r in rows], [r.get("loss", np.nan) for r in rows])
    ax[1].set_xlabel("env steps")
    ax[1].set_ylabel("mean loss")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ---------------------------------------------------------------- agents
def build_agent(name, pretrained):
    import agents
    if name == "dinos_t":
        if pretrained is None:
            raise SystemExit("--agent dinos_t needs --pretrained {1,2,5,10,20}")
        return agents.make_lstm(pretrained)
    if name == "lstm_scratch":
        return agents.make_lstm(None)
    if name == "derp":
        return agents.make_derp()
    raise ValueError(name)


def train_seeds(seed, n):
    from workloads import check_seed
    perm = np.random.default_rng(seed).permutation(1000)[:n]
    return [check_seed("train", int(s)) for s in perm]


def run_agent(args, tel):
    require_rev1_authorization(args, tel)
    import torch
    import agents
    seed_everything(args.seed)
    agent = build_agent(args.agent, args.pretrained)
    check_modules(args)
    scale = learn_scale_of(args)
    sim = DinosSimV2(args.workload, base_params(args))
    diag_rng = np.random.default_rng([int(args.seed), 0xD1A6])   # private stream for the diagnostics
    state_of = lambda w: agents.agent_state(args.agent, w)  # noqa: E731
    ep_len = sim.P.episode_steps
    n_ep = -(-args.train_steps // ep_len)
    seeds = train_seeds(args.seed, max(n_ep, 1))
    explore = args.protocol == "explore_w0"
    ev_seeds = explore_eval_seeds(args) if explore else eval_seeds(args.eval_episodes)
    total = args.train_steps + len(sensitivity_names(args.sensitivity)) * len(ev_seeds)
    step, ep = 0, 0
    rows, win_rew, win_loss = [], [], []
    t0 = time.time()
    while step < args.train_steps:
        window = sim.reset(seeds[ep])
        done = False
        ep_ret = 0.0
        while not done and step < args.train_steps:
            action = agent.act(state_of(window))
            nxt, r, done, _ = sim.step(action)
            agent.remember(state_of(window), action, r * scale, state_of(nxt), done)
            if len(agent.memory) > args.batch_size and step % args.replay_every == 0:
                n0 = len(agent.loss_history)
                agent.replay(args.batch_size)
                if len(agent.loss_history) > n0:
                    win_loss.append(agent.loss_history[-1])
            window = nxt
            ep_ret += r
            win_rew.append(r)
            step += 1
            if step % args.log_every == 0:
                row = dict(step=step, train_mean_reward=float(np.mean(win_rew)), epsilon=float(agent.epsilon),
                           loss=float(np.mean(win_loss)) if win_loss else float("nan"), train_episode=ep,
                           wall_s=time.time() - t0, reward_mode_dinos=int(reward_mode_of(args) == "dinos"),
                           learn_scale=scale)
                if args.diag_every and step % args.diag_every == 0:
                    row.update(td_diagnostics(agent, args.agent, args.batch_size, diag_rng))
                rows.append(row)
                tel.log(**row)
                tel.progress(step, total)
                win_rew, win_loss = [], []
            if args.checkpoint_every and step % args.checkpoint_every == 0 and tel.dir:
                agent.save_model(os.path.join(tel.dir, "checkpoints", "model.pth"))
        if done:
            tel.log(step=step, train_episode_return=ep_ret, train_episode=ep)
        ep += 1
    train_wall = time.time() - t0
    if tel.dir:
        agent.save_model(os.path.join(tel.dir, "checkpoints", "model.pth"))
    # ----- held-out evaluation, greedy; training config is the default, evaluation also under the sensitivity configs
    import sensitivity
    cfgs = sensitivity_names(args.sensitivity)
    write_config_index(tel, cfgs)
    eps_saved, agent.epsilon = agent.epsilon, 0.0   # checkpoint rule: the LAST agent state (no selection)
    traces, rets = [], []
    n_eval = len(cfgs) * len(ev_seeds)
    per_cfg = {}
    k = 0
    for ci, cname in enumerate(cfgs):
        esim = DinosSimV2(args.workload, sens_params(args, cname))
        for i, s in enumerate(ev_seeds):
            res, tr = rollout(esim, lambda w: agent.act(state_of(w)), s, record=(cname == "default"))
            log_eval(tel, step, i, s, res, train_wall_s=train_wall, config_index=ci)
            per_cfg.setdefault(cname, []).append(dict(seed=s, **res))
            if cname == "default":
                traces.append(tr)
                rets.append(res["ret"])
            k += 1
            tel.progress(args.train_steps + k, args.train_steps + n_eval)
    agent.epsilon = eps_saved
    if explore:
        write_explore_eval(tel, args, {args.agent: per_cfg}, cfgs, ev_seeds, extra=dict(
            train_steps=args.train_steps, train_seed=args.seed, train_wall_s=train_wall, checkpoint="last",
            reward_mode=reward_mode_of(args), dinos_variant=dinos_variant_of(args), learn_scale=scale))
    if tel.dir:
        plot_eval(traces, os.path.join(tel.dir, "plots", "evaluation_metrics.png"),
                  f"{args.agent} seed {args.seed} (Sim v2.1 provisional, {args.workload}, held-out, default config)")
        plot_training(rows, os.path.join(tel.dir, "plots", "training_curve.png"), f"{args.agent} seed {args.seed}")
    return dict(mean_return=float(np.mean(rets)) if rets else float("nan"), train_wall_s=train_wall)


# ---------------------------------------------------------------- baselines
def run_baselines(args, tel):
    from policies import AlwaysMax, AlwaysMin, OracleReplay, select_static, tune_threshold
    from oracle import oracle_bound
    import sensitivity
    seed_everything(args.seed)
    check_modules(args)
    P = base_params(args)
    sim = DinosSimV2(args.workload, P)
    t0 = time.time()
    # Selection on validation (tune) seeds 1000-1004 and the DEFAULT config only; frozen for every config.
    thr, table = tune_threshold(P, args.workload, TUNE_SEEDS, grid=THRESHOLD_GRID_V21)
    fixed, _ = select_static(P, args.workload, TUNE_SEEDS, "fixed")
    tuning_s = time.time() - t0
    edge = threshold_edge_axes(thr)
    tel.log(step=0, tuning_wall_s=tuning_s, threshold_up=thr.up, threshold_down=thr.down,
            threshold_cooldown=thr.cooldown, fixed_best_c=fixed.c, threshold_grid_points=len(table),
            threshold_on_grid_edge=len(edge))
    assert_threshold_interior(thr)
    pols = {"threshold": thr, "fixed_best": fixed, "always_min": AlwaysMin(), "always_max": AlwaysMax()}
    vres = policy_returns(P, args.workload, pols, TUNE_SEEDS)
    vgate = gate_check({k: [r["ret"] for r in v] for k, v in vres.items()})
    tel.log(step=0, gate_val_passed=int(vgate["passed"]), gate_val_oracle_sane=int(bool(vgate["oracle_sane"])),
            **{f"gate_val_diff_{c}": d["mean_diff"] for c, d in vgate["controls"].items()})
    if not vgate["passed"] or not vgate["oracle_sane"]:   # PLAN 8.1: report loudly, never pass silently
        if tel.dir:
            with open(os.path.join(tel.dir, "gate_failed.json"), "w") as f:
                json.dump(vgate, f, indent=1)
        raise RuntimeError(f"PLAN 8.1 sanity gate FAILED on validation seeds: {json.dumps(vgate)}")
    cfgs = sensitivity_names(args.sensitivity)
    write_config_index(tel, cfgs)
    names = ["threshold", "fixed_best", "always_min", "always_max", "oracle"]
    total = len(names) * len(cfgs) * args.eval_episodes
    done_n = 0
    summary, test_returns = {}, {}
    for ci, cname in enumerate(cfgs):
        Pc = sens_params(args, cname)
        simc = DinosSimV2(args.workload, Pc)
        for bi, name in enumerate(names):
            traces, rets = [], []
            for i in range(args.eval_episodes):
                s = eval_seeds(args.eval_episodes)[i]
                if name == "oracle":
                    simc.reset(s)
                    bound, acts = oracle_bound(Pc, simc.env.trace)  # re-solved under this config's parameters
                    pol = OracleReplay(acts)
                    pol.reset()
                    res, tr = rollout(simc, lambda w, p=pol: p.act(None), s, record=(cname == "default"))
                    res["oracle_bound"] = bound
                else:
                    pol = pols[name]
                    pol.reset()
                    res, tr = rollout(simc, lambda w, p=pol: p.act(simc.raw), s,
                                      init_pods=getattr(pol, "init_pods", None), record=(cname == "default"))
                log_eval(tel, bi, i, s, res, baseline_index=bi, config_index=ci,
                         **({"oracle_bound": res["oracle_bound"]} if name == "oracle" else {}))
                if cname == "default":
                    traces.append(tr)
                rets.append(res["ret"])
                if cname == "default":
                    test_returns.setdefault(name, []).append(res["ret"])
                done_n += 1
                tel.progress(done_n, total)
            summary.setdefault(cname, {})[name] = float(np.mean(rets))
            if tel.dir and cname == "default":
                plot_eval(traces, os.path.join(tel.dir, "plots", f"evaluation_{name}.png"),
                          f"{name} (Sim v2.1 provisional, {args.workload}, held-out, default config)")
    if tel.dir:
        with open(os.path.join(tel.dir, "baselines_summary.json"), "w") as f:
            json.dump(dict(mean_return=summary, baseline_index=names, configs=cfgs,
                           threshold=[thr.up, thr.down, thr.cooldown], fixed_best_c=fixed.c,
                           threshold_grid=THRESHOLD_GRID_V21, gate_validation=vgate,
                           gate_test_default_config=gate_check(test_returns)), f, indent=1)
    return summary


def policy_returns(P, workload, pols, seeds, config_name="default", timings=None):
    """Paired per-episode results of fixed policies + the oracle on `seeds` (default config unless given).
    Returns {name: [rollout dict (+ oracle_bound)]}; wall time per policy goes into `timings` (seconds)."""
    import sensitivity
    from policies import OracleReplay
    from oracle import oracle_bound
    Pc = sensitivity.params(config_name, P)
    sim = DinosSimV2(workload, Pc)
    out = {}
    for name, pol in list(pols.items()) + [("oracle", None)]:
        t0 = time.time()
        rows = []
        for s in seeds:
            if pol is None:
                sim.reset(s)
                bound, acts = oracle_bound(Pc, sim.env.trace)
                p = OracleReplay(acts)
                p.reset()
                res, _ = rollout(sim, lambda w, p=p: p.act(None), s)
                res["oracle_bound"] = bound
            else:
                pol.reset()
                res, _ = rollout(sim, lambda w, p=pol: p.act(sim.raw), s, init_pods=getattr(pol, "init_pods", None))
            rows.append(res)
        out[name] = rows
        if timings is not None:
            timings[name] = time.time() - t0
    return out


def run_preflight(args, tel):
    """Validation-only sanity study (default config, tune seeds 1000-1004): full widened-grid threshold tuning,
    best fixed(c), always-min/max, oracle; per-policy paired returns + reward decomposition + component runtimes.
    The threshold and fixed(c) are SELECTED on these same episodes, so the table is in-sample by design: it checks
    the PLAN 8.1 gate and measures cost, it is not an evaluation. No test seed is touched."""
    from policies import AlwaysMax, AlwaysMin, select_static, tune_threshold
    seed_everything(args.seed)
    check_modules(args)
    P = base_params(args)
    timings, steps_per_ep = {}, P.episode_steps
    t0 = time.time()
    thr, table = tune_threshold(P, args.workload, TUNE_SEEDS, grid=THRESHOLD_GRID_V21)
    timings["threshold_tuning"] = time.time() - t0
    tuning_steps = len(table) * len(TUNE_SEEDS) * steps_per_ep
    t0 = time.time()
    fixed, fscores = select_static(P, args.workload, TUNE_SEEDS, "fixed")
    timings["fixed_selection"] = time.time() - t0
    fixed_steps = len(fscores) * len(TUNE_SEEDS) * steps_per_ep
    pols = {"threshold": thr, "fixed_best": fixed, "always_min": AlwaysMin(), "always_max": AlwaysMax()}
    res = policy_returns(P, args.workload, pols, TUNE_SEEDS, timings=timings)
    returns = {k: [r["ret"] for r in v] for k, v in res.items()}
    gate = gate_check(returns)
    edge = threshold_edge_axes(thr)
    summary = {}
    for bi, (name, rows) in enumerate(res.items()):
        m = {k: float(np.mean([r[k] for r in rows])) for k in ("ret", "cost", "sla", "churn", "mean_pods",
                                                              "sla_violation_rate")}
        m["pod_seconds"] = m["mean_pods"] * steps_per_ep * P.control_step_s
        m["per_episode_return"] = returns[name]
        summary[name] = m
        tel.log(step=bi, policy_index=bi, **{f"val_{k}": v for k, v in m.items() if k != "per_episode_return"})
        for i, r in enumerate(rows):
            tel.log(step=bi, eval_episode=i, eval_seed=TUNE_SEEDS[i], eval_return=r["ret"], eval_cost=r["cost"],
                    eval_sla=r["sla"], eval_churn=r["churn"], policy_index=bi)
    steps_s = tuning_steps / timings["threshold_tuning"]
    est = dict(grid_points=len(table), tuning_env_steps=tuning_steps, fixed_env_steps=fixed_steps,
               tuning_steps_per_s=steps_s,
               note="W0 baselines entry repeats this tuning once, plus 5 policies x 5 test episodes x "
                    f"{len(sensitivity_names('all'))} configs (oracle re-solved per config)")
    report = dict(workload=args.workload, seeds=list(TUNE_SEEDS), policy_order=list(res), threshold=[thr.up, thr.down, thr.cooldown],
                  threshold_on_grid_edge=edge, fixed_best_c=fixed.c, summary=summary, gate=gate,
                  timings_s=timings, cost_model=est, versions=__import__("freeze").versions(),
                  caveat="in-sample: threshold and fixed(c) were selected on these same validation episodes")
    lines = [f"{'policy':<11}{'return':>12}{'cost':>12}{'sla':>12}{'churn':>10}{'pod-sec':>11}{'wall_s':>8}"]
    for name, m in summary.items():
        lines.append(f"{name:<11}{m['ret']:>12.1f}{m['cost']:>12.1f}{m['sla']:>12.1f}{m['churn']:>10.1f}"
                     f"{m['pod_seconds']:>11.0f}{timings[name]:>8.1f}")
    lines.append(f"threshold={thr.up:g}/{thr.down:g}/{thr.cooldown} fixed_c={fixed.c} edge_axes={edge} "
                 f"gate_passed={gate['passed']} oracle_sane={gate['oracle_sane']}")
    lines.append(f"tuning {timings['threshold_tuning']:.0f}s for {tuning_steps/1e6:.2f}M steps "
                 f"({steps_s:.0f} steps/s); fixed selection {timings['fixed_selection']:.0f}s")
    table_txt = "\n".join(lines)
    if tel.dir:
        with open(os.path.join(tel.dir, "preflight.json"), "w") as f:
            json.dump(report, f, indent=1)
        with open(os.path.join(tel.dir, "preflight_table.txt"), "w") as f:
            f.write(table_txt + "\n")
    tel.log(step=len(res), gate_passed=int(gate["passed"]), oracle_sane=int(bool(gate["oracle_sane"])),
            threshold_on_grid_edge=len(edge), tuning_wall_s=timings["threshold_tuning"])
    tel.progress(1, 1)
    print(table_txt)
    out = dict(gate_passed=gate["passed"], oracle_sane=gate["oracle_sane"], threshold_edge_axes=edge,
               timings_s=timings)
    if not gate["passed"] or not gate["oracle_sane"] or edge:
        raise RuntimeError(f"preflight sanity FAILED (reported in preflight.json): {json.dumps(gate)}; edge={edge}")
    return out


# ---------------------------------------------------------------- diagnose (R-20261017-001, after the failed preflight)
STARTUP_STEPS = 150
CYCLE_STEPS = 160            # one sine cycle = 1000 / (2 pi) = 159.2 steps
SPIKE_PAD = 10
TRACE_FIELDS = ("t", "arrivals", "lambda_ops", "obs_cpu_used", "cpu_after", "action", "ready", "pending", "rebalancing",
                "capacity", "backlog", "served", "failed", "slo_violations", "p95_ms", "reward", "reward_cost",
                "reward_sla", "reward_churn")


def trace_episode(sim, pol, seed, oracle_actions=None):
    """Per-step record of one episode (raw rewards). `pol.act(sim.raw)` or the oracle action list."""
    sim.reset(seed, init_pods=getattr(pol, "init_pods", None))
    if pol is not None:
        pol.reset()
    rows, done, t = [], False, 0
    while not done:
        cpu_used = float(sim.raw[1])
        a = oracle_actions[t] if oracle_actions is not None else pol.act(sim.raw)
        _, r, done, info = sim.step(a)
        rows.append(dict(t=t, arrivals=info["arrived"], lambda_ops=info["lambda"], obs_cpu_used=cpu_used,
                         cpu_after=float(sim.raw[1]), action=int(a), ready=info["ready"], pending=info["pending"],
                         rebalancing=int(bool(info["rebalancing"])), capacity=info["capacity"],
                         backlog=info["outstanding"], served=info["served"], failed=info["failed"],
                         slo_violations=info["slo_violations"], p95_ms=info["p95_ms"], reward=r,
                         reward_cost=info["reward_cost"], reward_sla=info["reward_sla"],
                         reward_churn=info["reward_churn"]))
        t += 1
    return rows


def spike_start(workload, seed, steps):
    """Start index of the W0 spike of `seed` (trace minus the same trace without the spike). None if no spike."""
    from workloads import W0
    if workload != "W0":
        return None
    d = W0().trace(seed, steps) - W0(spike_amp=0.0).trace(seed, steps)
    nz = np.nonzero(np.abs(d) > 1e-9)[0]
    return int(nz[0]) - 1 if nz.size else None


def diag_windows(p0, steps):
    """Startup (first 150 steps), one ordinary sine cycle (first 160-step block after startup clear of the spike
    window), spike window (p0 - 10 .. p0 + 110)."""
    spike = (max(p0 - SPIKE_PAD, 0), min(p0 + 100 + SPIKE_PAD, steps))
    start = STARTUP_STEPS
    while start + CYCLE_STEPS <= steps and not (start + CYCLE_STEPS <= spike[0] or start >= spike[1]):
        start += 10
    return {"startup": (0, STARTUP_STEPS), "sine_cycle": (start, start + CYCLE_STEPS), "spike": spike}


def accounting(rows):
    """Reward components and rebalancing fraction over the startup window vs the rest."""
    out = {}
    for name, sel in (("startup", rows[:STARTUP_STEPS]), ("rest", rows[STARTUP_STEPS:]), ("all", rows)):
        n = max(len(sel), 1)
        out[name] = dict(steps=len(sel), ret=sum(r["reward"] for r in sel), cost=sum(r["reward_cost"] for r in sel),
                         sla=sum(r["reward_sla"] for r in sel), churn=sum(r["reward_churn"] for r in sel),
                         rebalancing_fraction=sum(r["rebalancing"] for r in sel) / n,
                         mean_pods=sum(r["ready"] + r["pending"] for r in sel) / n)
    return out


def plot_diag(traces, windows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(threshold="tab:red", fixed_best="tab:blue", oracle="tab:green")
    rows = [("arrivals", "arrivals/step (solid) vs capacity ops/s x15 (dashed)"), ("pods", "ready (+pending dotted)"),
            ("obs_cpu_used", "observed CPU % seen by policy"), ("reward", "reward per step")]
    fig, axes = plt.subplots(len(rows), len(windows), figsize=(16, 11), squeeze=False)
    for j, (wname, (a, b)) in enumerate(windows.items()):
        for i, (key, ylabel) in enumerate(rows):
            ax = axes[i][j]
            for pname, tr in traces.items():
                x = [r["t"] for r in tr[a:b]]
                c = colors[pname]
                if key == "arrivals":
                    ax.plot(x, [r["arrivals"] for r in tr[a:b]], color=c, alpha=0.4, lw=0.8) if pname == "threshold" else None
                    ax.plot(x, [r["capacity"] * 15 for r in tr[a:b]], color=c, ls="--", lw=1, label=pname)
                elif key == "pods":
                    ax.plot(x, [r["ready"] for r in tr[a:b]], color=c, lw=1, label=pname)
                    ax.plot(x, [r["ready"] + r["pending"] for r in tr[a:b]], color=c, ls=":", lw=0.8)
                else:
                    ax.plot(x, [r[key] for r in tr[a:b]], color=c, lw=0.8, label=pname)
            if i == 0:
                ax.set_title(f"{wname} [{a}, {b})")
                ax.legend(fontsize=7)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=7)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close()


def diag_table_text(report):
    pols = report["policy_order"]
    L = ["Sensitivity (validation seeds, mean return over seeds; threshold/fixed_best NOT reselected; oracle re-solved per config)"]
    L.append(f"{'config':<26}" + "".join(f"{p:>14}" for p in pols) + f"{'thr-fixed':>12}{'thr wins':>9}")
    for cname, row in report["sensitivity"].items():
        L.append(f"{cname:<26}" + "".join(f"{row['mean_return'][p]:>14.0f}" for p in pols)
                 + f"{row['threshold_minus_fixed']:>12.0f}{row['threshold_wins_vs_fixed']:>7}/{report['n_seeds']}")
    L.append("")
    L.append(f"Accounting, default config, mean over seeds (startup = first {STARTUP_STEPS} steps)")
    L.append(f"{'policy':<11}{'window':<9}{'return':>11}{'cost':>11}{'sla':>11}{'churn':>9}{'rebal.frac':>11}{'pods':>7}")
    for p in report["accounting_mean"]:
        for w, m in report["accounting_mean"][p].items():
            L.append(f"{p:<11}{w:<9}{m['ret']:>11.0f}{m['cost']:>11.0f}{m['sla']:>11.0f}{m['churn']:>9.1f}"
                     f"{m['rebalancing_fraction']:>11.3f}{m['mean_pods']:>7.1f}")
    return "\n".join(L)


def run_diagnose(args, tel):
    """No re-tuning: the selected threshold and fixed(c) are LOADED from a finished preflight.json (sha256 recorded).
    Validation seeds only (1000-1004). (1) per-step traces + plots, (2) startup vs rest accounting,
    (3) the selected baselines under every frozen sensitivity config, full table."""
    import csv
    import freeze
    import sensitivity
    from policies import AlwaysMax, AlwaysMin, Fixed, Threshold
    from oracle import oracle_bound
    if not args.preflight_json:
        raise SystemExit("--agent diagnose needs --preflight-json PATH")
    seed_everything(args.seed)
    check_modules(args)
    pre_sha = freeze.sha256_file(args.preflight_json)
    with open(args.preflight_json) as f:
        pre = json.load(f)
    seeds = tuple(pre["seeds"])
    if seeds != tuple(TUNE_SEEDS) or pre["workload"] != args.workload:
        raise RuntimeError(f"preflight seeds/workload {seeds}/{pre['workload']} differ from validation set")
    up, down, cd = pre["threshold"]
    thr, fixed = Threshold(up, down, int(cd)), Fixed(pre["fixed_best_c"])
    P = base_params(args)
    steps = P.episode_steps
    t0 = time.time()
    cfgs = sensitivity_names("all")
    write_config_index(tel, cfgs)
    total = len(seeds) * 3 + len(cfgs) * 5 * len(seeds)
    done_n = 0
    traces, accounting_ps, windows_ps, csv_files = {}, {}, {}, []
    sim = DinosSimV2(args.workload, P)
    for s in seeds:
        sim.reset(s)
        bound, acts = oracle_bound(P, sim.env.trace)
        per = {"threshold": trace_episode(sim, thr, s), "fixed_best": trace_episode(sim, fixed, s),
               "oracle": trace_episode(sim, None, s, oracle_actions=list(acts))}
        for pname, rows in per.items():
            fn = os.path.join(tel.dir or ".", "traces", f"trace_seed{s}_{pname}.csv") if tel.dir else None
            if fn:
                os.makedirs(os.path.dirname(fn), exist_ok=True)
                with open(fn, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
                    w.writeheader()
                    w.writerows(rows)
                csv_files.append(os.path.relpath(fn, tel.dir))
            accounting_ps.setdefault(pname, {})[str(s)] = accounting(rows)
            tel.log(step=done_n, trace_seed=s, policy_index=list(per).index(pname),
                    trace_return=accounting_ps[pname][str(s)]["all"]["ret"],
                    startup_return=accounting_ps[pname][str(s)]["startup"]["ret"],
                    rebalancing_fraction=accounting_ps[pname][str(s)]["all"]["rebalancing_fraction"])
            done_n += 1
            tel.progress(done_n, total)
        p0 = spike_start(args.workload, s, steps)
        win = diag_windows(p0 if p0 is not None else steps // 2, steps)
        windows_ps[str(s)] = win
        traces[s] = per
        if tel.dir:
            plot_diag(per, win, os.path.join(tel.dir, "plots", f"diagnose_seed{s}.png"),
                      f"seed {s}, W0 default config, validation (threshold {up:g}/{down:g}/{cd}, fixed {fixed.c})")
    acc_mean = {p: {w: {k: float(np.mean([accounting_ps[p][str(s)][w][k] for s in seeds]))
                        for k in ("ret", "cost", "sla", "churn", "rebalancing_fraction", "mean_pods")}
                    for w in ("startup", "rest", "all")} for p in accounting_ps}
    # ----- (3) sensitivity table, selected baselines, no reselection
    names = ["threshold", "fixed_best", "always_min", "always_max", "oracle"]
    pols = {"threshold": thr, "fixed_best": fixed, "always_min": AlwaysMin(), "always_max": AlwaysMax()}
    sens = {}
    for ci, cname in enumerate(cfgs):
        Pc = sens_params(args, cname)
        simc = DinosSimV2(args.workload, Pc)
        per_ret = {n: [] for n in names}
        comp = {n: {"cost": [], "sla": [], "churn": []} for n in names}
        for s in seeds:
            for n in names:
                if n == "oracle":
                    simc.reset(s)
                    _, acts = oracle_bound(Pc, simc.env.trace)
                    pol = __import__("policies").OracleReplay(acts)
                    pol.reset()
                    res, _ = rollout(simc, lambda w, p=pol: p.act(None), s)
                else:
                    pol = pols[n]
                    pol.reset()
                    res, _ = rollout(simc, lambda w, p=pol: p.act(simc.raw), s, init_pods=getattr(pol, "init_pods", None))
                per_ret[n].append(res["ret"])
                for k in comp[n]:
                    comp[n][k].append(res[k])
                done_n += 1
            tel.progress(done_n, total)
        d = np.asarray(per_ret["threshold"]) - np.asarray(per_ret["fixed_best"])
        sens[cname] = dict(mean_return={n: float(np.mean(v)) for n, v in per_ret.items()},
                           per_episode_return=per_ret,
                           mean_components={n: {k: float(np.mean(v)) for k, v in c.items()} for n, c in comp.items()},
                           threshold_minus_fixed=float(d.mean()), threshold_wins_vs_fixed=int((d > 0).sum()))
        tel.log(step=ci, config_index=ci, **{f"val_return_{n}": v for n, v in sens[cname]["mean_return"].items()},
                threshold_minus_fixed=sens[cname]["threshold_minus_fixed"])
    # consistency: traced episodes (own loop) must reproduce the table's default-config returns
    for pn in ("threshold", "fixed_best", "oracle"):
        got = [accounting_ps[pn][str(s)]["all"]["ret"] for s in seeds]
        if not np.allclose(got, sens["default"]["per_episode_return"][pn], rtol=1e-9):
            raise RuntimeError(f"trace/table mismatch for {pn}: {got} vs {sens['default']['per_episode_return'][pn]}")
    report = dict(workload=args.workload, seeds=list(seeds), n_seeds=len(seeds), policy_order=names,
                  threshold=[up, down, cd], fixed_best_c=fixed.c, preflight_json=os.path.abspath(args.preflight_json),
                  preflight_sha256=pre_sha, no_reselection=True, windows=windows_ps, accounting_per_seed=accounting_ps,
                  accounting_mean=acc_mean, sensitivity=sens, trace_files=csv_files, wall_s=time.time() - t0,
                  versions=freeze.versions(),
                  caveat="validation seeds only; threshold and fixed(c) were selected on these same seeds (in-sample)")
    table = diag_table_text(report)
    if tel.dir:
        with open(os.path.join(tel.dir, "diagnose.json"), "w") as f:
            json.dump(report, f, indent=1)
        with open(os.path.join(tel.dir, "diagnose_table.txt"), "w") as f:
            f.write(table + "\n")
    tel.progress(total, total)
    print(table)
    return dict(wall_s=report["wall_s"], preflight_sha256=pre_sha)


# ---------------------------------------------------------------- Sim v2.1 protocol rev 1 (R-20261017-012, option 1)
GATE_REV1_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_rev1.json")
REV1_SEEDS = tuple(range(1200, 1210))   # fresh validation block, never evaluated before the rev-1 run (README seed audit)
REV1_BURNED = tuple(range(1100, 1110))  # first choice; exposed by a smoke test (Codex review), never used for the gate
PINNED_PREFLIGHT_RUN = "20261017-140714-R-20261017-004-e0-a1"
REV1_PROTOCOL = "Sim v2.1 protocol rev 1"


def guard_gate_block(seeds, frozen):
    """The gate seed block may only be evaluated by a frozen (spec-launched) rev-1 run, never by tests or smokes."""
    if set(seeds) & set(REV1_SEEDS) and not frozen:
        raise RuntimeError("the rev-1 gate seed block may only be evaluated by a frozen run (--freeze), not by tests/smokes")
    if set(seeds) & set(REV1_BURNED):
        raise RuntimeError("seeds 1100-1109 are burned (exposed by a smoke test) and must not be used")


def verify_original_preflight(path, cfg):
    """Pinned sha256 of the original preflight.json + the original FAIL (gate not passed, oracle sane, fixed c = 22)."""
    import freeze
    got = freeze.sha256_file(path)
    if got != cfg["original_preflight"]["sha256"]:
        raise RuntimeError(f"preflight.json sha256 {got} != pinned {cfg['original_preflight']['sha256']}")
    with open(path) as f:
        pre = json.load(f)
    if pre["gate"]["passed"] is not False or pre["fixed_best_c"] != cfg["original_preflight"]["fixed_best_c"]:
        raise RuntimeError("original preflight is not the recorded FAIL with the pinned fixed_best c")
    if [float(x) for x in pre["threshold"]] != [float(x) for x in cfg["threshold"]]:
        raise RuntimeError("original preflight threshold differs from the pre-registered one")
    return pre, got


def require_rev1_authorization(args, tel=None):
    """run_agent gate: needs a verified PASSING rev-1 artifact (path + sha256 argument) or an approved amendment id."""
    art, sha, amend = getattr(args, "rev1_artifact", None), getattr(args, "rev1_sha256", None), getattr(args, "amendment_id", None)
    rec = None
    if art:
        import freeze
        if not sha or freeze.sha256_file(art) != sha:
            raise SystemExit("refusing to train: --rev1-artifact sha256 does not match --rev1-sha256")
        with open(art) as f:
            rep = json.load(f)
        cfg = load_gate_rev1()
        ok = (rep.get("protocol") == REV1_PROTOCOL and rep.get("gate", {}).get("passed") is True
              and tuple(rep.get("eval_seeds", ())) == REV1_SEEDS and rep.get("no_reselection") is True
              and rep.get("preflight_sha256") == cfg["original_preflight"]["sha256"])
        if not ok:
            raise SystemExit("refusing to train: the rev-1 artifact is not a passing rev-1 gate report")
        rec = dict(kind="rev1_artifact", path=os.path.abspath(art), sha256=sha)
    elif amend:
        if not re.fullmatch(r"[A-Z]+-[A-Za-z0-9-]+", amend):
            raise SystemExit("refusing to train: --amendment-id must look like R-YYYYMMDD-NNN (approved PLAN 8.1 amendment)")
        row = verify_amendment_approved(amend, frozen=bool(getattr(args, "freeze", None)))
        rec = dict(kind="approved_amendment", id=amend, lab_db_record=row)
    else:
        raise SystemExit("refusing to train agents: pass a passing rev-1 artifact (--rev1-artifact PATH --rev1-sha256 HEX) "
                         "or an approved PLAN 8.1 amendment (--amendment-id ID)")
    if getattr(args, "protocol", None) == "explore_w0":
        want = load_explore()["amendment_id"]
        if rec["kind"] != "approved_amendment" or rec["id"] != want:
            raise SystemExit(f"refusing to train: the exploratory W0 protocol needs --amendment-id {want}")
    if tel is not None and tel.dir:
        with open(os.path.join(tel.dir, "rev1_authorization.json"), "w") as f:
            json.dump(rec, f, indent=1)
    return rec


LAB_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def lab_db_path(frozen):
    """The lab database (tools/labdb.db_path). QRL_LAB_DB overrides it for unfrozen tests only, never in a frozen run."""
    if not frozen and os.environ.get("QRL_LAB_DB"):
        return os.environ["QRL_LAB_DB"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("qrl_labdb_ro", os.path.join(LAB_ROOT, "tools", "labdb.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.db_path(LAB_ROOT)


def verify_amendment_approved(amend_id, frozen):
    """Read-only lookup (sqlite mode=ro, nothing written) of the request: it must exist with state 'approved'.
    Syntax alone is not enough (Codex review). Returns the row that is recorded in rev1_authorization.json."""
    import sqlite3
    path = lab_db_path(frozen)
    if not os.path.isfile(path):
        raise SystemExit(f"refusing to train: lab database {path} not found, cannot verify amendment {amend_id}")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT id, state, title, decided_at FROM requests WHERE id=?", (amend_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        raise SystemExit(f"refusing to train: amendment {amend_id} is not a request in {path}")
    if row[1] != "approved":
        raise SystemExit(f"refusing to train: amendment {amend_id} has state '{row[1]}', not 'approved'")
    return dict(id=row[0], state=row[1], title=row[2], decided_at=row[3], db=path)


def load_gate_rev1(path=GATE_REV1_JSON):
    with open(path) as f:
        return json.load(f)


def bootstrap_ci(d, resamples, seed, level=0.95):
    """Percentile bootstrap CI of the mean of the paired differences d (fixed seed -> deterministic)."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(int(resamples), d.size))
    means = d[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))


def gate_rev1_check(returns, cfg=None, startup=None):
    """Rev 1 gate. returns: {policy: per-episode returns (same episodes, same order)} with keys threshold,
    static_ramp_best, fixed_best (comparator, reported only), always_min, always_max, oracle.
    startup: optional {policy: [startup-window return per episode]} for the startup/rest split.
    passed = (threshold - static_ramp_best) has mean > 0 and a bootstrap CI with lower end > 0, AND threshold beats
    always_min and always_max (paired mean > 0), AND oracle mean >= threshold mean. No other path to `passed`."""
    cfg = cfg or load_gate_rev1()
    b = cfg["bootstrap"]
    thr = np.asarray(returns["threshold"], dtype=float)
    out = dict(n=int(thr.size), controls={}, comparators={})

    def one(name, with_ci):
        d = thr - np.asarray(returns[name], dtype=float)
        r = dict(mean_diff=float(d.mean()), wins=int((d > 0).sum()), n=int(d.size), per_episode_diff=d.tolist())
        if with_ci:
            lo, hi = bootstrap_ci(d, b["resamples"], b["seed"], b["level"])
            r.update(ci_lo=lo, ci_hi=hi, ci_excludes_zero=bool(lo > 0 or hi < 0))
        if startup is not None and name in startup and "threshold" in startup:
            ds = np.asarray(startup["threshold"], dtype=float) - np.asarray(startup[name], dtype=float)
            r.update(startup_mean_diff=float(ds.mean()), rest_mean_diff=float((d - ds).mean()),
                     startup_wins=int((ds > 0).sum()), rest_wins=int(((d - ds) > 0).sum()))
        return r
    ramp = one("static_ramp_best", True)
    ramp["beats"] = bool(ramp["mean_diff"] > 0 and ramp["ci_lo"] > 0)
    out["controls"]["static_ramp_best"] = ramp
    for c in ("always_min", "always_max"):
        r = one(c, False)
        r["beats"] = bool(r["mean_diff"] > 0)
        out["controls"][c] = r
    if "fixed_best" in returns:
        out["comparators"]["fixed_best_preprovisioned"] = one("fixed_best", True)
    out["oracle_sane"] = bool(np.mean(returns["oracle"]) >= thr.mean() - cfg["oracle_tolerance_abs"])
    out["passed"] = bool(all(c["beats"] for c in out["controls"].values()) and out["oracle_sane"])
    return out


def run_preflight_rev1(args, tel):
    """Gate rev 1 (pre-registered in PREREG_rev1.md / gate_rev1.json BEFORE this ran). The threshold (40/25/0) and the
    preprovisioned fixed_best c are LOADED from the failed preflight (sha256 recorded, never re-tuned); static_ramp's c
    is selected on tune seeds 1000-1004 (same grid and method as the fixed selection). Evaluation: default config, fresh
    validation seeds 1100-1109, no reselection. Every policy except the preprovisioned fixed_best starts at c0 =
    params.init_pods (the env default, 6)."""
    import freeze
    from policies import AlwaysMax, AlwaysMin, Fixed, Threshold, select_static
    from oracle import oracle_bound
    from workloads import check_seed
    if not args.preflight_json:
        raise SystemExit("--agent preflight_rev1 needs --preflight-json PATH")
    seed_everything(args.seed)
    check_modules(args)
    cfg = load_gate_rev1()
    frozen = bool(args.freeze)
    if args.rev1_eval_seeds:
        if frozen:
            raise RuntimeError("--rev1-eval-seeds is for tests/smokes only")
        seeds = tuple(int(x) for x in args.rev1_eval_seeds.split(","))
    else:
        seeds = tuple(cfg["eval_seeds"])
        if seeds != REV1_SEEDS or tuple(cfg["selection_seeds"]) != tuple(TUNE_SEEDS):
            raise RuntimeError("gate_rev1.json seeds differ from the pre-registered blocks")
    guard_gate_block(seeds, frozen)
    for s in seeds:
        check_seed("tune", s)
    if args.rev1_eval_seeds:   # smoke: fixture preflight, pins are checked by their own unit test
        pre_sha = freeze.sha256_file(args.preflight_json)
        with open(args.preflight_json) as f:
            pre = json.load(f)
    else:
        pre, pre_sha = verify_original_preflight(args.preflight_json, cfg)
    if tuple(pre["seeds"]) != tuple(TUNE_SEEDS) or pre["workload"] != args.workload:
        raise RuntimeError("preflight seeds/workload differ from the validation set")
    up, down, cd = pre["threshold"]
    if [float(up), float(down), int(cd)] != [float(x) for x in cfg["threshold"]]:
        raise RuntimeError("preflight threshold differs from the pre-registered 40/25/0")
    thr, fixed = Threshold(up, down, int(cd)), Fixed(pre["fixed_best_c"])
    P = base_params(args)
    if P.init_pods != cfg["c0"]:
        raise RuntimeError(f"env default init_pods {P.init_pods} != pre-registered c0 {cfg['c0']}")
    t0 = time.time()
    ramp, rscores = select_static(P, args.workload, TUNE_SEEDS, "ramp")
    t_sel = time.time() - t0
    pols = {"threshold": thr, "static_ramp_best": ramp, "fixed_best": fixed, "always_min": AlwaysMin(),
            "always_max": AlwaysMax()}
    names = list(pols) + ["oracle"]
    sim = DinosSimV2(args.workload, P)
    total = len(seeds) * len(names)
    done_n = 0
    acc = {n: [] for n in names}
    for s in seeds:
        sim.reset(s)
        bound, acts = oracle_bound(P, sim.env.trace)
        for n in names:
            if n == "oracle":
                rows = trace_episode(sim, None, s, oracle_actions=list(acts))
            else:
                rows = trace_episode(sim, pols[n], s)
            acc[n].append(accounting(rows))
            done_n += 1
            tel.progress(done_n, total)
    returns = {n: [a["all"]["ret"] for a in acc[n]] for n in names}
    startup = {n: [a["startup"]["ret"] for a in acc[n]] for n in names}
    gate = gate_rev1_check(returns, cfg, startup)
    for bi, n in enumerate(names):
        tel.log(step=bi, policy_index=bi, val_return=float(np.mean(returns[n])),
                val_startup_return=float(np.mean(startup[n])),
                val_rest_return=float(np.mean(returns[n]) - np.mean(startup[n])))
    r = gate["controls"]["static_ramp_best"]
    tel.log(step=len(names), gate_rev1_passed=int(gate["passed"]), oracle_sane=int(gate["oracle_sane"]),
            ramp_mean_diff=r["mean_diff"], ramp_ci_lo=r["ci_lo"], ramp_ci_hi=r["ci_hi"],
            static_ramp_best_c=ramp.c, selection_wall_s=t_sel)
    report = dict(protocol=REV1_PROTOCOL, workload=args.workload, eval_seeds=list(seeds),
                  selection_seeds=list(TUNE_SEEDS), c0=P.init_pods, threshold=[up, down, cd],
                  fixed_best_c=fixed.c, static_ramp_best_c=ramp.c,
                  static_ramp_selection_scores={str(c): v for c, v in rscores},
                  preflight_json=os.path.abspath(args.preflight_json), preflight_sha256=pre_sha,
                  mean_return={n: float(np.mean(returns[n])) for n in names}, per_episode_return=returns,
                  per_episode_startup_return=startup, gate=gate, no_reselection=True,
                  original_preflight_fail_stands="R-20261017-004 FAIL stands and is reported alongside this result",
                  versions=freeze.versions(), wall_s=time.time() - t0)
    lines = [f"{'policy':<18}{'return':>12}{'startup(150)':>14}{'rest':>12}"]
    for n in names:
        lines.append(f"{n:<18}{np.mean(returns[n]):>12.1f}{np.mean(startup[n]):>14.1f}"
                     f"{np.mean(returns[n]) - np.mean(startup[n]):>12.1f}")
    lines.append(f"threshold 40/25/0, static_ramp_best c={ramp.c}, fixed_best c={fixed.c}, c0={P.init_pods}")
    lines.append(f"vs static_ramp_best: mean diff {r['mean_diff']:.1f}, 95% CI [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}], "
                 f"wins {r['wins']}/{r['n']}; startup diff {r['startup_mean_diff']:.1f} (wins {r['startup_wins']}), "
                 f"rest diff {r['rest_mean_diff']:.1f} (wins {r['rest_wins']})")
    for c in ("always_min", "always_max"):
        g = gate["controls"][c]
        lines.append(f"vs {c}: mean diff {g['mean_diff']:.1f}, wins {g['wins']}/{g['n']}")
    f_ = gate["comparators"].get("fixed_best_preprovisioned")
    if f_:
        lines.append(f"vs preprovisioned fixed_best (comparator, not gated): mean diff {f_['mean_diff']:.1f}, "
                     f"CI [{f_['ci_lo']:.1f}, {f_['ci_hi']:.1f}], wins {f_['wins']}/{f_['n']}")
    lines.append(f"oracle_sane={gate['oracle_sane']}  GATE REV 1 {'PASSED' if gate['passed'] else 'FAILED'}")
    table = "\n".join(lines)
    if tel.dir:
        with open(os.path.join(tel.dir, "preflight_rev1.json"), "w") as f:
            json.dump(report, f, indent=1)
        with open(os.path.join(tel.dir, "preflight_rev1_table.txt"), "w") as f:
            f.write(table + "\n")
    tel.progress(total, total)
    print(table)
    if not gate["passed"]:   # reported in full above and in preflight_rev1.json, never passes silently
        raise RuntimeError("gate rev 1 FAILED (see preflight_rev1.json); agents run only after an approved PLAN 8.1 amendment")
    return dict(gate_passed=True, oracle_sane=gate["oracle_sane"], static_ramp_best_c=ramp.c,
                preflight_sha256=pre_sha, wall_s=report["wall_s"])


# ---------------------------------------------------------------- exploratory W0 (PLAN 8.1 amendment R-20261017-014)
EXPLORE_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_w0.json")
EXPLORE_SEEDS = tuple(range(1300, 1320))   # fresh development block, never evaluated before (PREREG_explore_w0.md)
EXPLORE_PROTOCOL = "exploratory W0 (PLAN 8.1 amendment R-20261017-014)"
EXPLORE_POLICIES = ("static_ramp_best", "fixed_best", "threshold", "k8s_hpa", "always_min", "always_max", "oracle")


def load_explore(path=EXPLORE_JSON):
    with open(path) as f:
        return json.load(f)


def guard_explore_block(seeds, frozen):
    """The exploratory block is evaluated only by frozen (spec-launched) runs; smokes may use seeds 1000-1004 only.
    No other protected block (rev-1 gate, burned, held-out, final test) is ever accepted."""
    seeds = tuple(seeds)
    if set(seeds) & set(EXPLORE_SEEDS) and not frozen:
        raise RuntimeError("the exploratory W0 seed block may only be evaluated by a frozen run (--freeze), not by tests/smokes")
    guard_gate_block(seeds, frozen=False)       # rev-1 gate block and burned seeds: never, frozen or not
    for s in seeds:
        if s not in EXPLORE_SEEDS and s not in TUNE_SEEDS:
            raise RuntimeError(f"seed {s} is neither in the exploratory block nor a development seed (1000-1004)")
    return seeds


def explore_eval_seeds(args):
    cfg = load_explore()
    frozen = bool(getattr(args, "freeze", None))
    if getattr(args, "explore_eval_seeds", None):
        if frozen:
            raise RuntimeError("--explore-eval-seeds is for tests/smokes only")
        seeds = tuple(int(x) for x in args.explore_eval_seeds.split(","))
    else:
        seeds = tuple(cfg["eval_seeds"])
        if seeds != EXPLORE_SEEDS:
            raise RuntimeError("explore_w0.json eval seeds differ from the pre-registered block")
    guard_explore_block(seeds, frozen)
    if frozen:   # the frozen run must match the registration exactly
        if args.sensitivity != cfg["sensitivity"] or args.workload != cfg["workload"]:
            raise RuntimeError("--sensitivity/--workload differ from explore_w0.json")
        if args.agent in cfg["training"]["agents"] and (args.train_steps != cfg["training"]["train_steps"]
                                                        or args.seed not in cfg["training"]["train_seeds"]):
            raise RuntimeError("--train-steps/--seed differ from explore_w0.json")
    return list(seeds)


def write_explore_eval(tel, args, results, cfgs, seeds, extra=None):
    """explore_eval.json: per policy, per config, per episode results (input of analyze_explore_w0.py)."""
    if not tel.dir:
        return
    out = dict(protocol=EXPLORE_PROTOCOL, agent=args.agent, seed=args.seed, workload=args.workload, configs=cfgs,
               eval_seeds=list(seeds), frozen=bool(args.freeze), results=results, **(extra or {}))
    with open(os.path.join(tel.dir, "explore_eval.json"), "w") as f:
        json.dump(out, f, indent=1)


def explore_pins(cfg, smoke):
    """Loads the frozen comparators from the pinned artifacts (sha256 verified), never re-selected:
    threshold 40/25/0 and preprovisioned fixed_best c from the original preflight, static_ramp_best c from rev 1."""
    import freeze
    pins = cfg["pins"]
    out = {}
    for key in ("original_preflight", "rev1"):
        path = pins[key]["path"]
        sha = freeze.sha256_file(path)
        if sha != pins[key]["sha256"] and not smoke:
            raise RuntimeError(f"{key} sha256 {sha} != pinned {pins[key]['sha256']}")
        with open(path) as f:
            out[key] = json.load(f)
        out[key + "_sha256"] = sha
    pre, rev = out["original_preflight"], out["rev1"]
    if not smoke:
        if [float(x) for x in pre["threshold"]] != [float(x) for x in pins["original_preflight"]["threshold"]] \
                or pre["fixed_best_c"] != pins["original_preflight"]["fixed_best_c"] or pre["gate"]["passed"] is not False:
            raise RuntimeError("original preflight differs from the pinned FAIL / threshold / fixed_best c")
        if rev.get("protocol") != REV1_PROTOCOL or rev["gate"]["passed"] is not False \
                or rev["static_ramp_best_c"] != pins["rev1"]["static_ramp_best_c"] or rev.get("no_reselection") is not True:
            raise RuntimeError("rev-1 artifact differs from the pinned FAIL / static_ramp_best c")
    return out


def run_explore_baselines(args, tel):
    """Comparators of the exploratory W0 study on the registered block, default + every frozen sensitivity config.
    Nothing is re-selected from evaluation data: threshold / fixed_best / static_ramp_best are loaded from pinned
    artifacts; the K8sHPA target is re-derived on development seeds 1000-1004 (default config) and must equal the
    registered value. Oracle replay is re-solved per config (clairvoyant reference)."""
    import sensitivity
    from policies import AlwaysMax, AlwaysMin, Fixed, K8sHPA, OracleReplay, StaticRamp, Threshold, tune_hpa
    from oracle import oracle_bound
    require_rev1_authorization(args, tel)      # the whole study runs under the approved amendment only
    seed_everything(args.seed)
    check_modules(args)
    cfg = load_explore()
    seeds = explore_eval_seeds(args)
    smoke = bool(args.explore_eval_seeds)
    pins = explore_pins(cfg, smoke)
    up, down, cd = pins["original_preflight"]["threshold"]
    P = base_params(args)
    t0 = time.time()
    hpa, table = tune_hpa(P, args.workload, TUNE_SEEDS, grid=tuple(cfg["hpa"]["target_grid"]))
    t_sel = time.time() - t0
    if hpa.target != cfg["hpa"]["target_pct"] and not smoke:
        raise RuntimeError(f"K8sHPA target re-derived on 1000-1004 is {hpa.target}, registered {cfg['hpa']['target_pct']}")
    if hpa.target in (min(cfg["hpa"]["target_grid"]), max(cfg["hpa"]["target_grid"])) and not smoke:
        raise RuntimeError(f"K8sHPA target {hpa.target} is on the grid edge")
    hpa = K8sHPA(cfg["hpa"]["target_pct"])
    pols = {"static_ramp_best": StaticRamp(pins["rev1"]["static_ramp_best_c"]),
            "fixed_best": Fixed(pins["original_preflight"]["fixed_best_c"]),
            "threshold": Threshold(up, down, int(cd)), "k8s_hpa": hpa, "always_min": AlwaysMin(),
            "always_max": AlwaysMax()}
    names = list(EXPLORE_POLICIES)
    cfgs = sensitivity_names(args.sensitivity)
    write_config_index(tel, cfgs)
    tel.log(step=0, hpa_target_pct=hpa.target, hpa_selection_wall_s=t_sel, threshold_up=float(up),
            threshold_down=float(down), threshold_cooldown=int(cd), fixed_best_c=pols["fixed_best"].c,
            static_ramp_best_c=pols["static_ramp_best"].c)
    total, done_n = len(cfgs) * len(seeds) * len(names), 0
    results = {n: {} for n in names}
    for ci, cname in enumerate(cfgs):
        Pc = sens_params(args, cname)
        sims = {False: DinosSimV2(args.workload, Pc), True: DinosSimV2(args.workload, Pc, **K8sHPA.env_kwargs)}
        for i, s in enumerate(seeds):
            for bi, n in enumerate(names):
                if n == "oracle":
                    sim = sims[False]
                    sim.reset(s)
                    bound, acts = oracle_bound(Pc, sim.env.trace)
                    pol = OracleReplay(acts)
                    pol.reset()
                    res, _ = rollout(sim, lambda w, p=pol: p.act(None), s)
                    res["oracle_bound"] = bound
                else:
                    pol = pols[n]
                    sim = sims[pol is hpa]
                    pol.reset()
                    res, _ = rollout(sim, lambda w, p=pol, sm=sim: p.act(sm.raw), s, init_pods=getattr(pol, "init_pods", None))
                log_eval(tel, bi, i, s, res, baseline_index=bi, config_index=ci)
                results[n].setdefault(cname, []).append(dict(seed=s, **res))
                done_n += 1
                tel.progress(done_n, total)
    write_explore_eval(tel, args, results, cfgs, seeds, extra=dict(
        policy_index=names, hpa_target_pct=hpa.target, hpa_selection_table=[[t, r] for t, r in table],
        threshold=[up, down, cd], fixed_best_c=pols["fixed_best"].c, static_ramp_best_c=pols["static_ramp_best"].c,
        pins={k: v for k, v in pins.items() if k.endswith("_sha256")}, dinos_variant=dinos_variant_of(args)))
    return dict(hpa_target_pct=hpa.target, wall_s=time.time() - t0,
                mean_return_default={n: float(np.mean([r["ret"] for r in results[n][cfgs[0]]])) for n in names})


def write_config_index(tel, cfgs):
    """config_index in metrics.jsonl is numeric; this file maps it to the sensitivity config name."""
    if tel.dir:
        with open(os.path.join(tel.dir, "configs.json"), "w") as f:
            json.dump({str(i): c for i, c in enumerate(cfgs)}, f, indent=1)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, choices=AGENTS)
    p.add_argument("--pretrained", type=int, default=None, help="k of modelCPU_k.pth (dinos_t)")
    seed_env = os.environ.get("QRL_RUN_SEED")
    p.add_argument("--seed", type=int, default=int(seed_env) if seed_env else 0)
    p.add_argument("--config", default=None, help="JSON file; keys override defaults (CLI flags win)")
    p.add_argument("--train-steps", type=int, default=50000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--workload", default="W0")
    p.add_argument("--freeze", default=None, help="approved spec JSON whose frozen_sha256/frozen_git_commit are "
                   "verified before anything runs")
    p.add_argument("--preflight-json", default=None, help="finished preflight.json (diagnose: selection is loaded, never redone)")
    p.add_argument("--rev1-artifact", default=None, help="preflight_rev1.json of a PASSING rev-1 gate (agents only)")
    p.add_argument("--rev1-sha256", default=None, help="expected sha256 of --rev1-artifact")
    p.add_argument("--amendment-id", default=None, help="id of an approved PLAN 8.1 amendment (agents only; recorded)")
    p.add_argument("--rev1-eval-seeds", default=None, help="tests/smokes only (needs --unfrozen): comma list replacing the gate seeds")
    p.add_argument("--protocol", default=None, choices=("explore_w0",), help="exploratory W0 (PREREG_explore_w0.md)")
    p.add_argument("--explore-eval-seeds", default=None, help="tests/smokes only (unfrozen, seeds 1000-1004): comma "
                   "list replacing the exploratory evaluation block")
    p.add_argument("--unfrozen", action="store_true", help="skip freeze verification (tests/smokes only)")
    p.add_argument("--sensitivity", default="default",
                   help="'default', 'all' (every config of sim_v2/sensitivity_v21.json) or a comma list; "
                        "evaluation only, training always uses the default config")
    p.add_argument("--reward-mode", default=None, choices=REWARD_MODES,
                   help="reward of the simulator: 'cost' (Sim v2; also when absent = old specs) or 'dinos' (adapter, "
                        "R-20261017-017). New specs always pass it explicitly (make_spec enforces).")
    p.add_argument("--dinos-variant", default=None,
                   help="reward_dinos.json variant used when --reward-mode dinos (default: 'default' = V2; "
                        "'legacy_p95' = old definition); validated, applied and recorded in run_config.json")
    p.add_argument("--learn-scale", type=float, default=None,
                   help="positive factor on the reward stored for learning (all learned agents identically); "
                        f"default {REWARD_SCALE} (frozen in new specs)")
    p.add_argument("--diag-every", type=int, default=None,
                   help="TD / Q / Huber diagnostics into metrics.jsonl every N steps (default = --log-every, 0 = off)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--replay-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=10000)
    if argv is None:
        argv = sys.argv[1:]
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        with open(pre.config) as f:
            p.set_defaults(**{k.replace("-", "_"): v for k, v in json.load(f).items()})
    a = p.parse_args(argv)
    if a.diag_every is None:
        a.diag_every = a.log_every
    reward_mode_of(a), learn_scale_of(a), dinos_variant_of(a)     # validate before anything else
    if a.agent in COST_ONLY_AGENTS and reward_mode_of(a) != "cost":
        raise SystemExit(f"--agent {a.agent} (gates, pins, threshold/HPA selections and the oracle bound) is defined "
                         "for reward_mode 'cost' only; reselect baselines for 'dinos' with select_baselines_dinos.py")
    import freeze
    tel = Telemetry(os.environ.get("QRL_RUN_DIR"))
    report = None
    if a.freeze:
        report = freeze.verify_freeze(a.freeze)     # raises FreezeError before any training or tuning
    elif not (a.unfrozen or os.environ.get("QRL_ALLOW_UNFROZEN") == "1"):
        raise SystemExit("refusing to run: pass --freeze SPEC (verifies hashes + git state); --unfrozen is for smokes")
    else:
        report = dict(unfrozen=True, versions=freeze.versions())
    report["argv"] = argv
    rcfg = run_config(a)
    report["run_config"] = rcfg
    if tel.dir:
        with open(os.path.join(tel.dir, "run_config.json"), "w") as f:
            json.dump(rcfg, f, indent=1)
        with open(os.path.join(tel.dir, "freeze_report.json"), "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(dict(freeze=report.get("git_head"), versions=report["versions"])))
    t0 = time.time()
    if a.agent == "explore_baselines" and a.protocol != "explore_w0":
        raise SystemExit("--agent explore_baselines needs --protocol explore_w0")
    runner = {"baselines": run_baselines, "preflight": run_preflight, "diagnose": run_diagnose,
              "preflight_rev1": run_preflight_rev1, "explore_baselines": run_explore_baselines}.get(a.agent, run_agent)
    out = runner(a, tel)
    out["total_wall_s"] = time.time() - t0
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    main()
