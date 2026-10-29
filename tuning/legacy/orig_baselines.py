#!/usr/bin/env python3
"""Cycle 9 baselines on the original env family (orig_variants.py), selected on val, reported per trace on val + test5.

  python3 tuning/legacy/orig_baselines.py --orig-variant orig --out tuning/results/c9a/baselines-orig.json [--seed 0]

Policies (greedy, deterministic, nothing is trained; one +-1 move per tick, as the env allows):
  fixed_N        move +-1 per tick from the start (6 VMs) to N, then hold; N in 1..20 on val, the best N on val
                 (= fixed_best) is the only one run on the report splits
  hpa_rule       the original threshold rule P/HPA/simpleagent.py (cpu > 50 or mem > 50 -> +1, else -1), verbatim
  always_down    action 0 every tick
  causal         n* = ceil(load_t / (10 * read_hat)), read_hat = capacity_t / (10 * VMs_t), clipped 1..20; one step
                 toward n* (capacity 0 -> n* = 20; load 0 -> n* = 1). Uses only the current observation.
  causal_be      ADDITION (not in the design): as causal, but n* = 1 when read_hat < 0.3, i.e. when an extra VM costs
                 more (3) than it can serve (10 * read). Same information as causal.
  hpa_demand     ADDITION (council, 2026-10-20): demand-based HPA with target utilisation u of capacity: +1 if
                 load_t > u * capacity_t, -1 if load_t < (u - 0.2) * capacity_t, else hold; u in {0.6, ..., 1.0}
                 selected on val (hpa_demand_u on val, the selected u only on the report splits)
  causal_trend   ADDITION (council): the causal rule on the forecast load_hat = load_t + k (load_t - load_{t-k}), k = 5
                 (load_hat = load_t until k past loads exist; the reset observation counts as one)
  oracle_1step   clairvoyant one-step: peeks at the true next tick (a clone of the env, same RNG state) for each of
                 the 3 actions and takes the best immediate reward (ties: hold, down, up). Knows next load and read.
Cycle 10 delay variants (orig_variants '<variant>_d<D>'; state[6] = pending pods). Every rule moves the TOTAL
(active + pending) pods toward its target, so it does not re-request pods that are already starting (without a
delay pending = 0 and every rule is exactly the cycle-9 rule):
  fixed_N, causal, causal_be, causal_trend   target vs active + pending (read_hat still from ACTIVE VMs, which is
                 what capacity reflects)
  hpa_demand     load vs u * capacity * (active + pending) / active (the capacity once the pending pods are up)
  hpa_rule       his rule, unchanged (it sees the 6 verbatim features; it ignores pods)
  oracle_1step   unchanged (best immediate reward; with a delay this is myopic by construction)
  causal_trend_d (delay variants only) causal on load_hat = load_t + D * (load_t - load_{t-k}) / k, k = 5: the k=5
                 slope extrapolated D ticks ahead (load_hat = load_t until k past loads exist)
  oracle_Dstep   (delay variants only) clairvoyant: a clone of the env (same RNG state) is stepped D + 1 ticks with
                 hold, giving the TRUE load and read (capacity / (10 active VMs)) at the tick where a pod requested now
                 first serves; target = the causal_be rule on that load/read. Knows the future; not an upper bound.
Splits: run_legacy.SPLITS val 1100-1104 (selection) and --report-splits (default test5 = 6000-6019). Episode with seed s
= a fresh env on RandomState(s), the same trace the c5 protocol evaluates agents on (c5_protocol.make_env), so every
comparison is paired per trace. Each policy reports mean/sd, mean pods, frac_const_pods and per_trace {seed: return}.
Telemetry (PLAN 7): $QRL_RUN_DIR/metrics.jsonl (step = episodes evaluated) and progress.json.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tuning.legacy import c5_protocol as C5  # noqa: E402
from tuning.legacy import orig_variants as OV  # noqa: E402
from tuning.legacy import run_legacy as RL  # noqa: E402
from tuning.legacy import simpleagent  # noqa: E402

FIXED = tuple(range(1, 21))
NMIN, NMAX = 1, 20
BREAK_EVEN_READ = 0.3       # 10 * read = 3: below it an extra VM serves less than it costs
HPA_U = (0.6, 0.7, 0.8, 0.9, 1.0)
HPA_BAND = 0.2
TREND_K = 5


def toward(n, target):
    return 2 if n < target else (0 if n > target else 1)


def pods(s):
    """Pods the controller has asked for: active (s[5]) + pending (s[6], cycle-10 delay variants only)."""
    return s[5] if len(s) <= 6 else s[5] + s[6]


def hold(N):
    return lambda env: (lambda s: toward(pods(s), N))


def causal_target(s, break_even=False, load=None):
    """load: override of the observed load (causal_trend passes its forecast)."""
    load = float(s[0]) if load is None else float(load)
    cap, n = float(s[1]), float(s[5])
    if load <= 0:
        return NMIN
    if cap <= 0:  # zero inferred read share: break-even says a VM serves nothing (Codex c9 review)
        return NMIN if break_even else NMAX
    read_hat = cap / (10.0 * n)
    if break_even and read_hat < BREAK_EVEN_READ:
        return NMIN
    return int(min(NMAX, max(NMIN, math.ceil(load / (10.0 * read_hat)))))


def causal(break_even=False):
    return lambda env: (lambda s: toward(pods(s), causal_target(s, break_even)))


def hpa_demand(u):
    def act(s):
        load, cap = float(s[0]), float(s[1])
        if len(s) > 6 and s[6] > 0:          # cycle 10: capacity once the pending pods are active
            cap = cap * (float(s[5]) + float(s[6])) / float(s[5])
        return 2 if load > u * cap else (0 if load < (u - HPA_BAND) * cap else 1)
    return lambda env: act


def causal_trend(k=TREND_K):
    """Per episode (policy built per env): remembers the observed loads; the first call sees the reset observation."""
    def make(env):
        hist = []

        def act(s):
            hist.append(float(s[0]))
            lt = hist[-1]
            lhat = lt + k * (lt - hist[-1 - k]) if len(hist) > k else lt
            return toward(pods(s), causal_target(s, load=lhat))
        return act
    return make


def causal_trend_d(k=TREND_K):
    """Cycle 10: the k-tick slope extrapolated D+1 ticks ahead: a pod requested by the next action (t+1) serves from
    t+1+D (Codex c10 review)."""
    def make(env):
        D, hist = env.delay, []

        def act(s):
            hist.append(float(s[0]))
            lt = hist[-1]
            lhat = lt + (D + 1) * (lt - hist[-1 - k]) / k if len(hist) > k else lt
            return toward(pods(s), causal_target(s, load=lhat))
        return act
    return make


def oracle_dstep(env):
    """Cycle 10: true load/read D + 1 ticks ahead (clone stepped with hold), causal_be target on them."""
    def act(s):
        c = env.clone()
        for _ in range(env.delay + 1):
            st = c.step(1)[0]
        n = float(st[5])
        fut = np.array([st[0], st[1], 0, 0, 0, n], dtype=float)
        return toward(pods(s), causal_target(fut, break_even=True))
    return act


def oracle_1step(env):
    def act(s):
        best, best_r = 1, None
        for a in (1, 0, 2):
            r = env.clone().step(a)[1]
            if best_r is None or r > best_r:
                best, best_r = a, r
        return best
    return act


def _hpa_rule(env):
    act = simpleagent.SimpleAgent(50, 50).act
    return act if getattr(env, "delay", None) is None else (lambda s: act(s[:6]))


POLICIES = {
    "hpa_rule": _hpa_rule,
    "always_down": lambda env: (lambda s: 0),
    "causal": causal(False),
    "causal_be": causal(True),
    "causal_trend": causal_trend(),
    "oracle_1step": oracle_1step,
}
DELAY_POLICIES = {"causal_trend_d": causal_trend_d(), "oracle_Dstep": oracle_dstep}


def policies(variant):
    """The cycle-9 set; delay variants add causal_trend_d and oracle_Dstep."""
    return dict(POLICIES, **DELAY_POLICIES) if OV.startup_delay(variant) is not None else dict(POLICIES)


def evaluate(variant, make_policy, seeds, ticks=RL.EP_TICKS):
    """One fresh env per seed (RandomState(seed)); stats as c5_protocol.eval_episode (+ seed)."""
    eps = []
    for s in seeds:
        env = OV.make(variant, int(s))
        s0 = env.reset()
        e = C5.eval_episode(env, make_policy(env), False, s0, ticks)
        e["seed"] = int(s)
        eps.append(e)
    r = RL.summarize(eps, comp_names=OV.COMP)
    return dict(mean_ret=r["mean_ret"], sd_ret=r["sd_ret"], mean_pods=r["mean_pods"],
                frac_const_pods=float(np.mean([e["frac_const_pods"] for e in eps])),
                components_mean=r["components_mean"], p95_latency_mean=r["p95_latency_mean"], n_episodes=len(eps),
                per_trace={str(e["seed"]): e["ret"] for e in eps}, episodes=eps)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--orig-variant", default="orig", type=OV.check_variant,
                   help=f"{OV.VARIANTS}, optionally + _d<D> (cycle-10 startup delay)")
    p.add_argument("--seed", type=int, default=int(os.environ.get("QRL_RUN_SEED", 0)))
    p.add_argument("--report-splits", default="test5", help="comma list of held-out splits (not val/fair)")
    p.add_argument("--ticks", type=int, default=RL.EP_TICKS, help="ticks per episode (5000; smaller only for tests)")
    p.add_argument("--out", required=True)
    p.add_argument("--confirm-manifest", default=None,
                   help="test10/test11 only: confirmation manifest frozen for that split by make_c18.py --confirm "
                        "(= env QRL_CONFIRM_MANIFEST)")
    a = p.parse_args(argv)
    reps = RL.parse_splits(a.report_splits)
    if any(sp in ("val", "fair") for sp in reps):
        p.error("--report-splits takes held-out splits only (selection is on val)")
    from tuning.legacy import confirm_guard as CG      # test10/test11: frozen, unused manifest of that split
    CG.require(reps, "orig_baselines.py", list(sys.argv[1:] if argv is None else argv), a.confirm_manifest)
    torch.set_num_threads(1)
    RL.check_seeds(a.seed, "fair", ("val",) + reps)
    CG.authorise(reps, "orig_baselines.main")
    RL.seed_all(a.seed)
    val = RL.SPLITS["val"]
    pols = policies(a.orig_variant)
    n_total = (len(FIXED) + len(HPA_U) + len(pols)) * len(val) \
        + (2 + len(pols)) * sum(len(RL.SPLITS[sp]) for sp in reps)
    tel = RL.Telemetry(os.environ.get("QRL_RUN_DIR"), n_total)
    t0, c0 = time.time(), time.process_time()
    done = [0]

    def ev(make_policy, seeds):
        r = evaluate(a.orig_variant, make_policy, seeds, a.ticks)
        done[0] += len(seeds)
        tel.log(done[0], mean_ret=r["mean_ret"], mean_pods=r["mean_pods"], frac_const_pods=r["frac_const_pods"])
        tel.progress(done[0])
        return r

    res = dict(env_family="orig", orig_variant=a.orig_variant, seed=a.seed, episode_ticks=a.ticks,
               val_seeds=list(val), selection="fixed_N, hpa_demand u: max mean episode return on val",
               report_splits=list(reps),
               **({"startup_delay": OV.startup_delay(a.orig_variant)} if OV.startup_delay(a.orig_variant) is not None
                  else {}),
               reward_components=list(OV.COMP), val={})
    fixed = {n: ev(hold(n), val) for n in FIXED}
    best = max(fixed, key=lambda n: fixed[n]["mean_ret"])     # first of equal maxima (smallest N)
    res["val_fixed"] = {str(n): {k: v[k] for k in ("mean_ret", "mean_pods", "per_trace")} for n, v in fixed.items()}
    hpa = {u: ev(hpa_demand(u), val) for u in HPA_U}
    best_u = max(hpa, key=lambda u: hpa[u]["mean_ret"])
    res["val_hpa_demand"] = {str(u): {k: v[k] for k in ("mean_ret", "mean_pods", "per_trace")} for u, v in hpa.items()}
    res["selected"] = {"fixed_best": {"N": int(best)}, "hpa_demand": {"u": best_u, "band": HPA_BAND}}
    res["val"]["fixed_best"] = fixed[best]
    res["val"]["hpa_demand"] = hpa[best_u]
    for name, mk in pols.items():
        res["val"][name] = ev(mk, val)
    res["val_cpu_s"] = time.process_time() - c0
    for sp in reps:                 # held-out: frozen selection only
        seeds = RL.SPLITS[sp]
        res[f"{sp}_seeds"] = list(seeds)
        res[sp] = {"fixed_best": ev(hold(int(best)), seeds), "hpa_demand": ev(hpa_demand(best_u), seeds)}
        for name, mk in pols.items():
            res[sp][name] = ev(mk, seeds)
    res["cpu_s"], res["wall_s"] = time.process_time() - c0, time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"variant": a.orig_variant, "fixed_best_N": int(best), "cpu_s": round(res["cpu_s"], 1),
                      **{f"{sp}_mean_ret": {k: round(v["mean_ret"]) for k, v in res[sp].items()}
                         for sp in ("val",) + reps}}))
    return res


if __name__ == "__main__":
    main()
