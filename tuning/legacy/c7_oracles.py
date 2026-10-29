#!/usr/bin/env python3
"""Cycle 7 (Track L): oracle and controller headroom on the coupled lstm_transfer env (design: .lab/tuning/cycles/
c6_codex.md + c6_synthesis.md). Nothing is trained; no policy draws random numbers.

  python3 tuning/legacy/c7_oracles.py --split val|test3|test4 --out tuning/results/c7/oracles_<split>.json [--seed 0]

Env and traces: exactly the evaluation episode of c5_protocol.evaluate / coupled_baselines.py: the verbatim
lstm_transfer env (env_transfer.py) wrapped by coupled_env, on a private numpy stream RandomState(s) (= np.random.seed(s)),
reset (start 6 pods; the reset draws are consumed and not rewarded), then 5000 ticks; action 0/1/2 = -1/hold/+1 pod,
bounds 1..20. The env draws a fixed number of values per tick independent of the action, so the exogenous draws
(L_t, c0_t, m0_t, l0_t), t = 1..5000, are generated ONCE per trace (draws()) and the reward of every pod count n at every
tick, R_t(n) (reward_matrix(), same expression and float operation order as coupled_env.CoupledEnv.step), is known.
Every policy's action sequence is then REPLAYED through a fresh env and its replayed return must equal the computed one
exactly (assert; the reported statistics come from the replay).

Policies (all start at 6 pods, +-1 per tick):
  fixed_5            ramp to N = 5 and hold (paired reference; also the val choice of cycle 6)
  fixed_valsel       ramp to N and hold, N in 1..20 maximising the mean return on the val traces (computed here)
  clairvoyant_greedy per tick the reachable n maximising that tick's realised R_t(n) (ties: hold, down, up)
  clairvoyant_dp     backward recursion V_t(n) = max_{m reachable from n} [R_t(m) + V_{t+1}(m)] on the realised draws:
                     the episode-optimal return under the +-1/bounds constraints (upper bound for every policy)
  loadonly_greedy    knows L_t and t only; metric noise marginalised (expected reward E[R_t(n) | L_t, t]); greedy
  loadonly_dp        same expected reward, backward recursion (plan fixed by the load trace; realised return reported)
  causal_persist     uses only the agent's observation window (what the LSTM sees, 10 x [L, cap, c, m, l, n]):
                     L_{t+1} ~ L_t, de-coupled latency l0 = l_t / g_t with g_t = 6 L_t / (70 n_t), l0_{t+1} ~ l0_t;
                     cpu/mem noise marginalised over the env's uniform ranges (no clock, so no spike knowledge);
                     picks -1/hold/+1 by one-step expected reward
  causal_trend       as causal_persist with L and l0 extrapolated one tick by a least-squares line over the valid
                     window rows (an extra variant, not in the c6 brief)
Splits: val 1100-1104 (run_legacy.SPLITS), test3 4000-4019 (NEW, defined here; never used before cycle 7),
test4 5000-5019 (cycle 8, run_legacy.SPLITS; paired with the cycle-8 agents' test4 evaluation).
fixed_valsel's N is always selected on val, also when --split test3. Paired differences are vs fixed_5, per trace.
Telemetry (PLAN 7): $QRL_RUN_DIR/metrics.jsonl (step = traces done) and progress.json.
"""
import argparse
import inspect
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

from tuning.legacy import c5_protocol as C5  # noqa: E402
from tuning.legacy import coupled_env as CE  # noqa: E402
from tuning.legacy import env_transfer  # noqa: E402
from tuning.legacy import run_legacy as RL  # noqa: E402

ENV_NAME = "lstm_transfer"
TEST3_SEEDS = tuple(range(4000, 4020))
SPLITS = {"val": tuple(RL.SPLITS["val"]), "test3": TEST3_SEEDS, "test4": tuple(RL.SPLITS["test4"])}
N_MIN, N_MAX, N0 = 1, 20, 6
NS = np.arange(N_MIN, N_MAX + 1, dtype=float)
T = RL.EP_TICKS
_E = env_transfer.Environment()          # constants only (ranges, periods, spike); never stepped
LAT_SD = 0.1                             # env_transfer.py latency noise N(0, 0.1)
POLICIES = ("fixed_5", "fixed_valsel", "clairvoyant_greedy", "clairvoyant_dp", "loadonly_greedy", "loadonly_dp",
            "causal_persist", "causal_trend")


# ---------------------------------------------------------------- exogenous draws and rewards
def draws(seed, ticks=T):
    """(L, c0, m0, l0) arrays for ticks 1..ticks of the trace of eval seed `seed`: the raw (uncoupled) values the
    coupled env feeds into coupling(). Same stream as c5_protocol.make_env (RandomState(seed)), reset first."""
    env = C5.StreamEnv(env_transfer.Environment(), int(seed))
    env.reset()
    out = np.empty((ticks, 4))
    for t in range(ticks):
        s = env.step(1)[0]
        out[t] = (s[0], s[2], s[3], s[4])
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


def reward_matrix(L, c0, m0, l0):
    """R[t, n-1] = coupled reward at tick t+1 with n pods; bit-identical to CoupledEnv.step (same operations, order)."""
    L, c0, m0, l0 = (np.asarray(x, dtype=float)[:, None] for x in (L, c0, m0, l0))
    n = NS[None, :]
    g = CE.REF_PODS * L / (CE.REF_LOAD * n)
    c = np.minimum(CE.PCT_MAX, g * c0)
    m = np.minimum(CE.PCT_MAX, g * m0)
    lat = g * l0
    return 3.0 * L + -5.0 * lat + -10.0 * n + -0.5 * c + -0.5 * m


def _in_spike(t):
    return (_E.spike_position <= t) & (t < _E.spike_position + _E.spike_length)


def _e_min_uniform(g, a, b, cap=CE.PCT_MAX):
    """E[min(cap, g U)] for U ~ U(a, b), g > 0 (array g, scalar or broadcastable a, b)."""
    g = np.asarray(g, dtype=float)
    a = np.broadcast_to(a, g.shape).astype(float)
    b = np.broadcast_to(b, g.shape).astype(float)
    full = g * (a + b) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.clip(cap / g, a, b)
        part = (g * (u * u - a * a) / 2.0 + cap * (b - u)) / (b - a)
    return np.where(g * b <= cap, full, np.where(g * a >= cap, cap, part))


def _rectified_mean(mu, sd):
    z = mu / sd
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return mu * Phi + sd * phi


def expected_reward_matrix(L, ticks=None):
    """E[R_t(n) | L_t, t]: load known, cpu/mem/latency noise marginalised (env_transfer.py laws, spike included)."""
    L = np.asarray(L, dtype=float)
    t = np.arange(1, len(L) + 1, dtype=float) if ticks is None else np.asarray(ticks, dtype=float)
    sp = _in_spike(t)
    bump = np.where(sp, _E.spike_amplitude, 0.0)
    lat_mu = _E.latency_amplitude * np.sin(2 * np.pi * 2 * t / _E.load_period) + _E.latency_amplitude
    lat_mu = lat_mu + np.where(sp, _E.spike_amplitude * 0.8 * np.abs(
        np.sin(2 * np.pi * 20 * (t - _E.spike_position) / _E.spike_length)), 0.0)
    el0 = _rectified_mean(lat_mu, LAT_SD)
    g = CE.REF_PODS * L[:, None] / (CE.REF_LOAD * NS[None, :])
    ec = _e_min_uniform(g, (_E.cpu_range[0] + bump)[:, None], (_E.cpu_range[1] + bump)[:, None])
    em = _e_min_uniform(g, (_E.memory_range[0] + bump)[:, None], (_E.memory_range[1] + bump)[:, None])
    return 3.0 * L[:, None] - 5.0 * g * el0[:, None] - 10.0 * NS[None, :] - 0.5 * ec - 0.5 * em


# ---------------------------------------------------------------- planners on a reward matrix
def _step(n, a):
    return min(N_MAX, n + 1) if a == 2 else (max(N_MIN, n - 1) if a == 0 else n)


def path_return(R, actions, n0=N0):
    """Forward sum (same order as the env evaluation) of R along the pod path of `actions`; returns (ret, pods)."""
    n, tot, pods = n0, 0.0, []
    for t, a in enumerate(actions):
        n = _step(n, a)
        tot += float(R[t, n - 1])
        pods.append(n)
    return tot, pods


def fixed_actions(N, ticks=T, n0=N0):
    acts, n = [], n0
    for _ in range(ticks):
        a = 2 if n < N else (0 if n > N else 1)
        acts.append(a)
        n = _step(n, a)
    return acts


def greedy_actions(R, n0=N0):
    """Per tick the reachable n with the largest R_t(n); ties resolved hold, down, up."""
    acts, n = [], n0
    for t in range(R.shape[0]):
        best_a, best = None, -np.inf
        for a in (1, 0, 2):
            m = _step(n, a)
            if (a == 0 and m == n) or (a == 2 and m == n):
                continue
            if R[t, m - 1] > best:
                best_a, best = a, R[t, m - 1]
        acts.append(best_a)
        n = _step(n, best_a)
    return acts


def dp_actions(R, n0=N0):
    """Backward recursion V_t(n) = max_{m in {n-1,n,n+1} within bounds} [R_t(m) + V_{t+1}(m)], V_{T+1} = 0.
    Returns (actions, V_1(n0)) (V by backward summation; the reported return is the forward path sum)."""
    Tn, K = R.shape
    V = np.zeros(K)
    choice = np.empty((Tn, K), dtype=np.int8)        # action taken at tick t from n_{t-1} = index k
    for t in range(Tn - 1, -1, -1):
        W = R[t] + V
        down = np.concatenate(([-np.inf], W[:-1]))     # m = n-1 (not available at n = 1)
        up = np.concatenate((W[1:], [-np.inf]))        # m = n+1 (not available at n = 20)
        stack = np.stack([W, down, up])                # order hold, down, up -> ties resolved hold, down, up
        idx = np.argmax(stack, axis=0)
        choice[t] = np.array([1, 0, 2], dtype=np.int8)[idx]
        V = stack[idx, np.arange(K)]
    acts, n = [], n0
    for t in range(Tn):
        a = int(choice[t, n - 1])
        acts.append(a)
        n = _step(n, a)
    return acts, float(V[n0 - 1])


# ---------------------------------------------------------------- causal controller
class CausalController:
    """Decides from the observation window ONLY (the LSTM agents' input, RL.SEQ x [L, cap, c, m, l, n], coupled values,
    zero rows = padding after reset). No clock, no trace, no env handle."""

    def __init__(self, trend=False):
        self.trend = bool(trend)
        self.c_rng = tuple(float(x) for x in _E.cpu_range)
        self.m_rng = tuple(float(x) for x in _E.memory_range)

    def _predict(self, y):
        if not self.trend or len(y) < 2:
            return float(y[-1])
        x = np.arange(len(y), dtype=float)
        k, b = np.polyfit(x, y, 1)
        return float(k * len(y) + b)

    def act(self, window):
        w = np.asarray(window, dtype=float)
        rows = w[w[:, 5] >= 1]                         # observed rows (padding has n = 0)
        L, lat, n = rows[:, 0], rows[:, 4], rows[:, 5]
        g = CE.REF_PODS * L / (CE.REF_LOAD * n)
        ok = g > 0
        l0 = np.where(ok, lat / np.where(ok, g, 1.0), 0.0)
        L_hat = max(self._predict(L), 0.0)
        l0_hat = max(self._predict(l0[ok]) if ok.any() else 0.0, 0.0)
        n_now = int(round(n[-1]))
        best_a, best = 1, -np.inf
        for a in (1, 0, 2):
            m = _step(n_now, a)
            if a != 1 and m == n_now:
                continue
            gm = CE.REF_PODS * L_hat / (CE.REF_LOAD * m)
            er = (3.0 * L_hat - 5.0 * gm * l0_hat - 10.0 * m
                  - 0.5 * float(_e_min_uniform(np.array([gm]), *self.c_rng)[0])
                  - 0.5 * float(_e_min_uniform(np.array([gm]), *self.m_rng)[0]))
            if er > best:
                best_a, best = a, er
        return best_a

    __call__ = act


# ---------------------------------------------------------------- replay through the real env
def replay(seed, policy, ticks=T):
    """Run `policy(window) -> action` (or a list of actions) on a fresh coupled env for seed; c5 eval statistics."""
    env = C5.make_env(ENV_NAME, "coupled", int(seed))
    if not callable(policy):
        it = iter(list(policy))
        policy = lambda _w: next(it)  # noqa: E731
    rec = []

    def logged(w):
        a = policy(w)
        rec.append(int(a))
        return a
    ep = C5.eval_episode(env, logged, True, env.reset(), ticks)
    ep["actions"] = rec
    return ep


def trace_policies(seed, N_val, ticks=T):
    """All policies on one trace: computed returns, replayed episodes; asserts replay == computed for the planned
    policies (causal: the replayed actions re-evaluated on R)."""
    L, c0, m0, l0 = draws(seed, ticks)
    R = reward_matrix(L, c0, m0, l0)
    ER = expected_reward_matrix(L)
    plans = {"fixed_5": fixed_actions(5, ticks), "fixed_valsel": fixed_actions(N_val, ticks),
             "clairvoyant_greedy": greedy_actions(R), "loadonly_greedy": greedy_actions(ER)}
    plans["clairvoyant_dp"], v_dp = dp_actions(R)
    plans["loadonly_dp"], _ = dp_actions(ER)
    out = {}
    for name in POLICIES:
        if name.startswith("causal"):
            ep = replay(seed, CausalController(trend=name == "causal_trend"), ticks)
            comp, _ = path_return(R, ep["actions"])
        else:
            comp, _ = path_return(R, plans[name])
            ep = replay(seed, plans[name], ticks)
            if ep["actions"] != list(plans[name]):
                raise AssertionError(f"{name} seed {seed}: replayed actions differ")
        if ep["ret"] != comp:
            raise AssertionError(f"{name} seed {seed}: replay return {ep['ret']!r} != computed {comp!r}")
        ep.pop("actions")
        ep["computed_ret"] = comp
        out[name] = ep
    out["clairvoyant_dp"]["dp_backward_value"] = v_dp
    return out


def val_select(seeds=None):
    """fixed_ramped N in 1..20 with the largest mean computed return on the val traces (ties: smaller N)."""
    seeds = SPLITS["val"] if seeds is None else seeds
    tot = np.zeros(N_MAX)
    for s in seeds:
        R = reward_matrix(*draws(s))
        for N in range(N_MIN, N_MAX + 1):
            tot[N - 1] += path_return(R, fixed_actions(N))[0]
    means = tot / len(seeds)
    return int(np.argmax(means)) + 1, {str(N): float(means[N - 1]) for N in range(N_MIN, N_MAX + 1)}


def paired(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    return dict(mean=float(d.mean()), sd=sd, se=sd / math.sqrt(len(d)) if len(d) > 1 else 0.0, min=float(d.min()),
                max=float(d.max()), n_positive=int((d > 0).sum()), n=int(len(d)))


def run(split, seed=0, tel_dir=None, seeds=None):
    seeds = tuple(SPLITS[split] if seeds is None else seeds)
    tel = RL.Telemetry(tel_dir, len(seeds))
    t0, c0 = time.time(), time.process_time()
    N_val, val_means = val_select()
    per = {p: [] for p in POLICIES}
    for i, s in enumerate(seeds, 1):
        r = trace_policies(s, N_val)
        for p in POLICIES:
            per[p].append(r[p])
        tel.log(i, **{f"{p}_ret": r[p]["ret"] for p in POLICIES})
        tel.progress(i)
    pol = {}
    ref = [e["ret"] for e in per["fixed_5"]]
    for p in POLICIES:
        rets = [e["ret"] for e in per[p]]
        pol[p] = dict(mean_ret=float(np.mean(rets)), sd_ret=float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0,
                      per_trace_ret=rets, mean_pods=float(np.mean([e["mean_pods"] for e in per[p]])),
                      p95_latency_mean=float(np.mean([e["p95_latency"] for e in per[p]])),
                      components_mean={c: float(np.mean([e[c] for e in per[p]])) for c in RL.COMP},
                      paired_vs_fixed_5=paired(rets, ref))
    pol["clairvoyant_dp"]["dp_backward_value_per_trace"] = [e["dp_backward_value"] for e in per["clairvoyant_dp"]]
    return dict(protocol="c7_oracles", split=split, seeds=list(seeds), seed=int(seed), env=ENV_NAME,
                env_variant="coupled", start_pods=N0, bounds=[N_MIN, N_MAX], episode_ticks=T,
                trace_generation="fresh coupled env on RandomState(s) (= np.random.seed(s)), reset, 5000 ticks",
                replay_equality="asserted exactly for every policy and trace",
                fixed_valsel_N=N_val, val_fixed_ramped_mean_ret=val_means, policies=pol,
                cpu_s=time.process_time() - c0, wall_s=time.time() - t0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", choices=tuple(SPLITS), required=True)
    p.add_argument("--seed", type=int, default=int(os.environ.get("QRL_RUN_SEED", 0)),
                   help="recorded only (no policy draws random numbers)")
    p.add_argument("--config", default=None, help="optional JSON; key 'seeds' overrides the split's seed list "
                                                  "(smoke tests only)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    seeds = None
    if a.config:
        with open(a.config) as f:
            seeds = json.load(f).get("seeds")
    res = run(a.split, a.seed, os.environ.get("QRL_RUN_DIR"), seeds)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"split": a.split, "fixed_valsel_N": res["fixed_valsel_N"], "cpu_s": round(res["cpu_s"], 1),
                      "mean_ret": {k: round(v["mean_ret"]) for k, v in res["policies"].items()},
                      "paired_vs_fixed_5": {k: round(v["paired_vs_fixed_5"]["mean"]) for k, v in
                                            res["policies"].items()}}))
    return res


def causal_inputs():
    """Names of the parameters CausalController.act receives (API check: the window only)."""
    return tuple(inspect.signature(CausalController.act).parameters)[1:]


if __name__ == "__main__":
    main()
