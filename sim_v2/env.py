"""Sim v2: transient fluid-queue autoscaling simulator (council C-20261016-01, approval R-20261016-013).

Model (equations in README.md):
  - pods: `ready` serve; `pending` pods are billed but serve only after `startup_delay_steps`; removal immediate
    (pending pods are cancelled first, newest first, then ready pods).
  - capacity C = mu * ready**theta, times (1 - delta) for `rebalance_steps` steps after an accepted change.
  - queue: fluid FIFO cohorts at `dt_s` resolution; backlog is served first; cohorts older than `timeout_s`
    fail. Fresh work served within its own sub-step has the per-pod M/M/1 sojourn tail
    P(S > x) = exp(-mu_pod (1 - rho) x); work served from the backlog waited >= dt_s.
  - bad requests = failed + served with latency > SLO; each request is counted once; work still queued at the
    end of the episode is settled as failed.
  - reward r = -(alpha * billed pod-seconds / cost_norm) - beta * bad / sla_norm - gamma * |accepted change|.
Actions: 0 = down, 1 = no-op, 2 = up; with max_delta = 2 also 3 = down 2, 4 = up 2.
HPA interface (both options default OFF, so every default trajectory is bit-identical to Sim v2/v2.1):
  replica_actions=True : the action is an absolute desired replica count (ready + pending) in [min_pods, max_pods]
                         instead of a delta; any change size is applied in one step (Kubernetes scale subresource).
  util_obs=True        : appends `cpu_util_pct` to the observation = the same noisy CPU reading as cpu_pct
                         (100 * served / capacity + cpu_offset_pct + noise, same random draw) WITHOUT the [0, 100]
                         clip; NaN at reset (no metric window yet). See README "HPA baseline".
All parameters are provisional (params.py).
"""
from collections import deque
import hashlib
import json
import math
import os

import numpy as np

from params import SimParams  # noqa: F401  (re-exported: `from env import SimParams`)

LN20 = math.log(20.0)
ACTION_DELTA = {0: -1, 1: 0, 2: 1, 3: -2, 4: 2}
# Default observation: only what can be instrumented on the real cluster (README "Observations").
OBS_NAMES = ("completed_ops", "cpu_pct", "p95_ms", "failed_per_s", "ready", "pending")
# oracle_obs=True appends simulator-internal quantities that are NOT deployable telemetry.
ORACLE_OBS_NAMES = OBS_NAMES + ("offered_lambda", "backlog")
UTIL_OBS_NAME = "cpu_util_pct"   # util_obs=True appends it last (metrics-server-like CPU usage / request, unclipped)


REWARD_DINOS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward_dinos.json")
_DINOS_CACHE = {}


def _sandbox_seed_guard(seed):
    """Runtime seed guard (sandbox runs only; no effect on dynamics). When the launcher sets QRL_SANDBOX_SEEDS
    (e.g. "0-999,1000-1004") a seed outside those ranges raises PermissionError. Unset = no check."""
    spec = os.environ.get("QRL_SANDBOX_SEEDS")
    if spec is None:
        return
    s = int(seed)
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        if a and int(a) <= s <= int(b or a):
            return
    raise PermissionError(f"seed {s} is outside the sandbox seed ranges QRL_SANDBOX_SEEDS={spec!r}")


def load_reward_dinos(path=REWARD_DINOS_PATH):
    """Frozen DInos-reward adapter constants + their sha256 (the file is listed in experiments freeze.FROZEN)."""
    if path not in _DINOS_CACHE:
        raw = open(path, "rb").read()
        _DINOS_CACHE[path] = (json.loads(raw), hashlib.sha256(raw).hexdigest())
    return _DINOS_CACHE[path]


def n_actions(params):
    return 3 if params.max_delta == 1 else 5


def pod_transition(P, ready, pipe, rb, action, absolute=False):
    """Pure pod-lifecycle transition, shared by SimV2 and the oracle DP.

    State before step: `ready` pods, `pipe` (tuple, len d; pipe[i] pods become ready at the start of the step
    i+1 steps from now), `rb` rebalance steps still owed. Order: promote, apply action, rebalance.
    absolute=True: `action` is the desired total replica count (HPA interface), not an action index.
    Returns (ready_serving, pipe_after, rebalancing_now, abs_change, billed_pods, rb_next).
    """
    d = P.startup_delay_steps
    if d > 0:
        promoted = pipe[0]
        r = ready + promoted
        p = list(pipe[1:]) + [0]
    else:
        promoted, r, p = 0, ready, []
    topo = promoted > 0
    total = r + sum(p)
    if absolute:
        target = int(action)
        if not P.min_pods <= target <= P.max_pods:
            raise ValueError(f"desired replicas {target} outside [{P.min_pods}, {P.max_pods}]")
    else:
        delta = ACTION_DELTA[action]
        if abs(delta) > P.max_delta:
            raise ValueError(f"action {action} not allowed with max_delta={P.max_delta}")
        target = min(max(total + delta, P.min_pods), P.max_pods)
    change = target - total
    if change > 0:
        if d > 0:
            p[-1] += change
        else:
            r += change
            topo = True
    elif change < 0:
        need = -change
        for i in range(len(p) - 1, -1, -1):  # cancel newest pending first
            take = min(need, p[i])
            p[i] -= take
            need -= take
        if need:
            r -= need
            topo = True
    if (P.rebalance_trigger == "action" and change != 0) or (P.rebalance_trigger == "topology" and topo):
        rb = P.rebalance_steps
    elif P.rebalance_trigger not in ("action", "topology"):
        raise ValueError("rebalance_trigger must be 'action' or 'topology'")
    active = rb > 0
    return r, tuple(p), active, abs(change), r + sum(p), max(rb - 1, 0)


class SimV2:
    """Gym-like env. reset(seed) -> (obs, info); step(a) -> (obs, reward, terminated, truncated, info)."""

    def __init__(self, params=None, workload="W0", trace=None, debug=False, oracle_obs=False,
                 replica_actions=False, util_obs=False):
        self.P = params or SimParams()
        self.oracle_obs = bool(oracle_obs)
        self.replica_actions = bool(replica_actions)
        self.util_obs = bool(util_obs)
        self.obs_names = (ORACLE_OBS_NAMES if self.oracle_obs else OBS_NAMES) + ((UTIL_OBS_NAME,) if self.util_obs else ())
        self.workload = workload
        self.fixed_trace = None if trace is None else np.asarray(trace, dtype=float)
        self.debug = debug
        self.n_actions = n_actions(self.P)
        self.trace = None
        if self.P.reward_mode not in ("cost", "dinos"):
            raise ValueError("reward_mode must be 'cost' or 'dinos'")
        self.dinos, self.dinos_sha256 = load_reward_dinos() if self.P.reward_mode == "dinos" else (None, None)
        if self.dinos is not None:
            vs = self.dinos["variants"]
            if self.P.dinos_variant not in vs:
                raise ValueError(f"dinos_variant must be one of {sorted(vs)}")
            self.dinos = {**self.dinos, **vs[self.P.dinos_variant]}
        self._mix = None

    # ----- lifecycle -----
    def reset(self, seed=0, init_pods=None):
        """init_pods overrides params.init_pods for this episode (used by the fixed(c) controls)."""
        _sandbox_seed_guard(seed)
        P = self.P
        ss = np.random.SeedSequence([int(seed), 0x51A2])
        self.rng = np.random.default_rng(ss.spawn(1)[0])  # telemetry noise only
        if self.fixed_trace is not None:
            self.trace = self.fixed_trace
        else:
            from workloads import make_workload
            wl = make_workload(self.workload) if isinstance(self.workload, str) else self.workload
            self.trace = wl.trace(seed, P.episode_steps)
        self.T = len(self.trace)
        self.t = 0
        self.ready = P.init_pods if init_pods is None else int(init_pods)
        if not P.min_pods <= self.ready <= P.max_pods:
            raise ValueError("init_pods outside [min_pods, max_pods]")
        self.pipe = tuple([0] * P.startup_delay_steps)
        self.rb = 0
        self.q = deque()  # FIFO cohorts [stamp_s, amount]
        self.backlog = 0.0
        self.tot = dict(arrived=0.0, served=0.0, failed=0.0, slo_violations=0.0)
        self.last_p95_ms = 1000.0 * LN20 / P.mu_ops_per_pod * P.tail_scale + 1000.0 * P.latency_add_s
        obs = self._obs(0.0, 0.0, 0.0, self.last_p95_ms, self.ready, 0, P.capacity(self.ready, False), at_reset=True)
        return obs, {"t": 0}

    @property
    def outstanding(self):
        return self.backlog

    # ----- dynamics -----
    def step(self, action):
        P = self.P
        if self.t >= self.T:
            raise RuntimeError("episode finished; call reset()")
        r, self.pipe, active, dpods, billed, self.rb = pod_transition(P, self.ready, self.pipe, self.rb, int(action),
                                                                   absolute=self.replica_actions)
        self.ready = r
        C = P.capacity(r, active)
        mu_pod = C / r
        lam = float(self.trace[self.t])
        T = P.control_step_s
        slo = P.slo_eff_s
        arrived = lam * T
        self._mix = None
        if self.backlog <= 0.0 and lam <= C:
            # fast path: no queue, constant rho within the step
            rho = lam / C
            served, failed = arrived, 0.0
            viol = served * math.exp(-mu_pod * (1.0 - rho) * slo)
            p95 = LN20 / (mu_pod * max(1.0 - rho, 1e-3))
        else:
            served, failed, viol, rho, p95 = self._substeps(lam, C, mu_pod, slo)
        self.t += 1
        terminated = self.t >= self.T
        settled = 0.0
        if terminated and self.backlog > 0.0:
            settled = self.backlog
            failed += settled
            self.q.clear()
            self.backlog = 0.0
        tot = self.tot
        tot["arrived"] += arrived
        tot["served"] += served
        tot["failed"] += failed
        tot["slo_violations"] += viol
        if self.debug:
            resid = tot["arrived"] - tot["served"] - tot["failed"] - self.backlog
            assert abs(resid) <= 1e-9 * max(1.0, tot["arrived"]), f"conservation violated: {resid}"
        bad = failed + viol
        cost = P.alpha * billed * T / P.cost_norm
        sla = P.beta * bad / P.sla_norm
        churn = P.gamma * dpods
        reward = -(cost + sla + churn)
        p95 = P.latency_add_s + P.tail_scale * p95  # observed p95 (identity at the defaults)
        p95_ms = 1000.0 * p95
        self.last_p95_ms = p95_ms
        pending = billed - r
        obs = self._obs(lam, served / T, failed / T, p95_ms, r, pending, C)
        if self.dinos is not None:
            reward, comps = self._dinos_reward(lam, served, failed, C, billed, T)
        info = {
            "t": self.t, "reward_cost": -cost, "reward_sla": -sla, "reward_churn": -churn,
            "billed_pod_s": billed * T, "bad_requests": bad, "failed": failed, "slo_violations": viol,
            "settled": settled, "arrived": arrived, "served": served, "outstanding": self.backlog,
            "p95_ms": p95_ms, "ready": r, "pending": pending, "capacity": C, "rho": rho,
            "accepted_change": dpods, "rebalancing": active, "lambda": lam,
        }
        if self.dinos is not None:
            info["dinos"] = comps
            info["reward_dinos"] = reward
        return obs, reward, terminated, False, info

    def _dinos_reward(self, lam, served, failed, C, billed, T):
        """DInos-style reward r = 3 L - 5 D - 10 N - 0.5 U - 0.5 M; coefficients from reward_dinos.json (verbatim
        from the old environment.py:90-94). All five terms are noise-free functions of the true state.
        L = scale * lambda_offered / mu_ref;  D = p95 over ALL requests of the step / SLO, where requests that
        failed (timed out) in this step count at timeout_s (so p95 = timeout_s when failed > 5% of
        served+failed); N = ready+pending (billed) pods; U = true CPU % = 100 served/(C T) in [0,100];
        M = constant placeholder memory % per pod (calibration stage B replaces it).
        Variants (reward_dinos.json 'variants'; file default d_def='mean', u_def='offered' = V2 since R-20261017-023; the definition described above is variant 'legacy_p95'): d_def='mean' uses
        the mean latency of all requests of the step (failed at timeout_s); u_def='offered' uses
        U = min(100, 100 lambda_offered / C). Prototypes, see experiments/dinos_simv2/reward_variants_report.md.
        Resolution-window convention (Codex review R-20261017-017): D is computed over the requests RESOLVED in
        this step, i.e. n_req = served + failed of the step. A request that arrives in step t and is still queued at
        the end of t is in neither count; it enters D of the later step in which it is served or times out (so a
        long queue is penalised when it resolves, with its age in the latency mixture). Whatever is still queued
        when the episode ends is charged as failures (info['settled']) INTO THE LAST STEP's failed count, hence
        into that step's D (the same rule as the cost reward). The sum over an episode of (served + failed) equals
        the total arrivals."""
        P, K = self.P, self.dinos
        L = K["load_scale"] * lam / P.mu_ops_per_pod
        n_req = served + failed
        d_def, u_def = K.get("d_def", "mean"), K.get("u_def", "offered")
        if n_req <= 0.0:
            p95_s = 0.0
        elif d_def == "mean":
            # variant V1/V2: MEAN latency over all requests of the step (failed at timeout_s); served part is the
            # volume-weighted mean of age + 1/rate (M/M/1 sojourn mean), mapped by the same affine latency model
            if served > 0.0:
                if self._mix is None:
                    m_srv = 1.0 / ((C / self.ready) * max(1.0 - lam / C, 1e-3))
                else:
                    w, ages, rates = self._mix
                    m_srv = sum(wi * (ai + 1.0 / ri) for wi, ai, ri in zip(w, ages, rates)) / sum(w)
                m_srv = P.latency_add_s + P.tail_scale * m_srv
            else:
                m_srv = 0.0
            p95_s = min((served * m_srv + failed * P.timeout_s) / n_req, P.timeout_s)
        else:
            tail = (K["latency_tail"] * n_req - failed) / served if served > 0.0 else -1.0
            if tail <= 0.0:
                p95_s = P.timeout_s
            elif self._mix is None:   # fast path: single M/M/1 mixture component, no queue
                rho = lam / C
                mu_pod = C / self.ready
                ln = LN20 if tail == K["latency_tail"] else math.log(1.0 / tail)
                p95_s = ln / (mu_pod * max(1.0 - rho, 1e-3))
            else:
                p95_s = _mixture_p95(*self._mix, tail=tail)
            p95_s = min(P.latency_add_s + P.tail_scale * p95_s, P.timeout_s)
        D = p95_s / P.slo_s
        N = float(billed)
        U = min(max(100.0 * (lam * T if u_def == "offered" else served) / (C * T), 0.0), 100.0)
        M = K["memory_pct"]
        comps = {"L": L, "D": D, "N": N, "U": U, "M": M, "p95_all_s": p95_s,
                 "r_L": K["c_load"] * L, "r_D": -K["c_latency"] * D, "r_N": -K["c_pods"] * N,
                 "r_U": -K["c_cpu"] * U, "r_M": -K["c_mem"] * M}
        return comps["r_L"] + comps["r_D"] + comps["r_N"] + comps["r_U"] + comps["r_M"], comps

    def _substeps(self, lam, C, mu_pod, slo):
        P = self.P
        dt = P.dt_s
        a = lam * dt
        budget0 = C * dt
        q = self.q
        served = failed = viol = 0.0
        w, ages, rates = [], [], []
        t0 = self.t * P.control_step_s
        rho_sum = 0.0
        self._mix = None
        for j in range(P.n_sub):
            now = t0 + j * dt
            while q and now - q[0][0] >= P.timeout_s - 1e-9:
                failed += q[0][1]
                self.backlog -= q[0][1]
                q.popleft()
            if a > 0.0:
                q.append([now, a])
                self.backlog += a
            budget = budget0
            sub_served = []
            while q and budget > 0.0:
                stamp, amt = q[0]
                x = amt if amt <= budget else budget
                budget -= x
                sub_served.append((now - stamp, x))
                if x >= amt:
                    q.popleft()
                else:
                    q[0][1] = amt - x
            done = budget0 - budget
            self.backlog -= done
            if not q:
                self.backlog = 0.0  # drop float drift; the queue is the ground truth
            served += done
            rho = done / budget0
            rho_sum += rho
            rate = mu_pod * (1.0 - rho)
            for age, x in sub_served:
                slack = slo - age
                viol += x if slack <= 0.0 else x * math.exp(-rate * slack)
                w.append(x)
                ages.append(age)
                rates.append(mu_pod * max(1.0 - rho, 1e-3))
        self._mix = (w, ages, rates)
        return served, failed, viol, rho_sum / P.n_sub, _mixture_p95(w, ages, rates)

    def _obs(self, lam, served_rate, failed_rate, p95_ms, ready, pending, C, at_reset=False):
        P = self.P
        rng = self.rng
        cpu_raw = 100.0 * served_rate / C + P.cpu_offset_pct + rng.normal(0.0, P.cpu_noise_pct)
        cpu = min(max(cpu_raw, 0.0), 100.0)
        lat = p95_ms * math.exp(rng.normal(0.0, P.latency_noise_log_sd))
        o = [served_rate, cpu, lat, failed_rate, ready, pending]
        if self.oracle_obs:
            o += [lam, self.backlog]
        if self.util_obs:   # same draw as cpu_pct, no clip; no extra random numbers, so the stream is unchanged
            o.append(float("nan") if at_reset else cpu_raw)
        return np.array(o, dtype=np.float64)

    def conservation_residual(self):
        t = self.tot
        return t["arrived"] - t["served"] - t["failed"] - self.backlog


def _mixture_p95(w, ages, rates, tail=0.05):
    """95th percentile (tail = 0.05; other tails for the DInos reward) of latency L = age + Exp(rate), mixture
    weighted by served volume."""
    if not w:
        return float("nan")
    w = np.asarray(w)
    ages = np.asarray(ages)
    rates = np.asarray(rates)
    tot = w.sum()
    if tot <= 0:
        return float("nan")

    wn = w / tot

    def tails(xs):  # P(L > x) for a vector of x, one numpy pass
        z = xs[:, None] - ages[None, :]
        return (wn * np.exp(-rates * np.maximum(z, 0.0))).sum(axis=1)

    ln = LN20 if tail == 0.05 else math.log(1.0 / tail)
    lo, hi = 0.0, float(ages.max() + ln * 4.0 / rates.min())
    for _ in range(3):  # 3 grid refinements of 64 points: relative precision ~ 64**-3
        xs = np.linspace(lo, hi, 65)
        above = np.nonzero(tails(xs) > tail)[0]
        i = int(above[-1]) if above.size else 0
        lo, hi = xs[i], xs[min(i + 1, 64)]
    return float(hi)


class ObsWindow:
    """Rolling window of the last `window` observations for LSTM agents (DInos used 10)."""

    def __init__(self, window=10, n_features=len(OBS_NAMES)):  # use len(env.obs_names)
        self.window = window
        self.buf = np.zeros((window, n_features))

    def reset(self, obs):
        self.buf[:] = obs
        return self.buf.copy()

    def push(self, obs):
        self.buf = np.roll(self.buf, -1, axis=0)
        self.buf[-1] = obs
        return self.buf.copy()


def run_actions(env, actions, seed=0):
    """Replay a fixed action sequence; returns (return, info_sums, residual_max)."""
    env.reset(seed)
    total = 0.0
    sums = {}
    res = 0.0
    for a in actions:
        _, r, term, _, info = env.step(a)
        total += r
        for k in ("reward_cost", "reward_sla", "reward_churn", "bad_requests", "failed", "served"):
            sums[k] = sums.get(k, 0.0) + info[k]
        res = max(res, abs(env.conservation_residual()))
        if term:
            break
    return total, sums, res


def resolution_convergence(params, trace, actions, dts=(1.0, 3.0, 5.0, 15.0), seed=0):
    """Replay the same actions on the same trace at several internal resolutions dt_s.

    Returns {dt: {"return", "bad_requests", "failed"}} and the max relative difference of each quantity against
    the finest dt. Used to decide whether 15 s buckets would suffice (card: 'queue resolution' open item)."""
    out = {}
    for dt in dts:
        env = SimV2(params.with_(dt_s=float(dt)), trace=trace)
        ret, sums, _ = run_actions(env, actions, seed)
        out[dt] = {"return": ret, "bad_requests": sums["bad_requests"], "failed": sums["failed"]}
    ref = out[min(dts)]
    rel = {k: max(abs(v[k] - ref[k]) / max(abs(ref[k]), 1e-12) for v in out.values())
           for k in ("return", "bad_requests", "failed")}
    return out, rel
