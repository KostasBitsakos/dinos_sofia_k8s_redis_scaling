"""Baseline policies for Sim v2. Each policy sees only the observation (env.OBS_NAMES), never the trace.

Actions: 0 = down, 1 = no-op, 2 = up. `oracle_bound` (re-exported from oracle.py) is not a policy: it is a
certified upper bound on the return of any policy on a given trace; `OracleReplay` replays its maximising action
sequence in the real dynamics (an achievable clairvoyant open-loop return, <= the bound).
"""
import itertools
import math

import numpy as np

from env import OBS_NAMES, SimV2
from oracle import oracle_bound  # noqa: F401

READY, PENDING, CPU = 4, 5, 1


class AlwaysMin:
    name = "always_min"

    def reset(self):
        pass

    def act(self, obs):
        return 0


class AlwaysMax:
    name = "always_max"

    def reset(self):
        pass

    def act(self, obs):
        return 2


class StaticRamp:
    """static-ramp(c): starts at params.init_pods, moves one pod per step to c and stays there
    (pays the ramp's churn, rebalance and startup costs). Not a fixed allocation: see Fixed."""

    def __init__(self, c):
        self.c = int(c)
        self.name = f"static_ramp_{self.c}"

    def reset(self):
        pass

    def act(self, obs):
        total = int(round(obs[READY] + obs[PENDING]))
        return 2 if total < self.c else (0 if total > self.c else 1)


class Fixed:
    """fixed(c): the episode starts with c ready pods and never scales (no ramp cost)."""

    def __init__(self, c):
        self.c = int(c)
        self.init_pods = self.c
        self.name = f"fixed_{self.c}"

    def reset(self):
        pass

    def act(self, obs):
        return 1


class Threshold:
    """HPA-like: observed CPU % > up -> scale up, < down -> scale down, at most one change per `cooldown`+1 steps."""

    def __init__(self, up=80.0, down=40.0, cooldown=2):
        self.up, self.down, self.cooldown = float(up), float(down), int(cooldown)
        self.name = f"threshold_{self.up:g}_{self.down:g}_{self.cooldown}"

    def reset(self):
        self.wait = 0

    def act(self, obs):
        if self.wait > 0:
            self.wait -= 1
            return 1
        a = 2 if obs[CPU] > self.up else (0 if obs[CPU] < self.down else 1)
        if a != 1:
            self.wait = self.cooldown
        return a


# ---------------------------------------------------------------- Kubernetes HPA (autoscaling/v2, CPU utilization)
# Reverse-engineered from kubernetes v1.34.0 pkg/controller/podautoscaler, as written up in qrl-lab
# docs/hpa_spec.md (cited below as "spec §n"). Simulator mapping and remaining gaps: sim_v2/README.md "HPA baseline".
HPA_TOLERANCE = 0.1            # spec §1: --horizontal-pod-autoscaler-tolerance default 0.1, both directions
HPA_DOWN_WINDOW_STEPS = 20     # spec §3: scaleDown stabilizationWindowSeconds 300 s / 15 s sync (spec §6)
HPA_UP_POLICIES = (("Percent", 100, 1), ("Pods", 4, 1))   # spec §3 default scaleUp: 100 % / 15 s and 4 pods / 15 s
HPA_DOWN_POLICIES = (("Percent", 100, 1),)                 # spec §3 default scaleDown: 100 % / 15 s
HPA_TARGET_GRID = tuple(range(10, 100, 5))                 # target utilization % candidates (only tuned value)


def _hpa_within(ratio, tol_up, tol_down):
    """spec §1 Tolerances.isWithin: (1 - tolDown) <= ratio <= (1 + tolUp)."""
    return (1.0 - tol_down) <= ratio <= (1.0 + tol_up)


def hpa_replica_calc(current, n_ready, n_unready, n_missing, util_pct, target_pct, tol_up=HPA_TOLERANCE,
                     tol_down=HPA_TOLERANCE):
    """ReplicaCalculator.GetResourceReplicas for CPU utilization, every pod with the same request (spec §2, §5).

    current: spec replicas; n_ready: pods with usable metrics (set M); n_unready: unready pods (set U, metrics
    removed); n_missing: pods without a metric entry; util_pct: mean CPU usage / request in % over the n_ready pods
    (raw, before the controller's integer truncation). Returns the raw desired replicas, or None for the controller's
    error path (no usable metrics -> no scaling, spec §2 "if none remain: error")."""
    if n_ready <= 0 or util_pct is None or not math.isfinite(util_pct):
        return None
    pct = max(float(util_pct), 0.0)                 # metrics-server usage is non-negative
    util = int(pct)                                 # spec §1: int32(metricsTotal*100 / requestsTotal), truncation
    ratio = util / float(target_pct)                # spec §1: ratio = currentUtilization / targetUtilization
    scale_up_unready = n_unready > 0 and ratio > 1.0                     # spec §2 scaleUpWithUnready
    if not scale_up_unready and n_missing == 0:     # spec §2 plain path: tolerance, then ceil(ratio * |M|)
        if _hpa_within(ratio, tol_up, tol_down):
            return current
        return int(math.ceil(ratio * n_ready))
    # spec §2 rebalance path: missing pods get max(100, target) % when ratio < 1, 0 % when ratio > 1 (none if == 1);
    # with scaleUpWithUnready the unready pods get 0 %. Recompute over all entries.
    total, n = pct * n_ready, n_ready
    if n_missing > 0 and ratio < 1.0:
        total += n_missing * max(100.0, float(target_pct))
        n += n_missing
    elif n_missing > 0 and ratio > 1.0:
        n += n_missing
    if scale_up_unready:
        n += n_unready
    new_ratio = int(total / n) / float(target_pct)
    # spec §2: keep current if within tolerance or the direction flipped
    if _hpa_within(new_ratio, tol_up, tol_down) or (ratio < 1.0 and new_ratio > 1.0) or (ratio > 1.0 and new_ratio < 1.0):
        return current
    new = int(math.ceil(new_ratio * n))             # spec §2: ceil(newRatio * len(metrics))
    if (new_ratio < 1.0 and new > current) or (new_ratio > 1.0 and new < current):   # spec §2 wrong-direction guard
        return current
    return new


class K8sHPA:
    """Kubernetes HPA (autoscaling/v2, one CPU-utilization metric, default behavior), one sync per control step.

    Needs SimV2(..., **K8sHPA.env_kwargs): `act` returns an ABSOLUTE desired replica count (multi-pod changes, unlike
    the +-1 agents) and reads the unclipped `cpu_util_pct` observation (usage / request with request = per-pod
    capacity). Unready pods = pending (warming-up) pods; ready pods all have metrics (no missing metrics in Sim v2).
    Time is in syncs: a stored item at sync s is inside a window of W syncs at sync t iff s > t - W (the code's
    `timestamp.After(now - window)` with exact 15 s syncs). Only `target_pct` is tuned."""
    env_kwargs = {"replica_actions": True, "util_obs": True}

    def __init__(self, target_pct, min_replicas=1, max_replicas=30, tolerance=HPA_TOLERANCE, up_window_steps=0,
                 down_window_steps=HPA_DOWN_WINDOW_STEPS, up_policies=HPA_UP_POLICIES, down_policies=HPA_DOWN_POLICIES,
                 select_up="Max", select_down="Max", util_index=len(OBS_NAMES)):
        self.target = int(target_pct)               # spec §1: target utilization is an integer percentage
        if self.target <= 0:
            raise ValueError("target utilization must be a positive integer percentage")
        self.min, self.max = int(min_replicas), int(max_replicas)   # spec §3: minReplicas default 1; task: max 30
        self.tol = float(tolerance)
        self.up_w, self.down_w = int(up_window_steps), int(down_window_steps)
        self.up_pol, self.down_pol = tuple(up_policies), tuple(down_policies)
        self.select_up, self.select_down = select_up, select_down
        self.util_index = int(util_index)
        self.name = f"k8s_hpa_{self.target}"

    def reset(self):
        self.t = 0
        self.recs = None        # [(sync, raw recommendation)]
        self.up_events = []     # [(sync, pods added)]
        self.down_events = []   # [(sync, pods removed)]

    # spec §3 rate limits: events of the last `period` syncs
    def _changed(self, events, period, now):
        return sum(n for s, n in events if s > now - period)

    def _scale_up_limit(self, cur, now):
        """spec §3 calculateScaleUpLimitWithScalingRules."""
        if self.select_up == "Disabled":
            return cur
        out = None
        for kind, value, period in self.up_pol:
            start = cur - self._changed(self.up_events, period, now) + self._changed(self.down_events, period, now)
            prop = start + value if kind == "Pods" else int(math.ceil(start * (1.0 + value / 100.0)))
            out = prop if out is None else (max(out, prop) if self.select_up == "Max" else min(out, prop))
        return out

    def _scale_down_limit(self, cur, now):
        """spec §3: mirror of the scale-up limit; Percent truncates; selectPolicy Max = the largest change (min)."""
        if self.select_down == "Disabled":
            return cur
        out = None
        for kind, value, period in self.down_pol:
            start = cur - self._changed(self.up_events, period, now) + self._changed(self.down_events, period, now)
            prop = start - value if kind == "Pods" else int(start * (1.0 - value / 100.0))
            out = prop if out is None else (min(out, prop) if self.select_down == "Max" else max(out, prop))
        return out

    def _normalize(self, cur, desired, now):
        """spec §3/§4 normalizeDesiredReplicasWithBehaviors: stabilize on the stored raw recommendations, store the
        raw value, then rate-limit (bounded by max/min replicas)."""
        up = down = desired
        for s, rec in self.recs:
            if s > now - self.up_w:
                up = min(up, rec)
            if s > now - self.down_w:
                down = max(down, rec)
        out = cur                                   # spec §3 code order: start from current
        if out < up:
            out = up
        if out > down:
            out = down
        self.recs = [(s, r) for s, r in self.recs if s > now - max(self.up_w, self.down_w)]   # drop expired samples
        self.recs.append((now, desired))            # raw (unstabilized) value stored after the scan
        if out > cur:                               # spec §3 convertDesiredReplicasWithBehaviorRate
            lim = max(self._scale_up_limit(cur, now), cur)
            out = min(out, min(self.max, lim))
        elif out < cur:
            lim = min(self._scale_down_limit(cur, now), cur)
            out = max(out, max(self.min, lim))
        return out

    def act(self, obs):
        now = self.t
        self.t += 1
        ready, pending = int(round(obs[READY])), int(round(obs[PENDING]))
        cur = ready + pending                       # spec §4: scale.Spec.Replicas (ready + warming-up pods)
        if self.recs is None:                       # spec §3/§4 recordInitialRecommendation (seeded with current)
            self.recs = [(now, cur)]
        if cur > self.max:                          # spec §4 step 2
            desired = self.max
        elif cur < self.min:
            desired = self.min
        else:
            raw = hpa_replica_calc(cur, ready, pending, 0, float(obs[self.util_index]), self.target, self.tol, self.tol)
            if raw is None:                         # spec §2 error path: no metrics, no scaling, nothing stored
                return cur
            desired = self._normalize(cur, raw, now)
        if desired > cur:                           # spec §4 step 4 storeScaleEvent
            self.up_events.append((now, desired - cur))
        elif desired < cur:
            self.down_events.append((now, cur - desired))
        return desired


def tune_hpa(params, workload, tune_seeds, grid=HPA_TARGET_GRID):
    """Select the HPA target utilization on tune seeds only (mean return; ties -> smallest target).
    Returns (best K8sHPA, [(target, mean return)])."""
    from workloads import check_seed
    for s in tune_seeds:
        check_seed("tune", s)
    table = []
    for tgt in grid:
        env = SimV2(params, workload=workload, **K8sHPA.env_kwargs)
        table.append((int(tgt), float(np.mean([run_episode(env, K8sHPA(tgt), s)["ret"] for s in tune_seeds]))))
    best = max(table, key=lambda x: (x[1], -x[0]))[0]
    return K8sHPA(best), table


class OracleReplay:
    """Clairvoyant open-loop: replays the oracle DP's action sequence (needs the trace; not deployable)."""
    name = "oracle_replay"

    def __init__(self, actions):
        self.actions = list(actions)

    def reset(self):
        self.i = 0

    def act(self, obs):
        a = self.actions[self.i]
        self.i += 1
        return a


def run_episode(env, policy, seed):
    """One episode. Returns dict with return, reward components, accounting and the conservation residual."""
    obs, _ = env.reset(seed, init_pods=getattr(policy, "init_pods", None))
    policy.reset()
    out = dict(ret=0.0, cost=0.0, sla=0.0, churn=0.0, bad=0.0, failed=0.0, served=0.0, arrived=0.0,
               settled=0.0, changes=0.0, pod_steps=0.0, max_resid=0.0, p95_sum=0.0, steps=0)
    while True:
        obs, r, term, trunc, info = env.step(policy.act(obs))
        out["ret"] += r
        out["cost"] += info["reward_cost"]
        out["sla"] += info["reward_sla"]
        out["churn"] += info["reward_churn"]
        out["bad"] += info["bad_requests"]
        out["failed"] += info["failed"]
        out["served"] += info["served"]
        out["arrived"] += info["arrived"]
        out["settled"] += info["settled"]
        out["changes"] += info["accepted_change"]
        out["pod_steps"] += info["ready"] + info["pending"]
        out["p95_sum"] += info["p95_ms"] if info["p95_ms"] == info["p95_ms"] else 0.0
        out["max_resid"] = max(out["max_resid"], abs(env.conservation_residual()))
        out["steps"] += 1
        if term or trunc:
            break
    out["outstanding_end"] = env.outstanding
    out["final_resid"] = out["arrived"] - out["served"] - out["failed"]
    return out


THRESHOLD_GRID = {"up": (60, 70, 80, 90), "down": (20, 30, 40, 50), "cooldown": (0, 1, 2, 4)}


def tune_threshold(params, workload, tune_seeds, grid=None):
    """Grid-search Threshold on tune seeds only (mean return). Returns (best Threshold, table)."""
    from workloads import check_seed
    for s in tune_seeds:
        check_seed("tune", s)
    grid = grid or THRESHOLD_GRID
    table = []
    for up, down, cd in itertools.product(grid["up"], grid["down"], grid["cooldown"]):
        if down >= up:
            continue
        pol = Threshold(up, down, cd)
        env = SimV2(params, workload=workload)
        rets = [run_episode(env, pol, s)["ret"] for s in tune_seeds]
        table.append(((up, down, cd), float(np.mean(rets))))
    best = max(table, key=lambda x: x[1])[0]
    return Threshold(*best), table


def select_static(params, workload, tune_seeds, family="fixed", candidates=None):
    """Best c of a static family ('fixed' or 'ramp') by mean return on tune seeds."""
    from workloads import check_seed
    for s in tune_seeds:
        check_seed("tune", s)
    cls = {"fixed": Fixed, "ramp": StaticRamp}[family]
    candidates = candidates or range(params.min_pods, params.max_pods + 1)
    scores = []
    for c in candidates:
        env = SimV2(params, workload=workload)
        scores.append((c, float(np.mean([run_episode(env, cls(c), s)["ret"] for s in tune_seeds]))))
    return cls(max(scores, key=lambda x: x[1])[0]), scores
