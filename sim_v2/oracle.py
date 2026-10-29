"""Certified upper bound on the Sim v2 return for a known arrival trace (perfect lambda).

Relaxation (why it is an upper bound). In SimV2, over a whole episode,
    bad = sum over sub-steps j of [ (a_j - f_j) + f_j * v(rho_j) ],
where a_j = arrivals, f_j = fresh work served in its own sub-step, rho_j >= f_j / cap_j the sub-step utilisation
and v(rho) = exp(-m (1 - rho)), m = mu_pod * SLO. Every unit carried past its sub-step is bad exactly once
(served late with wait >= dt_s >= SLO, timed out, or settled at the end). Two relaxations make this smaller:
  (1) the backlog is discarded for free once charged, so it never steals capacity (backlog dimension -> 0;
      the most optimistic backlog discretisation possible), and
  (2) the server may refuse fresh work when that is cheaper: f is chosen freely in [0, min(a, cap)].
Since v is increasing, (a - f) + f v(rho) >= min_f [(a - f) + f v(f/cap)] =: h(a, cap), pathwise, for every
action sequence. Billing and churn depend only on pods and are exact. Hence
    true return(actions) <= relaxed return(actions) <= max_actions relaxed return = DP value,
for every policy, adaptive or not, on that trace. h is degree-1 homogeneous, so the bound does not depend on
dt_s; it requires dt_s >= SLO (checked). The DP is exact over (ready, pending pipeline, rebalance counter).
"""
import numpy as np

from env import n_actions, pod_transition


def _xstar(m):
    """Root of exp(-m (1 - x)) (1 + m x) = 1 in (0, 1): argmin of g(x) = -x + x exp(-m (1 - x))."""
    m = np.asarray(m, dtype=float)
    lo, hi = np.zeros_like(m), np.ones_like(m)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pos = np.exp(-m * (1 - mid)) * (1 + m * mid) > 1.0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return lo


def bad_rate_lower_bound(lam, C, m):
    """h(lambda, C) per second: min over fresh service f <= min(lambda, C) of (lambda - f) + f exp(-m (1 - f/C))."""
    f = np.minimum(lam, _xstar(m) * C)
    return lam - f + f * np.exp(-m * (1.0 - f / C))


class _Model:
    def __init__(self, P):
        self.P = P
        d, md = P.startup_delay_steps, P.max_delta
        A = n_actions(P)
        states, index = [], {}
        start = (P.init_pods, tuple([0] * d), 0)
        stack = [start]
        index[start] = 0
        states.append(start)
        trans = []
        while stack:  # enumerate reachable states
            s = stack.pop()
            for a in range(A):
                r, p, act, dp, bill, rbn = pod_transition(P, s[0], s[1], s[2], a)
                ns = (r, p, rbn)
                if ns not in index:
                    index[ns] = len(states)
                    states.append(ns)
                    stack.append(ns)
                trans.append((index[s], a, index[ns], r, int(act), dp, bill))
        S = len(states)
        self.S, self.A = S, A
        self.nxt = np.zeros((S, A), dtype=np.int64)
        self.sr = np.zeros((S, A), dtype=np.int64)
        self.act = np.zeros((S, A), dtype=np.int64)
        self.det = np.zeros((S, A))  # deterministic per-step reward (cost + churn), negative
        for si, a, ni, r, act, dp, bill in trans:
            self.nxt[si, a] = ni
            self.sr[si, a] = r
            self.act[si, a] = act
            self.det[si, a] = -(P.alpha * bill * P.control_step_s / P.cost_norm + P.gamma * dp)
        self.s0 = 0
        self.index = index
        rr = np.arange(P.max_pods + 1, dtype=float)
        rr[0] = 1.0
        self.C = np.stack([P.mu_ops_per_pod * rr ** P.theta,
                           P.mu_ops_per_pod * rr ** P.theta * (1 - P.rebalance_loss)], axis=1)  # [r, active]
        self.m = self.C / rr[:, None] * P.slo_eff_s


def _solve(P, trace, keep_policy):
    if P.dt_s < P.slo_eff_s:
        raise ValueError("oracle bound needs dt_s >= SLO (late-served work must count as bad)")
    M = _Model(P)
    trace = np.asarray(trace, dtype=float)
    T = len(trace)
    k = P.beta * P.control_step_s / P.sla_norm
    V = np.zeros(M.S)
    pol = np.zeros((T, M.S), dtype=np.int8) if keep_policy else None
    xs = _xstar(M.m)
    for t in range(T - 1, -1, -1):
        lam = trace[t]
        f = np.minimum(lam, xs * M.C)
        B = lam - f + f * np.exp(-M.m * (1.0 - f / M.C))  # [r, active]
        Q = M.det - k * B[M.sr, M.act] + V[M.nxt]
        if keep_policy:
            a = Q.argmax(axis=1)
            pol[t] = a
            V = Q[np.arange(M.S), a]
        else:
            V = Q.max(axis=1)
    return M, V, pol


def _start_value(P, M, V, init_pods):
    key = (int(init_pods), tuple([0] * P.startup_delay_steps), 0)
    if key not in M.index:
        raise ValueError(f"start state {key} not reachable")
    return float(V[M.index[key]])


def oracle_bound(params, trace, return_actions=True, init_pods=None):
    """DP value (certified upper bound on any policy's return on `trace`, episode starting with `init_pods`
    ready pods, default params.init_pods) and the maximising action sequence from that start."""
    P = params
    M, V, pol = _solve(P, trace, return_actions)
    s = M.s0 if init_pods is None else M.index[(int(init_pods), tuple([0] * P.startup_delay_steps), 0)]
    bound = float(V[s])
    if not return_actions:
        return bound, None
    acts = []
    for t in range(len(trace)):
        a = int(pol[t, s])
        acts.append(a)
        s = M.nxt[s, a]
    return bound, acts


def oracle_solve(params, trace):
    """One DP pass: (bound from params.init_pods, its action sequence, {c: bound when starting at c ready pods})."""
    P = params
    M, V, pol = _solve(P, trace, True)
    by_start = {c: _start_value(P, M, V, c) for c in range(P.min_pods, P.max_pods + 1)}
    s, acts = M.s0, []
    for t in range(len(trace)):
        a = int(pol[t, s])
        acts.append(a)
        s = M.nxt[s, a]
    return float(V[M.s0]), acts, by_start
