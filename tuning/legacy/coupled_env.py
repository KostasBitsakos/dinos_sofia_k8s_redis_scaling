"""Cycle 4 (Track L): synthetic load-sharing coupling over the VERBATIM legacy envs (design: .lab/tuning/cycles/
c3_codex.md, c3_synthesis.md). The ported env files are not edited; this module wraps them.

Per tick, after the action has set the pod count n_t (legacy rule: +-1, bounds 1..20, start 6), the wrapped env's
step() runs unchanged, so it consumes exactly the same random draws as the legacy env. From its returned observation
we take the legacy draws L_t (load), c0 (cpu %), m0 (mem %), l0 (latency) and apply

    g_t = 6 * L_t / (70 * n_t)
    c_t = min(100, g_t * c0)      m_t = min(100, g_t * m0)      l_t = g_t * l0
    R_t = 3*L_t - 5*l_t - 10*n_t - 0.5*c_t - 0.5*m_t

The observation keeps the legacy layout [L, capacity, c, m, l, n] with c/m/l replaced by the coupled values (agents
see what the env reports); capacity (state[1]) and L are the legacy values. step() returns the legacy 8-tuple
(state, reward, 3L, -5l, -10n, -0.5c, -0.5m, done); reset() returns the coupled initial observation (n = 6).

Constants: 6 = the legacy initial pod count, 70 = mean of the legacy load sinusoid (amplitude 50 + offset 20),
100 = the percentage bound. At L_t = 70 and n_t = 6, g = 1 and every metric and the reward equal the legacy values.

ASSUMPTIONS (synthetic, a mechanism experiment, NOT a calibrated simulator):
  * load sharing: per-pod utilisation scales with load per pod, L_t / n_t;
  * inverse sharing (utilisation proportional to 1/n) and LINEAR latency in load per pod are NEW assumptions that the
    legacy simulation does not contain (there cpu/mem/latency are independent of pods and load);
  * latency is not clipped; cpu/mem are clipped at 100 (no queueing, no dropped requests, no saturation penalty);
  * the reward law and its weights are unchanged (3L is action-independent, as in the legacy env).
"""
import numpy as np

REF_PODS = 6
REF_LOAD = 70.0
PCT_MAX = 100.0


def coupling(L, n, c0, m0, l0):
    """Return (g, c, m, l) for load L, pods n and the legacy draws c0, m0, l0."""
    g = REF_PODS * L / (REF_LOAD * n)
    return g, min(PCT_MAX, g * c0), min(PCT_MAX, g * m0), g * l0


class CoupledEnv:
    """Wrap one instance of a verbatim legacy env class (8-tuple step()). Unknown attributes (time, num_vms, ...)
    are delegated to the wrapped env."""

    variant = "coupled"

    def __init__(self, base_cls):
        self.__dict__["base"] = base_cls()
        self.__dict__["last_g"] = 1.0

    def __getattr__(self, name):
        return getattr(self.__dict__["base"], name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.__dict__["base"], name, value)

    def _couple(self, s):
        L, n = float(s[0]), float(s[5])
        g, c, m, l = coupling(L, n, float(s[2]), float(s[3]), float(s[4]))
        self.__dict__["last_g"] = g
        return np.array([s[0], s[1], c, m, l, s[5]]), L, n, c, m, l

    def step(self, action):
        out = self.base.step(action)
        state, L, n, c, m, l = self._couple(out[0])
        load_reward = 3.0 * L
        latency_penalty = -5.0 * l
        vm_penalty = -10.0 * n
        cpu_penalty = -0.5 * c
        memory_penalty = -0.5 * m
        reward = load_reward + latency_penalty + vm_penalty + cpu_penalty + memory_penalty
        return state, reward, load_reward, latency_penalty, vm_penalty, cpu_penalty, memory_penalty, out[7]

    def reset(self):
        return self._couple(self.base.reset())[0]


def make(base_cls):
    """Factory with the same call signature as a legacy env class (Env())."""
    def f():
        return CoupledEnv(base_cls)
    f.__name__ = f"Coupled{base_cls.__module__.rsplit('.', 1)[-1]}"
    return f
