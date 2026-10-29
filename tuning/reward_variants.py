"""Reward variants for quick_test (config key "reward_variant": "v2" default | "d_sec"). No protected file is edited.

v2    : the adapter reward unchanged (r = 3L - 5D - 10N - 0.5U - 0.5M, D = mean latency / SLO, SLO = sim slo_s).
d_sec : identical except D = mean latency in SECONDS / 1.0 s (requests that failed are still counted at the 60 s
        timeout inside that mean, exactly as the env computes it: info["dinos"]["p95_all_s"] is the per-step mean
        latency in seconds under the default d_def='mean'). Same coefficients (3/5/10/0.5/0.5). Only the D term is
        replaced: r_dsec = r_v2 - r_D_v2 - c_latency * latency_s.
Cycle 2: config key "load_def" "offered" (default, unchanged) | "served" (see RewardWrapper); composes with both variants.
The wrapper rewrites the reward returned to the agent and info["dinos"] (keeps info["dinos"]["D_v2"] for audit).
"""
VARIANTS = ("v2", "d_sec")
D_UNIT_S = 1.0


LOAD_DEFS = ("offered", "served")


class RewardWrapper:
    """Rewrites the reward of the sim. variant "d_sec" replaces the D term (see above). load_def "served" replaces the
    load credit L = 10 lambda_offered / mu by L = 10 (served / T) / mu = 10 served / (15 mu) (served = requests served in
    the step, T = control step); only the r_L term changes (info["dinos"]["L_offered"] keeps the original L)."""

    def __init__(self, sim, variant="v2", load_def="offered"):
        self._sim = sim
        self.variant, self.load_def = variant, load_def
        K = sim.env.dinos
        self.c_lat = float(K["c_latency"])
        if variant == "d_sec" and K.get("d_def") != "mean":
            raise ValueError("d_sec needs the mean-latency D definition (variant V2)")

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def reset(self, *a, **k):
        return self._sim.reset(*a, **k)

    def step(self, action):
        win, r, done, info = self._sim.step(action)
        c = dict(info["dinos"])
        if self.variant == "d_sec":
            D = c["p95_all_s"] / D_UNIT_S
            r_D = -self.c_lat * D
            r = r - c["r_D"] + r_D
            c.update(D_v2=c["D"], D=D, r_D=r_D)
        if self.load_def == "served":
            K, P = self._sim.env.dinos, self._sim.env.P
            L = K["load_scale"] * (info["served"] / P.control_step_s) / P.mu_ops_per_pod
            r_L = K["c_load"] * L
            r = r - c["r_L"] + r_L
            c.update(L_offered=c["L"], L=L, r_L=r_L)
        return win, r, done, dict(info, dinos=c, reward_dinos=r)


def wrap(sim, variant="v2", load_def="offered"):
    if variant not in VARIANTS:
        raise ValueError(f"reward_variant must be one of {VARIANTS}, got {variant!r}")
    if load_def not in LOAD_DEFS:
        raise ValueError(f"load_def must be one of {LOAD_DEFS}, got {load_def!r}")
    if variant == "v2" and load_def == "offered":
        return sim
    return RewardWrapper(sim, variant, load_def)
