"""Cycle 9: adapter around the verbatim original env P/environment.py (env_orig.py), design .lab/tuning/cycles/c9_design.md.

The verbatim env (reward min(capacity, load) - 3*VMs, VMs 1..20, start 6, done at t >= 5000) is never edited. This
module executes the unchanged port source once more in a private namespace whose `np` is a proxy: every numpy
attribute is the real one, except `np.random`, which points to the RandomState of the adapter that is currently
calling. The env logic and its draw order (per tick: load normal, read normal, cpu uniform, mem uniform, latency
normal) are therefore exactly those of env_orig.py; with variant `orig` and RandomState(seed) the trajectory equals
the verbatim env under np.random.seed(seed) (tests/test_c9.py). The global numpy stream is never touched.

Variants (parameters of the verbatim env, no new physics):
  orig                the original env
  noise{K}            load noise sd K instead of 0.2 (e.g. noise5, noise15): the load() draw becomes normal(0, K),
                      same draw, so the trace is the orig one with the load noise scaled by K / 0.2
  spike300            load spike amplitude 300 instead of 150 (attribute load_spike_amplitude)
  multispike          4 spikes per episode (same shape and length 100, all metrics, as the original spike) at
                      positions drawn at every reset from a separate spike stream RandomState([*seed, 9]) (the noise
                      stream is unchanged, so traces stay paired with orig); stratified: spike k starts uniformly in
                      [1250k + 50, 1250k + 1100], so spikes never overlap. Implemented by pointing the verbatim
                      attribute spike_position at the active spike (or far away) before each tick.
  noise15_multispike  both
Episodes with eval seed s use RandomState(s) (+ spike stream [s, 9]), so train/val/test traces differ.

Cycle 10 (pod startup delay; every variant above stays exactly as it was): `<variant>_d<D>` (e.g. orig_d5,
spike300_d5, orig_d10; D = 0..100) adds a startup delay of D ticks to the same variant. Mechanics (around the
unchanged verbatim step(), which keeps num_vms = ACTIVE VMs, so capacity = 10 * ACTIVE * read as before):
  - pending pods: a list of remaining ticks. At the start of every step each counter is decremented; a pod whose
    counter reaches 0 becomes active (env.num_vms += 1) BEFORE that tick's capacity/reward is computed. Hence a pod
    requested by action 2 in step n serves from step n + D on (D = 0: in step n, exactly the verbatim env).
  - action 2: if active + pending < 20, a pod with counter D is queued (D = 0: the verbatim +1); the verbatim step
    receives action 1 (D > 0).
  - action 0: if a pod is pending, the most recently requested one is cancelled (verbatim step gets action 1); else
    the verbatim scale-down (active > 1). Bounds 1..20 hold on active + pending.
  - reward = min(capacity, load) - 3 * (active + pending): a starting pod costs. info vm_cost = -3 * (active + pending).
  - state = the 6 verbatim features + pending count (7th feature, the same for every controller); OBS_SCALE_DELAY
    divides it by 20. With D = 0 the pending count is always 0 and features 0-5, reward, info and the RNG draws are
    identical to the base variant (tests/test_c10.py parity).

step() is gym-like: (state, reward, done, info); state = [load, capacity, cpu, mem, latency, VMs] as the verbatim env,
info = {served: min(capacity, load), vm_cost: -3*VMs} (served + vm_cost == reward exactly).
"""
import copy
import os
import re

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np  # noqa: E402

from tuning.legacy import env_orig  # noqa: E402

VARIANTS = ("orig", "noise5", "noise15", "multispike", "spike300", "noise15_multispike")
COMP = ("served", "vm_cost")
N_SPIKES, SEG = 4, 1250
SPIKE_STREAM = 9
OFF = -10 ** 9          # spike_position far in the past: no tick lies in [OFF, OFF + 100)
# --obs-scale on reference values for [load, capacity, cpu, mem, latency, VMs] (design c9 section 2)
OBS_SCALE = np.array([100.0, 100.0, 60.0, 70.0, 2.4, 20.0])
# cycle 10 delay variants: + pending pods (bound 20)
OBS_SCALE_DELAY = np.append(OBS_SCALE, 20.0)
FEATURES = ("load", "capacity", "cpu", "mem", "latency", "pods")
FEATURES_DELAY = FEATURES + ("pending",)
MAX_DELAY = 100
# cycle 23: pending-pod maturity features (ready in 1..K ticks), each 0/1 -> divisor 1 (its bound)
MATURITY_K = 5
FEATURES_MATURITY = tuple(f"ready_in_{k}" for k in range(1, MATURITY_K + 1))
OBS_SCALE_MATURITY = np.ones(MATURITY_K)
DELAY_RE = re.compile(r"^(.+)_d(\d+)$")


class _NpProxy:
    """numpy with `random` replaced by the calling adapter's RandomState (set around every call)."""
    def __init__(self):
        self.random = None

    def __getattr__(self, name):
        return getattr(np, name)


class _Rng:
    """RandomState wrapper; while `load_sd` is set (inside load()), normal() uses it as the scale."""
    def __init__(self, rs):
        self.rs, self.load_sd = rs, None

    def normal(self, loc, scale):
        return self.rs.normal(loc, scale if self.load_sd is None else self.load_sd)

    def uniform(self, lo, hi):
        return self.rs.uniform(lo, hi)


_NP = _NpProxy()
_NS = {"__name__": "tuning.legacy._orig_private", "__file__": env_orig.__file__}
with open(env_orig.__file__) as _f:
    exec(compile(_f.read(), env_orig.__file__, "exec"), _NS)
_NS["np"] = _NP
_Env = _NS["Environment"]


def split_delay(variant):
    """'<base>_d<D>' -> (base, D); a plain variant -> (variant, None). No validation of base."""
    m = DELAY_RE.match(str(variant))
    return (m.group(1), int(m.group(2))) if m else (variant, None)


def startup_delay(variant):
    """Cycle 10: startup delay D of a '<base>_d<D>' variant, None for the plain (cycle-9) variants."""
    return split_delay(variant)[1]


def _check_maturity(variant, maturity):
    if maturity and startup_delay(variant) is None:
        raise SystemExit(f"--pending-maturity on needs a startup-delay variant '<variant>_d<D>', got {variant!r}")
    return bool(maturity)


def state_size(variant, maturity=False):
    k = 6 if startup_delay(variant) is None else 7
    return k + MATURITY_K if _check_maturity(variant, maturity) else k


def obs_scale(variant, maturity=False):
    d = OBS_SCALE if startup_delay(variant) is None else OBS_SCALE_DELAY
    return np.append(d, OBS_SCALE_MATURITY) if _check_maturity(variant, maturity) else d


def features(variant, maturity=False):
    f = FEATURES if startup_delay(variant) is None else FEATURES_DELAY
    return f + FEATURES_MATURITY if _check_maturity(variant, maturity) else f


def maturity_counts(pending, k=MATURITY_K):
    """cycle 23: [#pending pods with remaining counter j for j = 1..k] (pods active j ticks after the observation)."""
    out = np.zeros(k)
    for c in pending:
        if 1 <= c <= k:
            out[c - 1] += 1.0
    return out


def check_variant(variant):
    """argparse type: a cycle-9 variant or '<variant>_d<D>'."""
    parse(variant)
    return variant


def parse(variant):
    """-> (load noise sd or None, multispike, load spike amplitude or None). Accepts '<variant>_d<D>' (cycle 10)."""
    base, d = split_delay(variant)
    if base not in VARIANTS or (d is not None and not 0 <= d <= MAX_DELAY):
        raise SystemExit(f"--orig-variant must be one of {VARIANTS}, optionally with a startup delay suffix _d<D> "
                         f"(D = 0..{MAX_DELAY}), got {variant!r}")
    variant = base
    m = re.match(r"noise(\d+)", variant)
    return (float(m.group(1)) if m else None), variant.endswith("multispike"), (300 if variant == "spike300" else None)


def _seed_list(seed):
    return list(seed) if isinstance(seed, (list, tuple)) else [int(seed)]


class OrigEnv:
    def __init__(self, variant="orig", seed=0, maturity=False):
        self.variant = variant
        self.maturity = _check_maturity(variant, maturity)     # cycle 23 (off: the cycle-10 state, untouched)
        self.load_sd, self.multi, amp = parse(variant)
        self.seed = seed
        self.delay = startup_delay(variant)       # cycle 10; None = the cycle-9 env, untouched
        self.pending = []
        self.rng = _Rng(np.random.RandomState(seed))
        self.spike_rng = np.random.RandomState(_seed_list(seed) + [SPIKE_STREAM]) if self.multi else None
        self.spikes = None
        self.env = self._call(_Env)
        if amp is not None:
            self.env.load_spike_amplitude = amp
        self._install_load()

    def _install_load(self):
        """noise{K}: the verbatim load() runs unchanged, but its normal draw uses sd K (instance attribute; the verbatim
        step() calls self.load(t))."""
        self.env.__dict__.pop("load", None)
        if self.load_sd is None:
            return
        base_load, rng, sd = self.env.load, self.rng, self.load_sd

        def load(t):
            rng.load_sd = sd
            try:
                return base_load(t)
            finally:
                rng.load_sd = None
        self.env.load = load

    def _call(self, f, *a):
        prev = _NP.random
        _NP.random = self.rng
        try:
            return f(*a)
        finally:
            _NP.random = prev

    def _aim(self, t):
        """multispike: point the verbatim spike_position at the spike active at tick t (else OFF)."""
        if self.multi:
            L = self.env.spike_length
            self.env.spike_position = next((s for s in self.spikes if s <= t < s + L), OFF)

    def reset(self):
        if self.multi:
            self.spikes = [int(SEG * k + self.spike_rng.randint(50, SEG - 150 + 1)) for k in range(N_SPIKES)]
            self._aim(0)
        s0 = self._call(self.env.reset)
        if self.delay is None:
            return s0
        self.pending = []
        if self.maturity:
            return np.concatenate([s0, [0.0], np.zeros(MATURITY_K)])
        return np.append(s0, 0.0)

    def step(self, action):
        if self.delay is not None:
            return self._step_delay(action)
        self._aim(self.env.time + 1)
        state, reward, done = self._call(self.env.step, action)
        return state, reward, done, {"served": min(state[1], state[0]), "vm_cost": -3 * state[5]}

    def _delay_action(self, action):
        """Cycle 10: promote due pending pods, queue/cancel pending pods; -> the action for the verbatim step()."""
        if self.delay == 0:
            return action                     # pending stays empty: the verbatim env exactly
        left = [c - 1 for c in self.pending]
        self.env.num_vms += sum(1 for c in left if c <= 0)
        self.pending = [c for c in left if c > 0]
        if action == 2:
            if self.env.num_vms + len(self.pending) < 20:
                self.pending.append(self.delay)
            return 1
        if action == 0 and self.pending:
            self.pending.pop()                # cancel the most recently requested pod
            return 1
        return action

    def _step_delay(self, action):
        a = self._delay_action(action)
        self._aim(self.env.time + 1)
        state, reward, done = self._call(self.env.step, a)
        p = len(self.pending)
        served = min(state[1], state[0])
        if p:
            reward = served - 3 * (state[5] + p)
        if self.maturity:
            return np.concatenate([state, [float(p)], maturity_counts(self.pending)]), reward, done, \
                {"served": served, "vm_cost": -3 * (state[5] + p)}
        return np.append(state, float(p)), reward, done, {"served": served, "vm_cost": -3 * (state[5] + p)}

    def clone(self):
        """Independent copy (RNG states included): the clairvoyant oracle peeks at the next tick with it."""
        c = copy.copy(self)
        c.rng = _Rng(copy.deepcopy(self.rng.rs))
        c.spike_rng = copy.deepcopy(self.spike_rng)
        c.spikes = list(self.spikes) if self.spikes is not None else None
        c.pending = list(self.pending)
        c.env = copy.copy(self.env)
        c._install_load()
        return c

    @property
    def num_vms(self):
        return self.env.num_vms

    @property
    def n_pending(self):
        return len(self.pending)

    @property
    def time(self):
        return self.env.time


def make(variant, seed, maturity=False):
    return OrigEnv(variant, seed, maturity) if maturity else OrigEnv(variant, seed)
