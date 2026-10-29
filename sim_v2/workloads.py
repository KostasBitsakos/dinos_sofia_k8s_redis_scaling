"""Sim v2 workloads (PLAN §8.2). Each workload maps (seed, steps) -> arrival rate per control step in ops/s.

The trace *is* the realised arrivals (any Poisson/MMPP randomness is drawn here, from the seed), so the oracle
sees exactly what the env serves. Seed ranges are disjoint: train / tune / held-out (use `check_seed`).
Scaling: `pods_needed(lambda) = lambda / (mu * RHO_SIZING)`; W0's peak needs about 25 pods at mu = 1000.
All shapes and numbers are provisional (not fitted to any trace).
"""
import csv
import math

import numpy as np

# "heldout" (10000-10999) is BURNED: sim_v2 unit tests, gate runs (run_gates, seeds 10000+) and early smokes have
# seen it. "test" (20000-20999) has never been used by any code, test, smoke or gate; final evaluation uses it only.
SEED_RANGES = {"train": range(0, 1000), "tune": range(1000, 2000), "heldout": range(10000, 11000),
               "test": range(20000, 21000)}
RHO_SIZING = 0.8      # utilisation used only to express "pods needed" when sizing the workloads
MU_REF = 1000.0       # ops/s per pod used for sizing (SimParams.mu_ops_per_pod default)
CONTROL_STEP_S = 15.0


def check_seed(split, seed):
    if seed not in SEED_RANGES[split]:
        raise ValueError(f"seed {seed} is not in the {split} range {SEED_RANGES[split]}")
    return seed


def _rng(tag, seed):
    return np.random.default_rng(np.random.SeedSequence([int(seed), sum(map(ord, tag)) * 7919]))


def pods_needed(lam, mu=MU_REF):
    return np.asarray(lam) / (mu * RHO_SIZING)


class W0:
    """Legacy-like: sine (period 1000/(2 pi) steps, as environment.py) + small noise + one oscillating spike.

    Legacy units are scaled by `scale` ops/s; legacy peak 50+50+20+80 = 200 units -> 20000 ops/s = 25 pods.
    Per seed: random phase and spike position (the legacy file fixed both)."""
    name = "W0"

    def __init__(self, scale=100.0, amp=50.0, offset=20.0, period=1000 / (2 * math.pi), noise=0.2,
                 spike_len=100, spike_amp=80.0):
        self.__dict__.update(scale=scale, amp=amp, offset=offset, period=period, noise=noise,
                             spike_len=spike_len, spike_amp=spike_amp)

    def trace(self, seed, steps):
        g = _rng(self.name, seed)
        t = np.arange(steps, dtype=float)
        phase = g.uniform(0, 2 * math.pi)
        lam = self.amp * np.sin(2 * math.pi * t / self.period + phase) + self.amp + self.offset
        lam += g.normal(0, self.noise, steps)
        if steps > self.spike_len + 2:
            p = int(g.integers(0, steps - self.spike_len))
            s = slice(p, p + self.spike_len)
            lam[s] += self.spike_amp * np.abs(np.sin(2 * math.pi * 10 * (t[s] - p) / self.spike_len))
        return np.maximum(lam, 0.0) * self.scale


class W1:
    """Non-homogeneous Poisson with a daily profile (two peaks) and 2-state MMPP bursts.

    One simulated day = `day_steps` control steps (5760 x 15 s = 24 h). Base rate between `low` and `high` ops/s,
    bursts multiply by `burst_mult`; normal->burst prob `p_on`, burst->normal `p_off` per step.
    Arrivals per step ~ Poisson(lambda * 15 s), returned as a rate."""
    name = "W1"

    def __init__(self, low=2000.0, high=12000.0, burst_mult=1.6, p_on=0.003, p_off=0.05, day_steps=5760):
        self.__dict__.update(low=low, high=high, burst_mult=burst_mult, p_on=p_on, p_off=p_off,
                             day_steps=day_steps)

    def base(self, t, phase):
        x = 2 * math.pi * (t / self.day_steps + phase)
        prof = 0.6 * (0.5 - 0.5 * np.cos(x)) + 0.4 * (0.5 - 0.5 * np.cos(2 * x)) ** 2
        return self.low + (self.high - self.low) * prof

    def trace(self, seed, steps):
        g = _rng(self.name, seed)
        t = np.arange(steps, dtype=float)
        lam = self.base(t, g.uniform(0, 1))
        state = 0
        u = g.uniform(size=steps)
        mult = np.ones(steps)
        for i in range(steps):
            if state == 0 and u[i] < self.p_on:
                state = 1
            elif state == 1 and u[i] < self.p_off:
                state = 0
            mult[i] = self.burst_mult if state else 1.0
        lam = lam * mult
        return g.poisson(lam * CONTROL_STEP_S) / CONTROL_STEP_S


class W3:
    """Regime shift: W0-like sine regime, then at a random point in [0.4, 0.6] T a different regime
    (slower period, higher level, MMPP bursts)."""
    name = "W3"

    def __init__(self):
        self.a = W0(spike_amp=0.0)
        self.b = W1(low=6000.0, high=14000.0, burst_mult=1.4, p_on=0.006, p_off=0.08, day_steps=800)

    def trace(self, seed, steps):
        g = _rng(self.name, seed)
        cut = int(steps * g.uniform(0.4, 0.6))
        sub = int(g.integers(0, 2**31 - 1))
        return np.concatenate([self.a.trace(sub, cut), self.b.trace(sub, steps - cut)])


class W2:
    """Stub: replay a CSV trace (no download). Column `column` (default: the only numeric column; ambiguous -> error).

    The raw series is split chronologically BEFORE any normalisation or windowing: the first `splits[0]` share
    is train, the next `splits[1]` tune, the rest held-out. Scaling uses train statistics only
    (train max -> `peak_ops`). A seed draws a window of `steps` rows from the segment of its own seed range
    (SEED_RANGES), wrapping inside that segment if it is shorter than `steps`."""
    name = "W2"

    def __init__(self, path, column=None, peak_ops=20000.0, splits=(0.6, 0.2)):
        self.path, self.column, self.peak_ops, self.splits = path, column, peak_ops, splits
        self._seg = None

    def _read(self):
        vals = []
        with open(self.path, newline="", encoding="utf-8") as f:
            col = self.column
            for row in csv.DictReader(f):
                if col is None:
                    num = []
                    for k, v in row.items():
                        try:
                            float(v)
                            num.append(k)
                        except (TypeError, ValueError):
                            continue
                    if len(num) != 1:
                        raise ValueError(f"W2: pass column= explicitly (numeric columns: {num})")
                    col = num[0]
                vals.append(float(row[col]))
        return np.asarray(vals, dtype=float)

    def segments(self):
        if self._seg is None:
            raw = self._read()
            n = raw.size
            i1 = int(n * self.splits[0])
            i2 = int(n * (self.splits[0] + self.splits[1]))
            parts = {"train": raw[:i1], "tune": raw[i1:i2], "heldout": raw[i2:]}
            if any(p.size == 0 for p in parts.values()):
                raise ValueError("W2 trace too short for the train/tune/held-out split")
            scale = parts["train"].max()
            if scale <= 0:
                raise ValueError("W2 train segment is non-positive")
            self._seg = {k: v / scale * self.peak_ops for k, v in parts.items()}
        return self._seg

    def trace(self, seed, steps):
        split = next((k for k, r in SEED_RANGES.items() if seed in r), None)
        if split is None:
            raise ValueError(f"seed {seed} is in no split range")
        d = self.segments()["heldout" if split == "test" else split]   # W2 has one held-out segment
        start = int(_rng(self.name, seed).integers(0, len(d)))
        return np.resize(np.roll(d, -start), steps)


class Constant:
    """Constant-load control (not part of the superiority gate)."""
    name = "CONST"

    def __init__(self, rate=8000.0):
        self.rate = rate

    def trace(self, seed, steps):
        return np.full(steps, float(self.rate))


def make_workload(name, **kw):
    table = {"W0": W0, "W1": W1, "W3": W3, "W2": W2, "CONST": Constant}
    return table[name](**kw)
