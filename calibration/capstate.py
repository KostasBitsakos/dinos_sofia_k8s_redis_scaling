"""Capacity estimates handed from one sweep entry to later entries of the SAME frozen spec (stage B).

run_c.py writes the saturation-probe estimate of (tag, c) here; run_transition.py / run_metric_lag.py read it with
--capacity-auto so one spec can contain sweeps and the transitions that need C(hi) without a second approval.
The probe estimate is a lower-biased number (stage A c=1: probe 101 506 ops/s, sweep served up to 118 037); callers use
saturating fractions > 1 with it. Missing exact values are interpolated / extrapolated linearly in c and the method is
returned so the caller can log it. Nothing here is a result; fit.py is the only source of calibrated capacities.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def path():
    return os.environ.get("QRL_CAPACITY_STATE") or os.path.join(HERE, ".capacity_est.json")


def _load():
    try:
        with open(path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def record(tag, c, capacity, run=None):
    d = _load()
    d[f"{tag}:{int(c)}"] = {"capacity_est": float(capacity), "run": run}
    tmp = path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1, sort_keys=True)
    os.replace(tmp, path())


def estimate(tag, c):
    """(capacity_ops_per_s, method) for pod count c under `tag`; raises LookupError when nothing is recorded for the tag."""
    pts = sorted((int(k.split(":")[1]), v["capacity_est"]) for k, v in _load().items() if k.split(":")[0] == tag)
    if not pts:
        raise LookupError(f"no capacity estimate recorded for tag {tag!r} (run the sweeps of this tag first)")
    for pc, v in pts:
        if pc == c:
            return v, "exact"
    lo = [p for p in pts if p[0] < c]
    hi = [p for p in pts if p[0] > c]
    if lo and hi:
        (c0, v0), (c1, v1) = lo[-1], hi[0]
        return v0 + (v1 - v0) * (c - c0) / (c1 - c0), f"interpolated {c0}..{c1}"
    c0, v0 = (lo[-1] if lo else hi[0])
    return v0 * c / c0, f"linear extrapolation from c={c0}"
