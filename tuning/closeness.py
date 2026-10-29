"""Closeness of a cycle's results to the DInos paper simulation picture, as one percentage.

Target (docs/papers/dinos_claims.md C5): cumulative reward DERP 2.2e5, DInos 3.8e6 (ratio 17.3x). The original
plotting scripts sum 5 eval episodes, so we compare the 5-episode sum ("sum" in summary.json).

Score = mean of three parts, each in [0, 1]:
  derp  = 1 - |log10(DERP_sum / 2.2e5)|   (0 if DERP_sum <= 0 or off by >= 10x)
  dinos = 1 - |log10(DInos_sum / 3.8e6)|  (same)
  ratio = 1 - |log10(r / 17.3)| / log10(17.3), r = DInos_sum / DERP_sum (0 if r <= 0); for other targets the
          denominator is max(|log10 target ratio|, 1) so a target ratio near 1 (legacy-fair) does not divide by ~0
"DInos" = dinos_t (DInos + transfer, as in the paper), NOT lstm_scratch.
Usage: closeness.py <results_dir> [--variant v2|dsec|...] [--target paper|legacy-orig|legacy-fair]
  (best DInos and DERP config of that variant). --target paper (default) = 2.2e5 / 3.8e6 above; legacy-orig /
  legacy-fair = the numbers recomputed from the original trained weights through the verbatim legacy port
  (tuning/legacy/targets.json, written by tuning/legacy/recompute_targets.py; see tuning/legacy/README.md).
"""
import json
import math
import os
import sys

DERP_T, DINOS_T = 2.2e5, 3.8e6
RATIO_T = DINOS_T / DERP_T
TARGETS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy", "targets.json")


def targets(name="paper"):
    """(derp, dinos) target sums for 'paper' | 'legacy-orig' | 'legacy-fair'."""
    if name == "paper":
        return DERP_T, DINOS_T
    if name not in ("legacy-orig", "legacy-fair"):
        raise SystemExit(f"unknown target {name}")
    with open(TARGETS_JSON) as f:
        t = json.load(f)[name]
    return float(t["derp"]), float(t["dinos"])


def part(x, target):
    if x is None or x <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(math.log10(x / target)))


def ratio_part(dinos, derp, ratio_t=RATIO_T):
    if dinos is None or derp is None or dinos <= 0 or derp <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(math.log10((dinos / derp) / ratio_t)) / max(abs(math.log10(ratio_t)), 1.0))


def score(dinos_sum, derp_sum, target="paper"):
    de_t, di_t = targets(target)
    p = {"derp": part(derp_sum, de_t), "dinos": part(dinos_sum, di_t), "ratio": ratio_part(dinos_sum, derp_sum, di_t / de_t)}
    p["total"] = sum(p.values()) / 3
    return p


def best(rows, agent, variant):
    rs = [r for r in rows if r["config"].startswith(agent + "_") and (variant is None or r["config"].endswith(variant))]
    return max(rs, key=lambda r: r["sum"]) if rs else None


def main(argv):
    d = argv[1]
    variant = argv[argv.index("--variant") + 1] if "--variant" in argv else None
    target = argv[argv.index("--target") + 1] if "--target" in argv else "paper"
    rows = json.load(open(f"{d}/summary.json"))
    di, de = best(rows, "dinos_t", variant), best(rows, "derp", variant)
    s = score(di and di["sum"], de and de["sum"], target)
    print(json.dumps({"target": dict(zip(("derp", "dinos"), targets(target)), name=target),"dinos": di and {k: di[k] for k in ("config", "n", "sum")},
                      "derp": de and {k: de[k] for k in ("config", "n", "sum")},
                      "score": {k: round(v, 3) for k, v in s.items()}}, indent=1))


if __name__ == "__main__":
    main(sys.argv)
