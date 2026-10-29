#!/usr/bin/env python3
"""Record the K8sHPA target selection on development seeds 1000-1004 (default config) -> hpa_selection_dev.json.

Same style as the threshold grid: one grid (policies.HPA_TARGET_GRID), mean return over the selection seeds,
deterministic tie-break (smallest target), interior check. The exploratory baselines run repeats this selection and
refuses if the result differs from explore_w0.json `hpa.target_pct`. Takes about 30 s (18 x 5 episodes).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adapter  # noqa: E402,F401  (puts sim_v2 on sys.path)
import freeze  # noqa: E402
from params import SimParams  # noqa: E402
from policies import HPA_TARGET_GRID, tune_hpa  # noqa: E402

SEEDS = tuple(range(1000, 1005))


def main(path=os.path.join(HERE, "hpa_selection_dev.json")):
    t0 = time.time()
    best, table = tune_hpa(SimParams(), "W0", SEEDS, grid=HPA_TARGET_GRID)
    out = dict(workload="W0", config="default", seeds=list(SEEDS), grid=list(HPA_TARGET_GRID),
               table=[[t, r] for t, r in table], selected_target_pct=best.target,
               interior=bool(min(HPA_TARGET_GRID) < best.target < max(HPA_TARGET_GRID)),
               tie_break="smallest target among exactly equal mean returns", git_head=freeze.git_head(HERE),
               wall_s=time.time() - t0)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps({k: out[k] for k in ("selected_target_pct", "interior", "wall_s")}))
    return out


if __name__ == "__main__":
    main()
