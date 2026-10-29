#!/usr/bin/env python3
"""Generates the frozen stage-B run specs (deterministic; the committed JSON files are its output).
  python3 make_spec_stageB.py            -> spec_stageB.json            (main spec)
  python3 make_spec_stageB.py --optional16 -> spec_stageB_optional16.json (c=16 at 50m, separate approval)
Nothing here runs anything; entries are launched only by tools/launch_run.py after approval. See PREREG_stageB.md."""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "../../.venv/bin/python"
CTX, NS = "kubernetes-admin@kubernetes", "qrl-calib"
FRACS = "0.03,0.1,0.3,0.5,0.7,0.85,0.95,1.05,1.2,1.5,2.0"   # 2.0 = overload; low end gives CPU-vs-throughput linearity
LEVEL_S = "30"
REAL = {
    "kube_context": CTX, "namespace": NS, "deployment": "redis-autoscale-cluster",
    "rollback": "bash code/calibration/rollback.sh  (kubectl delete namespace qrl-calib; the default namespace is never touched)",
    "resource_limits": "namespace qrl-calib only; <= 17 pods; pod limits either 150m/8Mi requests + 200m/16Mi limits (as production) or "
                       "50m/8Mi requests + 50m/16Mi limits (stage-B variant k8s/qrl-calib-50m.yaml); load generator on the host: "
                       "<= 16 worker processes (stage A approval was <= 8: NEW, explicit approval item); a memory trip (pod cgroup working set "
                       "> 95% of 16Mi, fork/rewrite near the limit, unreadable memory) stops all scaling and triggers recovery",
    "data_loss": "accepted by the researcher for the calibration namespace (R-20261016-018, 2026-10-16)",
    "entries_must_run_in_order": True,
    "entry_gating": "every entry runs through gate.py: previous entry must have succeeded, else rollback.sh runs and the entry refuses; "
                    "sweep/transition entries also require a healthy cluster",
}


def entry(name, seed, script_args=None, health=False, allow_failed=False, raw=None):
    if raw:
        cmd = [PY, "gate.py", "--"] + raw
    else:
        cmd = [PY, "gate.py"] + (["--allow-failed"] if allow_failed else []) + (["--health"] if health else []) + ["--"] + [PY] + script_args
    return {"name": name, "seed": seed, "cmd": cmd}


def kube(*a):
    return ["kubectl", "--context", CTX, "--namespace", NS, *a]


def setup(es, yaml, tagname):
    es.append(entry(f"create-{tagname}", 0, raw=kube("apply", "-f", yaml)))
    es.append(entry(f"wait-3-pods-{tagname}", 0, raw=kube("rollout", "status", "deployment/redis-autoscale-cluster", "--timeout=180s")))
    es.append(entry(f"bootstrap-cluster-3-{tagname}", 0, ["scale_to.py", "--bootstrap", "3"]))


def sweep(es, seed, tag, c, rep, workers, extra=()):
    es.append(entry(f"sweep-{tag}-c{c}-{rep}", seed, ["run_c.py", "--c", str(c), "--seed", str(seed), "--tag", tag, "--workers", str(workers),
                                                       "--fractions", FRACS, "--level-s", LEVEL_S, *extra], health=True))


def delete_ns(es, name, final=False):
    es.append({"name": name, "seed": 0, "cmd": [PY, "gate.py"] + (["--allow-failed"] if final else []) + ["--", "bash", "rollback.sh"]})


def trans(es, seed, tag, lo, hi, keys, vsz, workers, reps=3, frac=1.2, pre=30, settle=45, label="saturated"):
    es.append(entry(f"transitions-{label}-{tag}-{lo}-{hi}-k{keys}x{vsz}B", seed,
                    ["run_transition.py", "--lo", str(lo), "--hi", str(hi), "--capacity-auto", "--tag", tag, "--rate-frac", str(frac),
                     "--reps", str(reps), "--pre-s", str(pre), "--settle-s", str(settle), "--keys", str(keys), "--value-size", str(vsz),
                     "--workers", str(workers), "--seed", str(seed)], health=True))


def main_spec():
    es, s = [], 1000
    # ---- namespace 1: production pod limits (200m) -------------------------------------------------------------
    setup(es, "k8s/qrl-calib.yaml", "200m")
    # c=4 at 200m is NOT run: ~400k ops/s expected capacity needs >= 1.2x = 480k offered, above the 16-worker generator budget of
    # ~358k ops/s (38 us/op, 0.85 core/worker); it could only measure the generator. c=2 (~200-240k) is marginal: levels above the budget are skipped by run_c.
    for c, w in ((1, 8), (2, 16)):
        s += 1; sweep(es, s, "200m", c, "a", w)
    for c, w in ((2, 16), (1, 8)):
        s += 1; sweep(es, s, "200m", c, "b", w)
    for r in (1, 2, 3):   # reward SLO (R-20261017-018): c=1 p95 at 50 % of max throughput, fine ladder around it, 3 repetitions
        s += 1
        es.append(entry(f"slo-200m-c1-r{r}", s, ["run_c.py", "--c", "1", "--capacity-auto", "--tag", "200m", "--fractions", "0.4,0.5,0.575,0.65",
                                                  "--level-s", "130", "--workers", "8", "--seed", str(s)], health=True))
    s += 1
    es.append(entry("metric-lag-200m-c1", s, ["run_metric_lag.py", "--c", "1", "--capacity-auto", "--tag", "200m", "--frac", "0.3",
                                               "--reps", "2", "--seed", str(s)], health=True))
    # memory: dataset ascending (0.55, 1.1, 2.2 MB total = at most 1.1 MB per pod at c=2, working set 0.65 + 0.07 = ~0.72 of 16Mi, margin to the 0.85 start gate), then connections
    for keys, w in ((500, 8), (1000, 8), (2000, 8), (2000, 2), (2000, 16)):
        s += 1
        es.append(entry(f"mem-200m-c2-k{keys}x1024B-w{w}", s, ["run_c.py", "--c", "2", "--capacity-auto", "--tag", "200m", "--keys", str(keys),
                                                               "--value-size", "1024", "--fractions", "0.3", "--level-s", "60",
                                                               "--workers", str(w), "--seed", str(s)], health=True))
    delete_ns(es, "delete-namespace-200m")
    # ---- namespace 2: 50m pod limits (generator ceiling avoided) -----------------------------------------------
    setup(es, "k8s/qrl-calib-50m.yaml", "50m")
    for c, w in ((1, 8), (2, 8), (4, 8), (8, 16)):
        s += 1; sweep(es, s, "50m", c, "a", w)
    for c, w in ((8, 16), (4, 8), (2, 8), (1, 8)):
        s += 1; sweep(es, s, "50m", c, "b", w)
    s += 1
    es.append(entry("overload-timeout-50m-c1", s, ["run_c.py", "--c", "1", "--capacity-auto", "--tag", "50m", "--timeout", "2",
                                                    "--max-outstanding", "20000", "--fractions", "1.0,1.5,2.0,3.0", "--level-s", "60",
                                                    "--workers", "8", "--seed", str(s)], health=True))
    s += 1
    trans(es, s, "50m", 2, 3, 2000, 64, 4, reps=10, frac=0.3, pre=10, settle=15, label="idle-startup")      # idle-ish: pod start-up distribution (>= 10)
    # C4: three offered loads (0.5, 0.9, 1.2 x C_est(hi)) at 2-3 with 1.2 MB data; pairs 4-5, 8-9 at 1.2; data volume 0.29 / 1.18 / 2.2 MB at 2-3 (1.2),
    # and 2.2 MB at 8-9. Volumes ascend because datasets only grow within a namespace. Per-pod data at lo <= 1.1 MB.
    s += 1; trans(es, s, "50m", 2, 3, 2000, 64, 8)
    for frac in (0.5, 0.9, 1.2):
        s += 1; trans(es, s, "50m", 2, 3, 2000, 512, 8, frac=frac, label=f"load{frac}")
    for lo, hi, w in ((4, 5, 12), (8, 9, 16)):
        s += 1; trans(es, s, "50m", lo, hi, 2000, 512, w)
    for lo, hi, w in ((2, 3, 8), (8, 9, 16)):
        s += 1; trans(es, s, "50m", lo, hi, 2000, 1024, w)
    delete_ns(es, "rollback-delete-namespace-final", final=True)
    return {"name": "simv2-calibration-stageB", "scope": "real", "cwd": "code/calibration", "max_parallel": 1, "max_attempts": 1,
            "limits": {"wall_clock_hours": 1.5}, "real": REAL, "entries": es}


def opt16_spec():
    es, s = [], 2000
    setup(es, "k8s/qrl-calib-50m.yaml", "50m")
    # c=16 needs ~400k ops/s of generator capacity; expected client-limited: fit.py's validity rules decide, that IS the result
    for rep in ("a", "b"):
        s += 1; sweep(es, s, "50m", 16, rep, 16, ("--max-rate", "700000"))
    delete_ns(es, "rollback-delete-namespace-final", final=True)
    return {"name": "simv2-calibration-stageB-optional-c16", "scope": "real", "cwd": "code/calibration", "max_parallel": 1, "max_attempts": 1,
            "limits": {"wall_clock_hours": 1.5}, "real": REAL, "entries": es}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optional16", action="store_true")
    a = ap.parse_args()
    spec, fn = (opt16_spec(), "spec_stageB_optional16.json") if a.optional16 else (main_spec(), "spec_stageB.json")
    with open(os.path.join(HERE, fn), "w") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    print(f"{fn}: {len(spec['entries'])} entries")


if __name__ == "__main__":
    main()
