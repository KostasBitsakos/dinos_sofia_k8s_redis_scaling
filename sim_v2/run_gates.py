#!/usr/bin/env python3
"""Run the frozen Sim v2 gate suite. Start only through tools/launch_run.py with an approved spec.

  python run_gates.py --config gates_config.json --seed 0 [--workers N] [--quick]

--seed selects the held-out seed block: seeds heldout_seeds_start + seed*n_seeds ... (+n_seeds - 1).
--quick shrinks everything (2 seeds, 300 steps, tiny grids) for a smoke test; its verdict is meaningless.
Outputs in $QRL_RUN_DIR (default ./gates_out): metrics.jsonl, progress.json (PLAN §7), gates_result.json,
gates_report.md. Exit code 0 when the suite ran (pass or fail is in the outputs), non-zero on a crash.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gates import RANKED, evaluate, frozen_selections, heldout_seeds, run_unit, units  # noqa: E402
from params import SimParams  # noqa: E402


def write_progress(run_dir, step, total):
    tmp = os.path.join(run_dir, "progress.json.tmp")
    with open(tmp, "w") as f:
        json.dump({"step": step, "total": total}, f)
    os.replace(tmp, os.path.join(run_dir, "progress.json"))


def report_md(cfg, seed, seeds, results, verdict, rows, rank, viol, elapsed):
    L = ["# Sim v2 gate report", "",
         "**Provisional parameters (not calibrated). These numbers are not paper results.**", "",
         f"Seed block {seed}: held-out seeds {seeds[0]}..{seeds[-1]}; tune seeds {cfg['tune_seeds']}; "
         f"wall clock {elapsed:.0f} s.", "",
         "fixed(c) = starts with c ready pods, never scales. static-ramp(c) = starts at init_pods and ramps to c.",
         "", "## Verdict", ""]
    L += [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in verdict.items()]
    L += ["", "## Base config, held-out mean return (components: cost / sla / churn)", "",
          "| workload | policy | mean return | cost | sla | churn | pod-steps | changes |",
          "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if not r["full"]:
            continue
        for k in RANKED + ("oracle_replay",):
            c = r["components"][k]
            name = r["names"].get(k, k)
            L.append(f"| {r['workload']} | {name} | {sum(r['returns'][k]) / len(r['returns'][k]):.0f} | "
                     f"{c['cost']:.0f} | {c['sla']:.0f} | {c['churn']:.0f} | {c['pod_steps']:.0f} | "
                     f"{c['changes']:.0f} |")
        L.append(f"| {r['workload']} | **oracle bound (start init_pods)** | "
                 f"{sum(r['bounds']) / len(r['bounds']):.0f} | | | | | |")
    L += ["", "## G2: threshold minus each fixed(c) and static-ramp(c) (paired, one-sided)", "",
          "Lower bounds are one-sided 97.5% (= two-sided 95% CI), **unadjusted**. The gate decision uses "
          "Holm-adjusted p-values.", "",
          "| workload | static | mean diff | unadjusted lower bound | p | p Holm |", "|---|---|---|---|---|---|"]
    for x in rows:
        L.append(f"| {x['workload']} | {x['static']} | {x['mean']:.0f} | {x['lb_unadjusted']:.0f} | "
                 f"{x['p']:.2e} | {x['p_holm']:.2e} |")
    for mode in ("retuned", "frozen"):
        L += ["", f"## G4 ranking ({mode}): paired 95% CI of row minus column policy per config", "",
              "Sign: + significantly better, - significantly worse, 0 CI includes 0.", "",
              "| workload | pair | config | mean | 95% CI | sign |", "|---|---|---|---|---|---|"]
        for (w, a, b), lst in rank[mode].items():
            for x in lst:
                sg = {1: "+", -1: "-", 0: "0"}[x["sign"]]
                L.append(f"| {w} | {a} - {b} | {x['config']} | {x['mean']:.0f} | "
                         f"[{x['lo']:.0f}, {x['hi']:.0f}] | {sg} |")
    L += ["", "## Selections per config (retuned; frozen = base-config row)", ""]
    for r in results:
        L.append(f"- {r['workload']} {r['label']}: {r['names']}")
    L += ["", "## Oracle margins (reported, not gated)", ""]
    for r in results:
        n = len(r["bounds"])
        b = sum(r["bounds"]) / n
        t = sum(r["returns"]["threshold"]) / n
        o = sum(r["returns"]["oracle_replay"]) / n
        L.append(f"- {r['workload']} {r['label']}: bound - threshold = {b - t:.0f} "
                 f"({100 * (b - t) / abs(t):.1f}% of |threshold|); oracle replay - threshold = {o - t:.0f}")
    if viol:
        L += ["", "## G3 violations", ""] + [f"- {v}" for v in viol[:50]]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "gates_config.json"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("QRL_RUN_SEED", 0)))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    with open(a.config) as f:
        cfg = json.load(f)
    if a.quick:
        cfg.update(n_seeds=2, tune_seeds=cfg["tune_seeds"][:1],
                   threshold_grid={"up": [50], "down": [30], "cooldown": [0, 2]},
                   sensitivity={"theta": [0.9, 1.0], "delta_factor": [1.0]})
        cfg.setdefault("param_overrides", {})["episode_steps"] = 300
    run_dir = os.environ.get("QRL_RUN_DIR") or os.path.join(os.getcwd(), "gates_out")
    os.makedirs(run_dir, exist_ok=True)
    SimParams(**cfg.get("param_overrides", {}))  # validate overrides early
    seeds = heldout_seeds(cfg, a.seed)
    all_units = units(cfg)
    base_units = [u for u in all_units if u[3]]
    rest_units = [u for u in all_units if not u[3]]
    total = len(all_units)
    write_progress(run_dir, 0, total)
    t0 = time.time()
    results = []
    mpath = os.path.join(run_dir, "metrics.jsonl")
    pool = Pool(a.workers) if a.workers > 1 else None
    mapper = pool.imap if pool else map

    def consume(it):
        for r in it:
            results.append(r)
            i = len(results)
            n = len(r["bounds"])
            line = {"step": i, "full": int(r["full"]), "theta": r["overrides"]["theta"],
                    "rebalance_loss": r["overrides"]["rebalance_loss"],
                    "workload_id": ["W0", "W1", "W2", "W3"].index(r["workload"]),
                    "bound_mean": sum(r["bounds"]) / n, "max_conservation_rel": r["cons"]}
            for k in r["returns"]:
                if "_best" in k or k in ("always_min", "always_max", "threshold", "threshold_frozen",
                                         "oracle_replay"):
                    line[f"{k}_mean"] = sum(r["returns"][k]) / n
            with open(mpath, "a") as f:
                f.write(json.dumps(line) + "\n")
            write_progress(run_dir, i, total)
            print(f"[{i}/{total}] {r['workload']} {r['label']} {r['names']}", flush=True)

    # phase 1: base config (selections to freeze); phase 2: sensitivity configs with frozen selections
    consume(mapper(run_unit, [(lab, ov, w, full, cfg, seeds, None) for lab, ov, w, full in base_units]))
    frozen = frozen_selections(results)
    consume(mapper(run_unit, [(lab, ov, w, full, cfg, seeds, frozen[w]) for lab, ov, w, full in rest_units]))
    if pool:
        pool.close()
        pool.join()
    verdict, rows, rank, viol = evaluate(cfg, results)
    elapsed = time.time() - t0
    with open(os.path.join(run_dir, "gates_result.json"), "w") as f:
        json.dump(dict(config=cfg, seed=a.seed, heldout_seeds=seeds, verdict=verdict, g2=rows,
                       ranking={m: [dict(workload=k[0], a=k[1], b=k[2], configs=v) for k, v in d.items()]
                                for m, d in rank.items()},
                       oracle_violations=viol, units=results, quick=a.quick, elapsed_s=elapsed), f, indent=1)
    with open(os.path.join(run_dir, "gates_report.md"), "w") as f:
        f.write(report_md(cfg, a.seed, seeds, results, verdict, rows, rank, viol, elapsed))
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
