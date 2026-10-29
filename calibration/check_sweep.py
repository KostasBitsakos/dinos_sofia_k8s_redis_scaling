#!/usr/bin/env python3
"""Offline validity check of stage-A sweep run dirs (read-only): counts and ranges per sweep, using fit.py's own validity
rules (generator lag, errors, CPU telemetry, host interference). Reports; never fits or invents.
Usage: check_sweep.py RUN_DIR [RUN_DIR ...]   (exit 1 if a run has no usable bracketed sweep)"""
import json
import sys

import fit


def check(d):
    rows, events, tele = fit.load_run(d)
    table = fit.level_table(rows, tele)
    out = {"run": d, "metric_rows": len(rows), "telemetry_rows": len(tele), "events": len(events),
           "probe": [], "sweep": [], "capacity": None, "reason": None, "problems": []}
    for (run, c, ph, lv), t in sorted(table.items(), key=lambda kv: (kv[0][2], kv[0][3])):
        rec = {"pods": c, "level": lv, "n_windows": t["n"], "target": t["target"], "offered": round(t["offered"], 1),
               "served": round(t["served"], 1), "drop": round(t["drop"], 1), "timeout": round(t["timeout"], 1),
               "p99_ms": round(t["p99"], 2) if t["p99"] == t["p99"] else None, "pods_busy": t.get("pods_busy"),
               "invalid": t["invalid"]}
        out["probe" if ph == 0 else "sweep"].append(rec)
    if not out["sweep"]:
        out["problems"].append("no phase-1 sweep levels")
    sw = [t for (r, c, ph, lv), t in table.items() if ph == 1]
    good = [t for t in sw if not t["invalid"]]
    out["valid_sweep_levels"] = f"{len(good)}/{len(sw)}"
    if good:
        cap, why = fit.sweep_capacity(good)
        out["capacity"], out["reason"] = cap, why
    if out["capacity"] is None:
        out["problems"].append(out["reason"] or "no valid levels")
    for ev in events:
        if ev["event"] in ("memory_abort", "loadgen_failed"):
            out["problems"].append(f"event {ev['event']}")
    return out


def main():
    bad = 0
    for d in sys.argv[1:]:
        r = check(d)
        print(json.dumps(r, indent=1, default=str))
        bad += bool(r["problems"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
