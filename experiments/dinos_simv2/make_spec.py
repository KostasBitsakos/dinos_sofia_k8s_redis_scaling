#!/usr/bin/env python3
"""Generates spec_explore_w0.json (exploratory W0, PLAN 8.1 amendment R-20261017-014, PREREG_explore_w0.md).
The older specs (w0_v21, preflight_v21, diagnose_v21, preflight_rev1) are HISTORICAL: their manifests record what was
approved/run and are no longer regenerated (`--historical` rewrites them, do not use it for approved specs).
Reward setting: every NEW spec names reward_mode explicitly (cost|dinos) and carries the frozen learn_scale; historical specs
(legacy=True) name neither and stay bit-identical. spec_explore_w0.json is tied to R-20261017-015: never regenerated.
Also spec_preflight_rev1.json (protocol rev 1, PREREG_rev1.md) and spec_diagnose_v21.json (diagnose of the failed preflight, no re-tuning).
Generate spec_w0_v21.json (Sim v2.1, R-20261017-001) and spec_preflight_v21.json.

W0 spec: 3 seeds x {dinos_t(pretrained 5), lstm_scratch, derp} at 100k steps + one baselines entry (threshold,
fixed(c), always-min/max, oracle). Training uses the default config; every entry is evaluated on FRESH test episodes
(seeds 20000..) under the default config and every sensitivity config of sim_v2/sensitivity_v21.json (evaluation
only). Per-entry wall-clock limit 1.5 h, max_parallel 5.
Preflight spec: ONE validation-only entry (full grid tuning + fixed(c) + always-min/max + oracle, tune seeds
1000-1004, default config) with component timings; wall-clock 1 h.
Both specs carry `frozen_sha256` for every file in freeze.FROZEN and `frozen_git_commit` (HEAD); every entry's
command has `--freeze <spec>`, and train_eval.py verifies hashes + git state before doing anything. This script
refuses to run if any frozen path is not committed (commit the code first, then generate and commit the specs).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import freeze  # noqa: E402
import adapter  # noqa: E402

PY = "../../../.venv/bin/python"
ENV = ["env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"]
COMMON = ["--workload", "W0", "--eval-episodes", "5", "--sensitivity", "all"]
W0_SPEC, PRE_SPEC, DIAG_SPEC = "spec_w0_v21.json", "spec_preflight_v21.json", "spec_diagnose_v21.json"
REV1_SPEC = "spec_preflight_rev1.json"
EXPLORE_SPEC = "spec_explore_w0.json"
HISTORICAL = (W0_SPEC, PRE_SPEC, DIAG_SPEC, REV1_SPEC, EXPLORE_SPEC)   # EXPLORE_SPEC: approved as R-20261017-015
LEARN_SCALE = adapter.REWARD_SCALE    # frozen learning scale written into new specs (R-20261017-017)


def _reward_args(reward_mode, legacy):
    """Command-line flags and spec fields for the reward setting. A NEW spec must name its reward mode explicitly
    (and carries the frozen learn_scale); `legacy=True` (historical specs only, kept bit-identical) names neither."""
    if legacy:
        if reward_mode not in (None, "cost"):
            raise SystemExit("legacy (historical) specs are reward_mode 'cost' only")
        return [], {}
    if reward_mode not in ("cost", "dinos"):
        raise SystemExit("make_spec: reward_mode must be given explicitly ('cost' or 'dinos') for a new spec")
    return (["--reward-mode", reward_mode, "--learn-scale", repr(float(LEARN_SCALE))],
            {"reward_mode": reward_mode, "learn_scale": float(LEARN_SCALE)})

AMENDMENT = "R-20261017-014"
PREFLIGHT_JSON = "/home/ubuntu/qrl-lab/.lab/runs/20261017-140714-R-20261017-004-e0-a1/preflight.json"


def _frozen(allow_dirty=False):
    paths = [os.path.join(HERE, r) for r in freeze.FROZEN]
    dirty = freeze.dirty_paths(paths, HERE)
    if dirty and not allow_dirty:
        raise SystemExit(f"frozen paths not committed: {dirty}")
    return {"frozen_sha256": freeze.make_manifest(HERE), "frozen_git_commit": freeze.git_head(HERE)}


def build(allow_dirty=False, reward_mode=None, legacy=False):
    ra, rf = _reward_args(reward_mode, legacy)
    fz = ["--freeze", W0_SPEC] + ra
    entries = [{"name": "baselines-w0-v21", "seed": 0,
                "cmd": ENV + [PY, "train_eval.py", "--agent", "baselines", "--seed", "0"] + COMMON + fz}]
    for seed in (0, 1, 2):
        for agent, extra in (("dinos_t", ["--pretrained", "5"]), ("lstm_scratch", []), ("derp", [])):
            entries.append({"name": f"{agent}-w0-v21-s{seed}", "seed": seed,
                            "cmd": ENV + [PY, "train_eval.py", "--agent", agent, "--seed", str(seed),
                                          "--train-steps", "100000"] + COMMON + extra + fz})
    return {"name": "dinos-simv21-w0-provisional", "scope": "sim", "cwd": "code/experiments/dinos_simv2",
            "entries": entries, "max_parallel": 5, "max_attempts": 1, "limits": {"wall_clock_hours": 1.5},
            **rf, **_frozen(allow_dirty)}


def build_preflight(allow_dirty=False, reward_mode=None, legacy=False):
    ra, rf = _reward_args(reward_mode, legacy)
    cmd = ENV + [PY, "train_eval.py", "--agent", "preflight", "--seed", "0", "--workload", "W0",
                 "--freeze", PRE_SPEC] + ra
    return {"name": "dinos-simv21-preflight", "scope": "sim", "cwd": "code/experiments/dinos_simv2",
            "entries": [{"name": "preflight-w0-v21", "seed": 0, "cmd": cmd}],
            "max_parallel": 1, "max_attempts": 1, "limits": {"wall_clock_hours": 1.0}, **rf, **_frozen(allow_dirty)}


def build_diagnose(allow_dirty=False, reward_mode=None, legacy=False):
    ra, rf = _reward_args(reward_mode, legacy)
    cmd = ENV + [PY, "train_eval.py", "--agent", "diagnose", "--seed", "0", "--workload", "W0",
                 "--preflight-json", PREFLIGHT_JSON, "--freeze", DIAG_SPEC] + ra
    return {"name": "dinos-simv21-diagnose", "scope": "sim", "cwd": "code/experiments/dinos_simv2",
            "entries": [{"name": "diagnose-w0-v21", "seed": 0, "cmd": cmd}],
            "max_parallel": 1, "max_attempts": 1, "limits": {"wall_clock_hours": 1.0}, **rf, **_frozen(allow_dirty)}


def build_rev1(allow_dirty=False, reward_mode=None, legacy=False):
    ra, rf = _reward_args(reward_mode, legacy)
    cmd = ENV + [PY, "train_eval.py", "--agent", "preflight_rev1", "--seed", "0", "--workload", "W0",
                 "--preflight-json", PREFLIGHT_JSON, "--freeze", REV1_SPEC] + ra
    return {"name": "dinos-simv21-preflight-rev1", "scope": "sim", "cwd": "code/experiments/dinos_simv2",
            "entries": [{"name": "preflight-rev1-w0-v21", "seed": 0, "cmd": cmd}],
            "max_parallel": 1, "max_attempts": 1, "limits": {"wall_clock_hours": 1.0}, **rf, **_frozen(allow_dirty)}


def build_explore(allow_dirty=False, reward_mode=None, legacy=False):
    """Exploratory W0: 1 comparator entry + 3 training seeds x {dinos_t(5), lstm_scratch, derp} x 100k steps, all on
    the registered block (explore_w0.json), default + every frozen sensitivity config; 2 h per entry, 5 parallel."""
    ra, rf = _reward_args(reward_mode, legacy)
    common = ["--workload", "W0", "--sensitivity", "all", "--protocol", "explore_w0", "--amendment-id", AMENDMENT,
              "--eval-episodes", "20", "--freeze", EXPLORE_SPEC] + ra
    entries = [{"name": "explore-w0-comparators", "seed": 0,
                "cmd": ENV + [PY, "train_eval.py", "--agent", "explore_baselines", "--seed", "0"] + common}]
    for seed in (0, 1, 2):
        for agent, extra in (("dinos_t", ["--pretrained", "5"]), ("lstm_scratch", []), ("derp", [])):
            entries.append({"name": f"explore-w0-{agent}-s{seed}", "seed": seed,
                            "cmd": ENV + [PY, "train_eval.py", "--agent", agent, "--seed", str(seed),
                                          "--train-steps", "100000"] + extra + common})
    return {"name": "dinos-simv21-explore-w0", "scope": "sim", "cwd": "code/experiments/dinos_simv2",
            "entries": entries, "max_parallel": 5, "max_attempts": 1, "limits": {"wall_clock_hours": 2.0},
            **rf, **_frozen(allow_dirty)}


if __name__ == "__main__":
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools")))
    from labdb import validate_spec
    # Nothing is regenerated by default: every existing spec is historical (approved/run). A new spec needs a new
    # builder that passes reward_mode explicitly. `--historical` rewrites the historical specs, do not use it for
    # approved ones (spec_explore_w0.json is tied to R-20261017-015).
    todo = []
    if "--historical" in sys.argv:
        todo += [(build(legacy=True), W0_SPEC), (build_preflight(legacy=True), PRE_SPEC),
                 (build_diagnose(legacy=True), DIAG_SPEC), (build_rev1(legacy=True), REV1_SPEC)]
        if "--include-approved" in sys.argv:
            todo += [(build_explore(legacy=True), EXPLORE_SPEC)]
    for spec, fn in todo:
        validate_spec(spec)
        path = os.path.join(HERE, fn)
        with open(path, "w") as f:
            json.dump(spec, f, indent=2)
            f.write("\n")
        print(path, "validate_spec OK,", len(spec["entries"]), "entries")
