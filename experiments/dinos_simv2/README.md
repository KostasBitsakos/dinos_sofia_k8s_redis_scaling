# DInos agents on Sim v2 (provisional) - W0

Nothing here is a paper result: Sim v2 parameters are provisional and unmeasured (`code/sim_v2/README.md`).

Files: `adapter.py` (Sim v2 -> 10x6 window), `agents.py` (imports the researcher's agents), `train_eval.py`
(train + held-out evaluation + baselines), `make_spec.py` -> `spec_w0.json`, `tests/`.
Tests: `cd code/experiments/dinos_simv2 && ../../../.venv/bin/python -m pytest -q tests` (about 30 s).

## Feature mapping (documented constants in adapter.py)
Slots 0..5 = `completed_ops, cpu_pct, p95_ms, failed_per_s, ready, pending`, normalised with FIXED constants:
ops / (mu*max_pods = 30000), cpu / 100, log10(1+p95_ms) / 4, log1p(failed) / log1p(30000), ready / 30, pending / 4.
Window = 10, initialised with the first observation repeated. Rewards are multiplied by 0.02 for learning only;
all reported numbers are raw Sim v2 rewards (return = -(cost + SLA + churn), components logged).

## Transfer (dinos_t)
`modelCPU_k.pth` (the researcher's LSTM 6-128x3-3, trained on the old environment) is loaded as a **weight
initialisation only**. Slot meanings differ from the old environment (load, capacity, cpu, memory, latency, vms),
so the pretrained function is not expected to be meaningful on the new features; what is tested is whether this
initialisation (plus the pretrained-agent epsilon 0.2) helps relative to `lstm_scratch`. Both use the fixed
non-legacy `DDQNAgentLSTM` (double DQN, F3.3 zero hidden state, batched replay, target sync every 100 replays),
imported from `code/sim/timeseries/Pytorch/lstm_transfer`, with its default hyper-parameters.
Pretrained: 5 in the spec (`--pretrained {1,2,5,10,20}`).

## DERP (derp)
`DDQNAgent` of `code/sim/timeseries/Pytorch/dense/ddqnagent.py` (dense 6-128-128-3 net), used as published (no
fixes: per-sample replay, target sync after every replay, memory 500). It sees only the latest observation row
(no window). Trained from scratch (the dense checkpoints have the old-env features).

## Protocol
Training: 5000-step episodes with train seeds 0-999 (a per-run permutation), epsilon-greedy, one replay of batch 16
per env step. Evaluation: greedy, fresh test seeds 20000.. (`--eval-episodes`; 10000-10999 are burned, see Seed audit), identical for all methods
(paired). Baselines (`--agent baselines`): threshold (grid of `policies.tune_threshold`) and fixed(c) (best c) tuned
on tune seeds 1000-1004, always_min, always_max, and OracleReplay with the certified oracle bound logged
(`oracle_bound`; the replay is the achievable clairvoyant open-loop return, the bound is looser under overload).
Telemetry: `metrics.jsonl` (train rows every `--log-every` steps; eval rows per episode with return, cost, sla, churn,
mean pods, SLA-violation rate = bad requests / arrived, mean p95 and p95 of step-p95), `progress.json`,
`checkpoints/model.pth`, `plots/evaluation_metrics.png` (2x2 load/CPU/latency/failures, pods on the twin axis) and
`plots/training_curve.png`. Baseline eval rows use `step` = baseline index (0 threshold, 1 fixed_best,
2 always_min, 3 always_max, 4 oracle).

## Limits
One workload, one reward weighting (alpha, beta, gamma provisional), epsilon schedule from the agent defaults
(0.995 per replay to 0.1), no hyper-parameter search; 3 seeds are a screening number, not a confidence statement.

## Sim v2.1 (R-20261017-001)
`spec_w0_v21.json` (from `make_spec.py`, validated with `tools/labdb.validate_spec`; not started): 3 seeds x
{dinos_t, lstm_scratch, derp} at 100k steps + one baselines entry, 5 held-out episodes each, evaluated under the default
config and every config of `sim_v2/sensitivity_v21.json` (`--sensitivity all`; training always on default;
`config_index` in `metrics.jsonl`, names in `configs.json`). Baseline selections (threshold, fixed c) are made once
on tune seeds 1000-1004 under the default config and frozen for all configs; the oracle bound is re-solved per config.
The threshold grid is `THRESHOLD_GRID_V21` (up 25..95, down 5..60, cooldown 0..8, 435 valid points); the baselines run
fails (`assert_threshold_interior`) if the selected point is on the edge (cooldown 0 is the natural floor, not an edge).
Spec edits require rerunning `make_spec.py` (the test compares it with the file; it hashes the frozen sources).

## Seed audit (Codex review of c5beef7..99b1ab4; R-20261017-001)
Seed ranges (`sim_v2/workloads.SEED_RANGES`, `check_seed`): train 0-999, tune/validation 1000-1999, held-out
10000-10999, and the new **test 20000-20999**.
Where seeds were used (grep of code, tests and configs at this commit):
- tune seeds 1000-1004: baselines selection (`TUNE_SEEDS`) and the Sim v2 gate suite (`gates_config.json` `tune_seeds`);
  both are validation use, so they are legitimate for selection and are never evaluation data.
- held-out 10000-10004: used by `train_eval.py` for evaluation (`HELDOUT0 = 10000`) and by the smoke runs and tests of
  this experiment (the eval-seed assertion, my earlier smokes); 10000-10003 by the sim_v2 unit tests (workload
  determinism, oracle bound, regression digests); 10000-10009 and beyond (`heldout_seeds_start + seed*n_seeds`) by the
  Sim v2 gate runs. I did not inspect the stored gate run outputs (outside my scope), so any of 10000-10999 is treated
  as seen.
Outcome: evaluation seeds 10000-10004 were already seen, so **held-out evaluation moved to fresh seeds 20000-20004**
(`TEST0`, `eval_seeds`). 20000-20999 appear in no code, test, config, smoke or gate; `check_seed("test", s)` guards them.
10000-10999 stay in `SEED_RANGES["heldout"]` as a burned range (`train_eval.BURNED`) and are never evaluated on.
Test `test_seed_sets_disjoint_and_fresh` asserts that train / validation / test seed sets are pairwise disjoint and
that test does not touch the burned range. Rule: smokes and unit tests must not use 20000-20999.
W2 (trace replay) maps the test range to its held-out segment.

## Freeze enforcement
Specs carry `frozen_sha256` for every file in `freeze.FROZEN` (agents, adapter, train_eval, freeze, sim_v2
env/params/oracle/policies/workloads/sensitivity + `sensitivity_v21.json`, the LSTM and DERP agent sources and the
`modelCPU_5.pth` checkpoint) and `frozen_git_commit`. Every entry has `--freeze <spec>`; `train_eval.py` calls
`freeze.verify_freeze` first (before any training or tuning) and refuses if a file is missing/untracked/modified,
a sha256 differs, the commit is not an ancestor of HEAD, or a frozen file changed since it. After the agent is built it
also checks that every source file imported from `code/` is in the manifest. Dependency versions (python, numpy,
torch, pandas, platform) and the git HEAD go to `freeze_report.json` in the run directory. `--unfrozen` /
`QRL_ALLOW_UNFROZEN=1` exist for tests only and are asserted absent from the specs. Workflow: commit code, run
`make_spec.py` (refuses on a dirty frozen path), commit the specs.

## PLAN 8.1 gate and preflight
`gate_check`: the tuned threshold must beat fixed_best, always_min and always_max (mean paired return) and the oracle
replay must not lose to it. `baselines` runs it on the validation seeds before evaluating and raises (writing
`gate_failed.json`) if it fails; the same check on the test seeds is reported in `baselines_summary.json`.
`spec_preflight_v21.json` (one entry, `--agent preflight`, validation seeds and default config only, 1 h limit) writes
`preflight.json` (per-policy paired returns, cost / SLA / churn decomposition, pod-seconds, component runtimes,
env-step throughput) and `preflight_table.txt`. Its numbers are in-sample (selection on the same episodes).

## Sim v2.1 protocol rev 1 (R-20261017-012, option 1)
Pre-registered in `PREREG_rev1.md` + `gate_rev1.json` (both frozen) before any rev-1 run; the original preflight FAIL
(R-20261017-004, sha256 pinned and verified) stands and is reported alongside. Common start c0 = `SimParams.init_pods` = 6;
`StaticRamp` (one pod per step toward c, then hold; `sim_v2/policies.py` unchanged) selected on tune seeds 1000-1004 with
ties broken to the smallest c; threshold 40/25/0 loaded, not re-tuned; gate on fresh validation seeds **1200-1209**
(n = 10, paired 95% bootstrap CI, 10k resamples, seed 20261017). Seed audit: 1100-1109 were the first choice and are
BURNED (a smoke test evaluated them); 1200-1209 appear nowhere else (grep, and no range-iterating code path). Tests cannot
evaluate them (conftest guard; `guard_gate_block` needs `--freeze`); smokes use `--rev1-eval-seeds` with 1000-1004.
`--agent preflight_rev1 --preflight-json PATH` writes `preflight_rev1.json` / `preflight_rev1_table.txt` and raises if the
gate fails. Agents (`run_agent`) refuse unless given a passing rev-1 artifact + sha256 or an approved `--amendment-id`
(recorded). Spec: `spec_preflight_rev1.json` (1 entry, 1 h limit). The agent entries of `spec_w0_v21.json` carry neither
flag yet, so they will refuse until a spec with an authorisation is generated and approved.

## Exploratory W0 (PLAN 8.1 amendment R-20261017-014)
Pre-registered in `PREREG_explore_w0.md` + `explore_w0.json` (frozen, before any exploratory run). Gates stay FAILED;
nothing here is confirmatory. Spec `spec_explore_w0.json` (`make_spec.py`): 1 comparator entry (`--agent
explore_baselines`: static_ramp_best c = 22 and preprovisioned fixed_best c = 22 primary, threshold 40/25/0 and K8sHPA
disclosed, always_min/max and oracle replay (clairvoyant +-1 reference, not an upper bound on multi-pod HPA) reference; all loaded from sha256-pinned artifacts, K8sHPA target 40 %
re-derived on 1000-1004 and checked) + 3 training seeds x {dinos_t, lstm_scratch, derp} x 100k steps; evaluation on the
registered block of 20 development-range seeds under the default config (primary) and every frozen sensitivity config
(descriptive); last checkpoint, no selection. Every entry needs `--amendment-id R-20261017-014`, verified read-only as
`approved` in `~/qrl-lab/.lab/lab.db`. Disclosed asymmetry: K8sHPA changes many pods per step, the agents +-1.
Afterwards: `analyze_explore_w0.py --out DIR <10 run dirs>` (paired bootstrap CIs vs static_ramp_best and fixed_best, conditional on the three trained models per agent).
`select_hpa_dev.py` re-creates `hpa_selection_dev.json`. Specs of the earlier protocols are HISTORICAL (not regenerated).

## Reward mode, learning scale and diagnostics (R-20261017-017)
- `--reward-mode cost|dinos` (train_eval.py): the simulator reward of the run (training, evaluation, sensitivity
  configs). Absent = `cost` (old specs). New specs must name it: `make_spec` refuses a spec without `reward_mode`
  (`legacy=True` is for historical specs only; `spec_explore_w0.json` is tied to R-20261017-015 and is never
  regenerated). The baselines/gate/pin agents (`baselines`, `preflight*`, `diagnose`, `explore_baselines`) are `cost`
  only. Recorded in `run_config.json` and in every training row of `metrics.jsonl` (`reward_mode_dinos`, `learn_scale`),
  and in `explore_eval.json`.
- `--learn-scale` (default 0.02 = `adapter.REWARD_SCALE`): one positive factor on the reward stored for learning, the
  same for every learned agent. Reported returns are raw. No new value is selected here.
- Diagnostics every `--diag-every` steps (default `--log-every`): `diag_td_abs_{mean,p50,p90,p99,max}`,
  `diag_q_abs_{mean,max}`, `diag_target_abs_{mean,max}`, `diag_reward_abs_mean`, `diag_huber_saturation`
  (share of |TD| > 1) in learning units, on a batch drawn with a private RNG (training is not perturbed).
- Resolution-window convention of the `dinos` reward: see `sim_v2/env.py` `_dinos_reward`.
- `select_baselines_dinos.py`: dev-seed (1000-1004) reselection of K8sHPA target, threshold, fixed and static_ramp c
  under a reward mode -> `selection_baselines_<mode>.json`. No training.
