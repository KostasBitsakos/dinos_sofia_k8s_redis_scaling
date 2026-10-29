# Pre-registration: exploratory W0 study (PLAN 8.1 amendment R-20261017-014)

Written and committed BEFORE any exploratory run. Machine-readable twin: `explore_w0.json`; both are in the freeze
manifest of `spec_explore_w0.json`. **EXPLORATORY: no confirmatory claim follows from this study.**

## Authorisation (PLAN 8.1 amendment, quoted verbatim)
> "The original and rev-1 gates failed and remain failed. After observing these failures, we authorize a bounded
> exploratory W0 study under unchanged, uncalibrated parameters. This waives §8.1's threshold-superiority
> prerequisite; it does not establish simulator validity. Static-ramp and preprovisioned fixed_best remain primary
> deployable comparators; oracle replay is a clairvoyant reference. No parameter, workload, or acceptance-rule
> changes will follow these results. Confirmatory claims require calibration and separately frozen evaluation."

PLAN 8.1 adds: a Kubernetes-HPA baseline (docs/hpa_spec.md) is a disclosed post-hoc comparator, frozen before the run;
final test seeds (20000+) stay untouched until the calibrated study. Gate results that stand: original preflight FAIL
(R-20261017-004) and rev-1 FAIL (R-20261017-013; threshold vs static_ramp -6.9k, 95% CI [-8.5k, -5.2k], 0/10).

Enforcement: every agent and comparator entry needs `--amendment-id R-20261017-014`; `train_eval.py` checks that this
id is a request with state `approved` in the lab database (`~/qrl-lab/.lab/lab.db`, opened read-only, path from
`tools/labdb.db_path`; a syntax check alone is not enough) and records the row in `rev1_authorization.json`. Frozen
runs ignore the test-only `QRL_LAB_DB` override.

## Design (nothing below may change after approval)
- Simulator: Sim v2.1 defaults (provisional, uncalibrated), workload W0, 5000-step episodes. Agents train on the default
  config only.
- Agents: `dinos_t` (LSTM initialised from `modelCPU_5.pth`), `lstm_scratch`, `derp`; 3 training seeds each (0, 1, 2;
  episode seeds drawn from the train range 0-999), 100,000 environment steps, hyper-parameters as in `train_eval.py`
  and the agent defaults, unchanged. Agents act with +-1 pod per step (3 actions).
- Checkpoint rule (frozen): the LAST agent state at step 100,000, greedy (epsilon 0). No checkpoint, seed or
  hyper-parameter selection on any evaluation or validation episode.
- Comparators (nothing re-selected on evaluation data):
  - primary, deployable: `static_ramp_best` (c = 22 from the rev-1 run, sha256-pinned) and the preprovisioned
    `fixed_best` (c = 22 from the original preflight, sha256-pinned);
  - disclosed: `threshold` 40/25/0 (original preflight) and `k8s_hpa` (K8sHPA, target utilization 40 %);
  - reference: `always_min`, `always_max`, and `oracle` replay = **clairvoyant +-1 reference** (re-solved per config;
    not deployable). It is NOT an upper bound on the multi-pod K8sHPA (or any multi-pod policy).
- K8sHPA: `sim_v2/policies.py`, from docs/hpa_spec.md (kubernetes v1.34.0); tolerance 0.1, scale-up
  max(100 %, 4 pods) per 15 s without stabilization, scale-down 100 % per 15 s with a 300 s (20-step) stabilization
  window on the stored raw recommendations, 1..30 replicas. Only the target utilization is tuned: grid 10..95 step 5,
  mean return on development seeds 1000-1004, default config, ties -> smallest target. Recorded in
  `hpa_selection_dev.json` (`select_hpa_dev.py`): **40 %**, interior. The comparator run re-derives it and refuses if it
  differs. Its metric is `cpu_util_pct` = served / capacity (CPU usage over a request equal to the per-pod capacity),
  unclipped, a busy-fraction proxy: served / effective capacity, saturating at 100 % before noise/offset.
  Request = limit is an ASSUMPTION; the proxy cannot distinguish moderate from extreme overload (at target 40 % a saturated reading requests about 2.5x the ready replicas, whatever the real backlog).
  Requests do NOT shrink during rebalancing or with theta < 1: the proxy's denominator is the reduced effective capacity, so it reads higher than usage over a fixed request would.
  A real request < limit needs a calibrated CPU usage/request model (future work). Further fidelity gaps: `sim_v2/README.md` ("HPA baseline").
- **Disclosed asymmetry**: K8sHPA may change many pods in one step (absolute replica target); the agents and every other
  comparator are limited to +-1 pod per step. The agents are not changed. This favours K8sHPA and is reported with every
  K8sHPA number.
- Evaluation block: seeds **1300-1319** (n = 20, development/tune range), never used before (see Seed audit).
  Every method is evaluated on the same 20 episodes (paired), under the default config (primary) and every frozen
  sensitivity config of `sim_v2/sensitivity_v21.json` (descriptive only).

## Analysis (`analyze_explore_w0.py`, frozen)
- Agent return per evaluation episode = mean over its 3 training seeds; per-seed results are reported too.
- Paired differences (method - static_ramp_best) and (method - fixed_best) for each agent, threshold and k8s_hpa:
  mean, wins out of 20, 95% percentile bootstrap CI over episodes (10,000 resamples, numpy `default_rng` seed 20261017).
  The CIs resample evaluation episodes only: they are CONDITIONAL on the three trained models and omit the
  uncertainty over the population of training seeds.
- Default config is primary; sensitivity configs are descriptive. There is no decision rule: nothing is accepted or
  rejected, and the report is labelled EXPLORATORY. Unfrozen (smoke) runs and incomplete sets are refused
  (`--allow-incomplete` labels the report INCOMPLETE).
- Allowed wording: "in this uncalibrated simulator, on W0, method X had a mean paired return difference of ... vs ...".
  Not allowed: simulator validity, superiority claims, anything about quantum models or real clusters.

## Seed audit
- 1300-1319: no literal in any code, test, config or document of `code/` (grep, also over the full git history of
  `sim_v2/` and `experiments/dinos_simv2/`), no seed in any run output under `~/qrl-lab/.lab/runs` (grep of
  `metrics.jsonl` and JSON outputs; the calibration runs use seeds about 100, unrelated to Sim W0), and no code path
  iterates over a range containing them (selections use 1000-1004 only; train seeds come from 0-999; the Sim v2 gate
  suite uses 1000-1004 and held-out 10000+). Smokes before this registration used 1000-1004 only.
- Guards: `train_eval.guard_explore_block` refuses the block unless the run is frozen (`--freeze`); smokes may override
  the block only with seeds 1000-1004 (`--explore-eval-seeds`, refused under `--freeze`); the rev-1 gate block, the
  burned 1100-1109, the held-out 10000-10999 and the final test 20000-20999 are refused in every mode. Both test
  `conftest.py` files raise on any simulator reset with a protected seed.
- Final-test seeds 20000-20999 stay untouched and reserved for the calibrated study.

## Cost (measured smoke step times, single thread, this VM)
Training about 0.019-0.026 s per step (100k steps: 32-43 min); greedy evaluation 5.6-10 s per 5000-step episode
(220 episodes = 11 configs x 20: 21-37 min); per agent entry 63-69 min alone (limit 2 h per entry to allow for five-way contention). Comparator entry about 15 min.
Total about 10 CPU-h; with max_parallel 5 about 2.3 h wall clock.
