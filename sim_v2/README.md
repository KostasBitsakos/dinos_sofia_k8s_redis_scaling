# Sim v2 (provisional)

Transient fluid-queue autoscaling simulator for QRL-Lab (council C-20261016-01, option 1; approval R-20261016-013).
**Every parameter is provisional and unmeasured. Nothing computed with these defaults is a paper result.**

Files: `params.py` (SimParams), `env.py` (SimV2, ObsWindow, resolution check), `workloads.py` (W0–W3),
`policies.py` (baselines, tuner), `oracle.py` (certified bound), `stats.py`, `gates.py` + `run_gates.py`
(frozen gate suite), `gates_config.json`, `gates_spec.json` (spec template for tools/launch_run.py), `tests/`.

Tests: `cd code/sim_v2 && ../../.venv/bin/python -m pytest -q -s tests`.
Gate smoke test (meaningless verdict): `QRL_RUN_DIR=/tmp/x ../../.venv/bin/python run_gates.py --quick`.
The real gate batch runs only through `tools/launch_run.py` with an approved spec.

## Model

Control step T = 15 s, internal sub-step dt (default 1 s), episode 5000 steps.
Actions: 0 = down, 1 = no-op, 2 = up (DInos); with `max_delta=2` also 3 = down 2, 4 = up 2.

**Pods.** Ready pods r and a pending pipeline. Per step: (1) pods whose delay elapsed become ready,
(2) the action changes the total n = r + pending, clipped to [1, 30]; an up adds a pending pod that serves after
d steps; a down cancels the newest pending pod first, otherwise removes a ready pod immediately,
(3) rebalance: after an accepted change (`rebalance_trigger="action"`; or `"topology"` = when the ready set
changes) capacity is reduced for k steps. Billed pods = ready + pending after the action.

**Capacity.** C = μ r^θ (1 − δ·1[rebalancing]); per-pod rate μ_pod = C / r.

**Queue.** Arrivals λ_t (ops/s, from the workload trace, constant within the step) enter a FIFO list of
cohorts, one per sub-step, stamped with the sub-step start. Each sub-step: cohorts with age ≥ timeout fail;
the new cohort a = λ dt joins; capacity C dt serves cohorts oldest first. Utilisation ρ = served / (C dt).
Latency of work served with wait A is A + S, where S is the per-pod M/M/1 sojourn, P(S > x) = exp(−μ_pod(1−ρ)x).
A request is *bad* if it fails (timeout, or still queued at episode end: settlement) or is served with
latency > SLO: fraction exp(−μ_pod(1−ρ)·max(0, SLO − A)). Each request is counted once.
Fast path: with no backlog and λ ≤ C the step is solved in closed form (identical at every dt).
p95 latency of a step: 95th percentile of the served-volume mixture of A + S.
Conservation: arrived = served + failed + outstanding (asserted every step with `debug=True`).

**Reward.** r_t = −α·(billed pod-seconds / T) − β·(bad requests / (μ T)) − γ·|accepted change|.
`info` returns `reward_cost`, `reward_sla`, `reward_churn` (sum = reward) and the raw counts.
Typical episode returns: −1e5…−1.5e5 for reasonable policies, ~−3e6 for always-min.

**Observations.** The default observation contains only what can be instrumented on the real cluster
(`env.OBS_NAMES`):

| index | name | meaning | instrumentation needed on the cluster |
|---|---|---|---|
| 0 | `completed_ops` | requests completed per s in the step (served / 15 s) | Redis `INFO stats` `total_commands_processed` delta per pod, or the load generator's completed counter |
| 1 | `cpu_pct` | 100·served/C + N(0, 2) | cAdvisor / metrics-server CPU of the Redis pods relative to their limit (Prometheus) |
| 2 | `p95_ms` | p95 latency of completed requests × lognormal(0.05) | client-side latency histogram in the load generator (not server-side `latency` stats) |
| 3 | `failed_per_s` | timed-out or rejected requests per s | client-side timeout/error counter in the load generator |
| 4 | `ready` | ready pods | kube-state-metrics / `kubectl get pods` ready condition |
| 5 | `pending` | requested but not ready pods | Deployment/StatefulSet desired minus ready replicas |

`SimV2(..., oracle_obs=True)` appends `offered_lambda` (exact offered traffic) and `backlog` (simulator queue).
These are simulator internals, not deployable telemetry; off by default, for diagnostics and oracle studies only.
Memory was removed from the observation: in the simulator it could only encode the backlog.
`ObsWindow(10, len(env.obs_names))` stacks the last 10 observations for LSTM agents.

**SLO.** Default SLO = 2 × the p95 sojourn of one M/M/1 pod at ρ = 0.5 = 4 ln 20 / μ = 11.98 ms.
(The brief said "2× the service time at ρ = 0.5"; this reading — p95 response time at ρ = 0.5 — is an
interpretation to confirm.)

## Sim v2.1 sensitivity knobs (R-20261017-001; defaults reproduce Sim v2 bit for bit)
Frozen one-at-a-time configs in `sensitivity_v21.json` (loader `sensitivity.py`), each value with its source or the
label PROVISIONAL (+-50% design rule where the literature gives no range). Agents train on `default`; the rest are
evaluation only.
- `rebalance_steps` k, `rebalance_loss` delta: existing fields, now varied (k=14 legacy migration, delta 0.05/0.15/0.45).
- `latency_add_s`: constant added to every latency (observed p95 and the SLO test). `tail_scale` kappa >= 1 multiplies
  the sojourn (age + M/M/1 part). A request is bad iff add + kappa (A + S) > SLO, i.e. the plain test with
  `SimParams.slo_eff_s = (SLO - add) / kappa`; env and oracle both use `slo_eff_s`, so the oracle bound is unchanged in form.
- `cpu_offset_pct`: added to the OBSERVED cpu_pct only (then noise, clip to [0,100]); capacity is unchanged. Chosen
  because the design has no model of what consumes the CPU; a capacity-stealing background load needs a decision.
- theta in {0.8, 0.9, 1}.

## Oracle bound (oracle.py)
Over an episode, bad = Σ_sub-steps [(a − f) + f·v(ρ)] with f the fresh work served in its own sub-step and
v(ρ) = exp(−m(1 − ρ)), m = μ_pod·SLO, because carried work is bad exactly once (needs dt ≥ SLO).
Relaxation: backlog is discarded for free once charged (it never steals capacity), and the server may refuse
fresh work when cheaper, so bad ≥ Σ h(λ, C), h = min_{f ≤ min(λ, C)} (λ − f) + f·exp(−m(1 − f/C)). Billing and
churn are exact. A DP over (ready, pending pipeline, rebalance counter) with perfect λ maximises the relaxed
return, so it is ≥ the return of every action sequence, hence of every policy, on that trace.
It is loose under overload (tiny test: bound −354 vs brute-force optimum −960), tight when no backlog forms.
`OracleReplay` replays the DP actions in the true dynamics: an achievable clairvoyant return.

## Workloads (workloads.py)
Rates in ops/s per 15 s step. Seeds: train 0–999, tune 1000–1999, held-out 10000–10999 (burned: seen by tests, gates and smokes), test 20000–20999 (fresh, final evaluation only) (`check_seed`).
- W0: legacy sine (period 1000/2π steps), noise, one oscillating spike; legacy units × 100 ops/s, peak
  20000 ops/s ≈ 25 pods at ρ = 0.8. Phase and spike position random per seed.
- W1: NHPP with a two-peak daily profile (2000–12000 ops/s, 1 day = 5760 steps) × 2-state MMPP bursts (×1.6).
- W3: W0-like regime, then at 40–60 % of the episode a slower, higher regime with bursts.
- W2: CSV replay stub (`W2(path, column, peak_ops)`), no download. The raw series is split chronologically
  (60 % train, 20 % tune, 20 % held-out) before any scaling or windowing; scaling uses the train max only; a seed
  reads windows only from the segment of its own seed range. An ambiguous numeric column raises an error.
- CONST: constant-load control (not in the superiority gate).

## Policies
always_min, always_max, threshold(up, down, cooldown) on observed CPU (tuned on tune seeds), and two static
families: **fixed(c)** starts the episode with c ready pods and never scales (true fixed allocation), and
**static-ramp(c)** starts at `init_pods` = 6 and ramps to c (pays churn, rebalance and startup). `OracleReplay`
replays the oracle DP actions (clairvoyant, not deployable).

## HPA baseline (K8sHPA, R-20261017-014 + researcher request M-20261017-068)

`policies.K8sHPA` re-implements the Kubernetes HPA controller (autoscaling/v2, one CPU-utilization metric, default
`behavior`) step by step from `docs/hpa_spec.md` (kubernetes v1.34.0, `pkg/controller/podautoscaler`); every block in
the code cites the spec section. `hpa_replica_calc` is `GetResourceReplicas` (integer-truncated utilization, ratio,
tolerance 0.1 both ways, unready pods 0 % on scale-up, missing-metric fallback 100 %/0 %, direction-flip recheck,
wrong-direction guard). `K8sHPA._normalize` is `normalizeDesiredReplicasWithBehaviors`: stabilization over the stored
RAW recommendations (scale-up window 0, scale-down 300 s = 20 syncs, seeded with the current replicas on the first
sync), then the rate limits (scale-up max(100 %, 4 pods) per 15 s, scale-down 100 % per 15 s, selectPolicy Max),
bounded by min 1 / max 30 replicas. Only the target utilization is tuned (`tune_hpa`, grid `HPA_TARGET_GRID`
10..95 step 5, mean return on tune seeds, ties -> smallest target). Tests: `tests/test_hpa.py`.

**Env interface (both default OFF; default trajectories stay bit-identical, `test_v21_defaults_bit_identical_to_sim_v2`).**
`SimV2(..., replica_actions=True)`: the action is the absolute desired replica count (ready + pending), any size in
one step, range-checked. `SimV2(..., util_obs=True)`: appends `cpu_util_pct` = 100 * served / capacity +
cpu_offset_pct + noise, the SAME random draw as `cpu_pct` but not clipped to [0, 100] (NaN at reset: no metric window
yet). It consumes no extra random numbers, so all other observation fields are unchanged.

**Utilization choice: served / capacity (a busy-fraction proxy), not arrivals / capacity.** See the Fidelity row "Request / metric": request = limit is an assumption and requests do not shrink in reality. metrics-server reports CPU *usage*, i.e. work the
pods actually did; with request = per-pod capacity, usage / request = served / capacity (the busy fraction of the ready
pods, the same quantity `cpu_pct` already reports). Arrivals / capacity is offered load, which no CPU metric measures
(it would need a request-rate custom metric) and would give the HPA information a real CPU-based HPA does not have.
Consequence: under overload the ratio saturates at 100 / target (2.5x at target 40 %), as for a real pod whose CPU
limit equals its request. The unclipped field only removes the clip of the noisy reading at 0 and 100 %, which would
bias the mean below 100 % at saturation. **Agents do not see it**: the DInos adapter keeps the 6 default slots, and
agents are trained and evaluated with `util_obs` off; they see the clipped `cpu_pct`, which carries the same reading.

### Fidelity: what still differs from the real HPA
| Item | Real HPA (v1.34.0, hpa_spec.md) | Sim v2 K8sHPA | Effect / note |
|---|---|---|---|
| Sync period | 15 s, wall clock with jitter (work queue) | exactly 1 control step = 15 s | windows counted in syncs: item at sync s is in a W-sync window at t iff s > t - W (exact `After(now - W)`); real jitter can include/exclude the boundary sample |
| Rate-limit period boundary | scale events within `periodSeconds` (15 s) counted by timestamp | events of the current sync only (periodStart = current) | boundary at exactly 15 s not verified (spec open item) |
| Metric source / window | metrics-server, scrape resolution and `metric.Window` (value not found) | mean busy fraction of the previous 15 s step, read at the next sync | a fixed one-step lag; no extra metrics-server delay modelled (unknown) |
| Request / metric | CPU usage / request per container; request set by the user, fixed | `cpu_util_pct` = busy-fraction proxy: served / effective capacity, saturating at 100 % before noise/offset. Request = limit is an ASSUMPTION; the proxy cannot distinguish moderate from extreme overload (at target 40 % a saturated reading requests about 2.5x the ready replicas, whatever the real backlog). Requests do NOT shrink during rebalancing or with theta < 1: the proxy's denominator is the reduced effective capacity, so it reads higher than usage over a fixed request would. A real request < limit needs a calibrated CPU usage/request model (future work) |
| Readiness | Ready condition; `cpu-initialization-period` 5 min, `initial-readiness-delay` 30 s, first-sample skip | pending (warming-up) pods = unready; a pod is ready exactly d steps after the scale-up | the post-readiness metric skip and never-ready pods are not modelled |
| Missing metrics | 100 % / 0 % fallback | implemented in `hpa_replica_calc`, never triggered (every ready pod has a metric) | |
| Per-pod spread | per-pod usage summed | only the mean is observed (all pods equally loaded) | identical for the ratio because requests are equal |
| Noise | metrics-server sampling | Gaussian sd 2 % on the mean (`cpu_noise_pct`) | |
| Rebalance capacity loss | not part of Kubernetes (Redis slot migration) | capacity x (1 - delta) for k steps after any change; it lowers capacity, so the busy fraction rises | HPA reacts to it like to load |
| Pod removal | graceful termination, pod keeps serving until gone | pending pods cancelled first, ready pods removed instantly | |
| Scale-to-zero, multiple metrics, `behavior: nil` legacy path | supported | not used (min 1, one metric, explicit default behavior) | legacy cap max(2c, 4) differs slightly from the default policies |
| HPA history | lives across deployments | created at episode start with c0 = 6 (initial recommendation blocks scale-down for 20 syncs) | |
| Action size | multi-pod | multi-pod (`replica_actions`) | **asymmetry**: agents and other comparators are limited to +-1 pod per step |

## Gates (gates.py, frozen in gates_config.json)
- G1: conservation.
- G2: the tuned threshold beats each fixed(c) and each static-ramp(c), c ∈ [1, 30]: paired one-sided t over 10
  held-out seeds, Holm over all workload × family × c comparisons, α = 0.025 one-sided. The report labels the
  per-comparison lower bounds *unadjusted*; the decision uses the Holm-adjusted p-values.
- G3: the oracle bound for the episode's start state (init_pods, or c for fixed(c)) ≥ every return.
- G4: ranking of {always_min, always_max, fixed_best, static_ramp_best, threshold} over θ ∈ {0.8, 0.9, 1} × δ ×
  {0.5, 1, 1.5}, reported with paired 95 % CIs on every pairwise difference. Fails only on a significant reversal
  (a pair ordered with CI excluding 0 one way in one config and the other way in another). Two versions, both
  gated: *retuned* (selections redone per config) and *frozen* (the base-config selections evaluated unchanged).
- G5: always_min not optimal.

Oracle margins are reported, not gated. Selections use tune seeds 1000–1004 only.
The gate runs in two phases: base config first (its selections are frozen), then the sensitivity configs.

## Provisional parameters
| field | default | unit | source |
|---|---|---|---|
| `control_step_s` | 15.0 | s | DInos convention (code/sim/timeseries/Pytorch/lstm_transfer/environment.py); 15 s control step |
| `dt_s` | 1.0 | s | engineering choice for Sim v2 (R-20261016-013), not measured; internal fluid sub-step, convergence-checked against 15 s |
| `episode_steps` | 5000 | steps | DInos convention (code/sim/timeseries/Pytorch/lstm_transfer/environment.py); 5000-step episodes |
| `min_pods` | 1 | pods | task R-20261016-013: c in [1, 30] |
| `max_pods` | 30 | pods | task R-20261016-013: c in [1, 30]; claude-r2 (30 pods ~ 6 of 32 vCPU) |
| `init_pods` | 6 | pods | DInos convention (code/sim/timeseries/Pytorch/lstm_transfer/environment.py); legacy env starts at 6 VMs |
| `max_delta` | 1 | pods/step | engineering choice for Sim v2 (R-20261016-013), not measured; 1 = actions {0,1,2}; 2 adds 3 = -2 and 4 = +2 |
| `startup_delay_steps` | 2 | steps | council C-20261016-01 card (provisional starting value, not measured); d = 2 steps (30 s) |
| `mu_ops_per_pod` | 1000.0 | ops/s | engineering choice for Sim v2 (R-20261016-013), not measured; mu = 1000 ops/s per pod (Redis pod limited to 200m CPU) |
| `theta` | 1.0 |  | council C-20261016-01 card (provisional starting value, not measured); linear capacity, sensitivity at 0.8 and 0.9 |
| `rebalance_loss` | 0.3 | fraction | council C-20261016-01 card (provisional starting value, not measured); delta = 0.3 |
| `rebalance_steps` | 1 | steps | council C-20261016-01 card (provisional starting value, not measured); k = 1 step |
| `rebalance_trigger` | action |  | engineering choice for Sim v2 (R-20261016-013), not measured; 'action' = any accepted change (task text); 'topology' = when the ready set changes |
| `timeout_s` | 60.0 | s | engineering choice for Sim v2 (R-20261016-013), not measured; client timeout 60 s, timed-out requests are failures |
| `slo_latency_s` | None | s | engineering choice for Sim v2 (R-20261016-013), not measured; None -> 2 x p95 sojourn of one M/M/1 pod at rho = 0.5 (= 4 ln20 / mu, 11.98 ms at mu = 1000) |
| `alpha` | 1.0 |  | council C-20261016-01 card (provisional starting value, not measured); alpha = 1 per billed pod-step |
| `beta` | 100.0 |  | council C-20261016-01 card (provisional starting value, not measured); beta = 100 per normalised bad-request volume |
| `gamma` | 0.5 |  | council C-20261016-01 card (provisional starting value, not measured); gamma = 0.5 per pod added or removed |
| `cost_norm_s` | None | s | engineering choice for Sim v2 (R-20261016-013), not measured; None -> control_step_s (pod-seconds -> pod-steps) |
| `sla_norm_requests` | None | requests | engineering choice for Sim v2 (R-20261016-013), not measured; None -> mu * control_step_s (one pod-step of work) |
| `cpu_noise_pct` | 2.0 | % | engineering choice for Sim v2 (R-20261016-013), not measured; Gaussian sd on CPU % |
| `latency_noise_log_sd` | 0.05 |  | engineering choice for Sim v2 (R-20261016-013), not measured; lognormal sd on observed p95 |

`None` defaults are derived (see `params.py`).

## How calibration replaces them
The calibration pilot (separate approval, real cluster, c ∈ {1, 2, 4, 8, 16, 30}) should measure: μ and θ
(throughput at saturation per c), the latency-vs-ρ curve (replaces the M/M/1 tail; then `v` and the oracle's
`h` must use the fitted curve, and the bound proof needs v increasing in ρ), d (pod ready time), δ, k and whether
the loss starts at the action or when the pod joins (`rebalance_trigger`), the client timeout, and the client-side
latency and failure counters needed for the observations. The researcher sets SLO, α, β, γ before any agent is evaluated. Calibrated values go into a new
SimParams instance with `source` pointing to the run id; the gate suite is re-run unchanged.

## What the provisional simulator does NOT support
- Any absolute number (cost, latency, violations) or any claim about the real cluster.
- Claims above the measured pod range; capacity at 30 pods is extrapolated (θ sensitivity only).
- Millisecond latency tails beyond the assumed M/M/1-per-pod shape.
- Quantum vs classical conclusions (rankings may change after calibration), and any speed-up claim.
- Statements that a learner has (or lacks) room to improve: the oracle bound is loose under overload.

## Known findings with the provisional values (micro-benchmarks, not the gate run)
- With δ = 0.3 applied at every accepted action, frequent scaling is expensive (the 30 % capacity dip hits the
  steep SLO tail); on single seeds a tuned threshold did not beat static-best on W0/W3; with δ = 0 it did.
  The gate batch will measure this properly; δ, k and the trigger are calibration priorities.
- Resolution: on an overloaded 300-step W0 case, dt = 15 s vs 1 s changes the return by 0.5 % and failures by
  1.2 % (`test_resolution_convergence_helper`).
