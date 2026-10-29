---
description: Cycle 2 (Poisson microservice workload) plan - env, workload, models, equal tuning budget, seeds, splits, metrics, branches, timeline to 2/11
---
# Cycle 2 plan — Poisson microservice workload (J12, 2026-10-29; R-20261028-001, M-20261029-017)
Status: PHASE A (design) — no runs. Needs the researcher's approval of §2-§6 (spec) before phase B code and any run
(launcher: approved frozen spec). Codex blind review: PENDING (peer call failed: Codex model `gpt-5.1-codex-max` is not
supported on a ChatGPT account, `/tmp/j12_peer2.out`; brief kept in `docs/expert/codex_brief_cycle2.md`).
Expert background: `docs/expert/`; skills: `rl-eval-protocol`, `drl-tuning`, `poisson-workload`.

## 1. Goal and claims
- Compare all our agents + one adapted published baseline on a new, noisy, correctly generated Poisson microservice
  workload with latency/CPU/memory signals. Cycle-1 numbers are not targets (M-20261029-017); its lessons are kept
  (n-step 3, reward-div 10, obs-scale, ≥10 seeds, single-use test, fixed-sequence tests: `c24_freeze.md`, STATE §5-6).
- Researcher's desired order: V.2 > LSTM+transfer > Dinos > LSTM > paper model > DERP > HPA. We test it; we do not
  engineer it. If V.2 does not win on return, the pre-registered secondary questions (§6 H-noise, H-shift, H-sample)
  look for its real advantage; reported either way.

## 2. Env + workload (one shared, frozen implementation for both branches)
- Base: his env + pod start delay D=5 (cycle-1 env), action set and horizon unchanged unless both branches agree in
  phase B; obs window identical for all arms (10 steps, as cycle 1; DRe-SCale also run with its native 1-step obs as
  a disclosed variant only if budget allows).
- Workload `poisson_ms` (spec in `docs/expert/poisson_workloads.md`): per tick N_t ~ Poisson(Λ_t);
  λ(t) = base · diurnal sinusoid · MMPP-2 burst factor · exp(AR(1) log-noise). Request-level path with exponential
  inter-arrivals for latency. Parameters (base, amplitude, burst rates/sojourns, noise σ) chosen in phase B so that the
  peak needs ≈ max pods and no fixed pod count is near-optimal (checked by fixed_best and oracle_Dstep on val).
- Shifted workload `poisson_ms_shift` (held out, for H-shift): different burst rate and noise σ; used only on its own
  val/test traces, never in tuning.
- Microservice model: pods × c workers × μ; bounded queue; drops; mean/p95 latency; CPU util = served/capacity + noise;
  memory = base/pod + k·in-flight. Obs adds latency p95, CPU, memory to the current features (common to all arms).
- Reward (common): served-load term − pod cost − SLO-violation penalty (p95 > L_slo) − drop penalty. Weights fixed in
  phase B on a random/fixed policy only, before any agent tuning; reward-div 10 kept.
- Unit tests (skill `poisson-workload`): dispersion, KS vs Exponential, determinism per seed, F1 (served/latency
  depend on pods). Codex reviews env code before any tuning run.
- Owner: Claude implements env+workload on `dinos-poisson`; Codex reviews; the merged commit is the frozen env for both.

## 3. Models (arms)
LSTM · LSTM+transfer · Dinos · Dinos+Transfer V.2 (primary) · DERP (his original) · enhanced DERP (derp_hist10) ·
**adapted DRe-SCale LSTM-PPO** (RecurrentPPO, LSTM-256, actor/critic [64,64], `docs/expert/autoscaling_baselines.md`) ·
his HPA (as-is) · HPA-demand tuned on val · references fixed_best, oracle_Dstep (heuristic, not a bound).
Matched MLP controls (mlp_a1h, transfer_mlp_a1h) kept only if budget allows (they matter for any "architecture" claim).
Gwydion: cited and discussed, not ported in this cycle (reasons: `autoscaling_baselines.md`). Transfer source for
transfer arms: cycle-1 env spike300_d5 (pretraining ticks charged to the arm).

## 4. Seeds and splits
- Tuning seeds: 3 per config (fresh, e.g. 100-102). Final training seeds: 10 (e.g. 110-119). Exact values checked
  unused by `make_c18 --confirm` in phase B.
- Traces: train = per-episode random seeds; val = 20 fixed traces; **test12** = 20 fixed traces, single use, paired
  across arms; `poisson_ms_shift` val/test = 10 + 20 traces. Trace ids disjoint from 12000-12019 (test11) and earlier.

## 5. Tuning budget (equal per arm) and CPU-h
- Budget 160 CPU-h / 7 days (R-20261029-001). Cycle-1 cost 1.26 CPU-h per 20k-tick run (test10: 58.0 CPU-h / 46 runs,
  `c24_freeze.md`); the new env's cost is **unknown** → measured in the phase-B smoke run, then all counts below scale
  equally.
- Per arm per branch: 6 configs × 3 tuning seeds at 5k ticks (≈0.3 CPU-h each → ≈5.7 CPU-h) + top-2 × 3 seeds at 20k
  (≈7.6 CPU-h): ≈13 CPU-h/arm/branch is too much for 7 arms × 2 branches (≈180 CPU-h). **Adopted split:** each branch
  gets 6 learning arms' worth of **4 configs × 2 seeds × 5k + top-1 × 2 seeds × 20k ≈ 5.6 CPU-h/arm → ≈34 CPU-h/branch,
  68 CPU-h both.** Final: 7 learning arms × 10 seeds × 20k ≈ 88 CPU-h (+ transfer pretraining, + shift eval, eval-only
  baselines cheap). Total ≈ 156-160 CPU-h: at the limit. If the smoke run shows a higher cost, cut configs, not seeds.
  If it still does not fit, ask the researcher for more CPU-h (do not cut to <10 final seeds).
- DERP/enhanced DERP are tuned with the same trial count (their own knobs); HPA-demand threshold grid on val (cheap).
- Branch merge without leakage: for each arm, the val-best config of the two branches (val only, locked rule) goes
  to the final run; the choice is written in `cycle2_freeze.md` before test12.

## 6. Metrics and hypotheses
- Metrics per test trace: return; SLO-violation rate (ticks p95 > L_slo); p95 latency; pod-time (pod-ticks); churn
  (|Δpods| per tick); drops. Mean, IQM, stratified bootstrap 95% CI, P(improvement) (skill `rl-eval-protocol`).
- Fixed-sequence primary (α=0.05, stop at first failure): H1 V.2 > DRe-SCale · H2 V.2 > enhanced DERP · H3 V.2 > Dinos ·
  H4 V.2 > LSTM+transfer · H5 Dinos > LSTM. (Order proposed; council/Codex review before freeze.)
- Secondary, descriptive, pre-registered: H-noise (rank stability as noise σ grows, val), H-shift (V.2 vs others on
  `poisson_ms_shift`), H-sample (val return at 1k/2k/5k ticks: transfer sample efficiency, pretraining charged).

## 7. Branches and roles
- Claude: `dinos-poisson` (worktree `code-poisson`); Codex: `dinos-poisson-astra` (`code-poisson-astra`). Same frozen
  env commit; each branch tunes all arms independently within §5; daily report in the thread (M-20261029-006).
- Final run and analysis on a joint merge commit reviewed by both models.

## 8. Timeline (deadline 2/11)
- 29/10: phase A (this file) → researcher approval of spec + Codex review (once the Codex model config works).
- 30/10: phase B: env + workload + DRe-SCale port + tests (Claude), review (Codex); smoke run → cost per run; sandbox
  request (160 CPU-h) filed; tuning specs approved.
- 31/10: tuning on both branches (val only); daily report.
- 1/11: freeze (`cycle2_freeze.md`), final 10-seed run on test12 + shift traces.
- 2/11: analysis, figures, result md, report; lessons to `docs/paper_notes.md`.
Main risk: the env cost or a bug in the new env eats the 4 days. Mitigation: env tests + F1 check + smoke before any
tuning; a dead env → stop and report, no tuning on a broken env.
