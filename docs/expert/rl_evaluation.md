---
description: Reliable deep-RL evaluation practice (rliable IQM, bootstrap CIs, seeds, single-use test) for cycle 2
---
# Reliable RL evaluation — what we do and why
## Sources
- Agarwal, Schwarzer, Castro, Courville, Bellemare, "Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021,
  arXiv 2108.13264; library github.com/google-research/rliable. (B; already cited as practice in c24, `c24_freeze.md`)
- Henderson, Islam, Bachman, Pineau, Precup, Meger, "Deep Reinforcement Learning that Matters", AAAI 2018,
  arXiv 1709.06560. (V: key `henderson2018drl` in `docs/related_work_sota_autoscaler.md:286`)
- Colas, Sigaud, Oudeyer, "How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments", arXiv 1806.08295. (B)
- Patterson, Neumann, White, White, "Empirical Design in Reinforcement Learning", arXiv 2304.01315 (JMLR 2024). (B)
## Rules
1. **Seeds.** ≥10 independent training seeds per arm (Henderson: 5 seeds can give opposite conclusions; our cycle 1
   used 5 and H3 failed with CI [−2.5k,+3.7k], `c24_test11_result.md`). Seeds fresh, listed in the freeze file.
2. **Splits.** train workload seeds ≠ val traces ≠ test traces. Checkpoint and hyperparameter choice on val only.
   Test split used once, after a written freeze (as `c24_freeze.md`); no reselection, no reruns, no seed replacement.
3. **Paired evaluation.** All arms see identical test traces (same arrival seeds); analyse per-trace paired differences.
4. **Aggregate.** Report mean and IQM over seeds (rliable), stratified bootstrap 95% CIs (≥2000 resamples),
   probability of improvement P(X>Y), and performance profiles if space allows. Point estimates alone are not claims.
5. **Hypotheses.** Pre-register an ordered list; fixed-sequence testing at α=0.05 (stop at first failure; rest descriptive),
   the procedure already used in c21/c24.
6. **Multiple metrics.** Return is the training objective; reviewers of autoscalers judge SLO violations, tail
   latency, pod-time (cost) and churn. Report all; a return win with worse SLO is not a win.
7. **Budget disclosure.** Report tuning budget per arm (configs × seeds × ticks, CPU-h) and transfer pretraining cost.
8. **Learning curves.** Val return at fixed checkpoints (1k,2k,5k,10k,20k ticks) → sample-efficiency claims
   (area under curve, ticks to reach X% of best) — this is where transfer can show a real advantage.
9. **Simulator honesty.** No claims about real-cluster behaviour from simulation; PLAN §2 F1 check (served load must
   depend on pods, else the policy cannot matter).
