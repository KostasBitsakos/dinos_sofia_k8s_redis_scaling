# Lessons from cycle 1 (DInos tuning loop, simulation, 19–24/10/2026)

The cycle-1 numbers are not paper targets; these are the lessons we carry into cycle 2. Result files are in
`results/cycle1/`; the full loop log and per-cycle notes stay in the lab's private workspace.

## Protocol
1. **Evaluate every policy greedily.** In the DERP evaluation code we inspected, DERP is evaluated at epsilon = 1
   (random actions). We have not verified that this code produced the published ~17x gap of DInos over DERP, so this
   is a likely explanation, not a proven one. With greedy evaluation the gap is much smaller: on test11, Dinos 88.0k vs
   DERP 62.2k and V.2 84.0k vs DERP 62.2k (`results/cycle1/c24_test11_result.md`).
2. **Select checkpoints on validation only, across seeds, and lock them before test.** Each model is tested at its
   locked checkpoint; test splits are single-use (test10 and test11 are spent).
3. **Pre-register hypotheses and test them in a fixed sequence.** On test11, V.2 > DERP (+21.8k) and V.2 > enhanced
   DERP (+15.4k) were confirmed; Dinos > LSTM (+0.7k, CI [−2.5k, +3.7k]) was not, so the sequence stopped
   (`results/cycle1/c24_test11_result.md`).
4. **Five seeds are not enough** to separate the top recurrent/transfer agents; differences of a few thousand return
   units had CIs crossing 0. Cycle 2 uses ≥10 seeds, paired traces, IQM + bootstrap CIs (`docs/expert/rl_evaluation.md`).
5. **Always include parameter-matched classical controls.** A parameter-matched MLP (`mlp_a1h`) had a higher
   observed test11 mean than Dinos (90.3k vs 88.0k), but the difference is inconclusive (Dinos − mlp_a1h −2.2k,
   CI [−6.6k, +2.1k], `results/cycle1/c24_test11_result.md`). Architecture claims need these controls.
6. **Baselines must be meaningful.** The original HPA baseline behaves like a random walk (its CPU/memory inputs are
   random and independent of the VMs); report it as-is but add a load-based HPA tuned on validation.
7. **The D-step oracle is a look-ahead heuristic, not an optimal bound**; do not call it an upper bound.

## Training recipe (frozen in `results/cycle1/c24_freeze.md`)
- LSTM only in the first layer (variant a1h) and n-step 3 returns (both chosen on earlier, now-spent splits).
- Workload spike300_d5: the original environment plus a pod start-up delay of D = 5 ticks.

## Dead ends (do not repeat without a new reason)
- Simulator v2 as the primary env, the coupled env, start-up delays D = 15/20, noise15_multispike_d5,
  spike300_d7 / multispike_d5 as acceptance envs, noise for the B1 arm.
- Dueling heads, prioritized replay, architecture a12 and the c23 package (each scored lower on validation, or below
  the adoption rule, in our screens; the screen logs are not part of this export).

## Engineering
- Wall-clock timings from simulators prove nothing about speed; quantum parameter savings are claimed only against a
  parameter-matched classical control.
- The simulator is not yet calibrated against the real cluster (the calibration sweep is paused).
