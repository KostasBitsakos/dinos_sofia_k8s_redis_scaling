# test11 result (c24c) — 2026-10-23 22:10 UTC; frozen recipe .lab/tuning/cycles/c24_freeze.md; single use of test11 (12000-12019)
Runs: 46/46 exit 0 (launch_run.py list, tag c24c, R-20261023-001). Seeds 80-84, n=5. Summary .lab/tuning/c24c_summary.json (sum_c10 --split
test11), order .lab/tuning/c24c_order.json, convergence .lab/tuning/c24c_convergence.json.

Means on test11 (locked ck): V.2 83983 (10k) · Dinos 88026 (20k) · LSTM+transfer 89331 (20k) · LSTM 87347 (20k) · MLP 88923 · mlp_a1h 90254 ·
transfer_mlp_a1h 88575 · enhanced DERP 68592 · DERP 62228 · HPA-demand 47572 · his HPA 249 · fixed_best 60216 · oracle_Dstep 88822 (heuristic).

## Fixed sequence (crossed 95% CI on test11)
| H | contrast | diff | CI95 crossed | result |
|---|---|---|---|---|
| H1 | V.2 > DERP | +21755 | [+10493, +33061] | CONFIRMED |
| H2 | V.2 > enhanced DERP | +15391 | [+9745, +21728] | CONFIRMED |
| H3 | Dinos > LSTM | +679 | [-2478, +3709] | NOT CONFIRMED → testing stops |
| H4 | V.2 > LSTM+transfer | -5348 | [-10405, -245] | descriptive (opposite direction, CI excludes 0) |
| H5 | V.2 > Dinos | -4043 | [-10009, +2191] | descriptive |
| H6 | Dinos > LSTM+transfer | -1305 | [-8166, +5802] | descriptive |
Descriptive: Dinos − mlp_a1h -2228 [-6565, +2086]; V.2 − transfer_mlp_a1h -4592 [-8222, -1395]; MLP − DERP +26695 [+17511, +35059].
Convergence (descriptive): transfer arms reach 90% of their OWN best val by 5k target ticks; scratch arms at 10k-20k (source pretraining excluded).
IQM (5 seeds x 20 traces, .lab/tuning/c24c_iqm.txt): V.2 84082 · Dinos 88949 · LSTM 87617 · LSTM+transfer 90309 · mlp_a1h 90186 ·
transfer_mlp_a1h 88730 · MLP 89216 · enhanced DERP 69907 · DERP 61752 (same ordering picture as the means).
Arm sd over seeds (deep arms): 2.2k-5.7k (incl. scratch MLPs).

## Reading (revised after Codex review, thread 01a16acd, before any posting)
- Confirmed (frozen sequence): H1, H2 — the deep agents (V.2) beat DERP and enhanced DERP. Same as test10 (c21_test10_result.md).
- The complete ordering among the deep agents (V.2, Dinos, LSTM, LSTM+transfer, the MLPs) is unresolved at n=5. Several descriptive
  intervals exclude 0 and go AGAINST V.2: V.2 < LSTM+transfer, V.2 < transfer_mlp_a1h; both scratch MLPs are also above V.2 descriptively.
- "V.2 is best" is not established on test11, nor was it on test10: continued failure to establish superiority. V.2's 10k checkpoint is the
  across-seed val maximum of a run trained to 20k — not truncation, and no reselection is allowed.
- Transfer arms reach their own-best val thresholds earlier during target training (t90 5k vs 10-20k, both test10 and test11). Thresholds differ
  per arm, no uncertainty, pretraining excluded: total training efficiency is NOT established.
- Paper: test10 and test11 reported as separate confirmations with full disclosure (seeds 60-64 vs 80-84, distinct test traces, val-selected
  checkpoints, changed hypothesis order, development in between, asymmetric pretraining recipes). Any pooled analysis = exploratory, keeps
  batch/seed/trace structure, cannot revive hypotheses blocked by either stopping rule.
- order_check (c24c_order.json, 3/7) uses a different ordering and is not the frozen test; its "test9 dev" note is stale.
