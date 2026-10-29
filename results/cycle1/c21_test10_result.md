# test10 confirmation — RESULT (2026-10-23 04:40 UTC). Manifest 7f9face1, 46/46 runs done, seeds 60-64, spike300_d5, split test10 (single use)
Summary .lab/tuning/c18c_summary.json; convergence c18c_convergence.json; order check fair_progress.json "test10 confirmation".
| arm (n-step 3 unless noted) | locked ck | val | test10 | sd over seeds |
|---|---|---|---|---|
| LSTM+transfer | 20k | 90371 | 90635 | 3167 |
| MLP (mlp_hist10) | 20k | 90039 | 90186 | — |
| LSTM | 20k | 89770 | 89966 | 2874 |
| transfer_mlp_a1h (matched) | 20k | 89736 | 89832 | — |
| Dinos+Transfer V.2 a1h (PRIMARY) | 10k | 87599 | 87781 | 5126 |
| Dinos a1h | 20k | 85909 | 86148 | 11004 |
| mlp_a1h (matched) | 10k | 82154 | 82203 | — |
| enhanced DERP (no n-step) | 20k | 69705 | 69803 | — |
| DERP (no n-step) | 20k | 62907 | 62998 | — |
Baselines test10: oracle_Dstep 88620, causal_be 71184, fixed_best 60200, HPA-demand 47530, his HPA (hpa_rule) −885.
## Pre-registered fixed-sequence hypotheses (crossed 95% CI, test10)
- H1 V.2 > DERP: +24784 [+17854, +31755] → CONFIRMED
- H2 V.2 > enhanced DERP: +17979 [+9811, +29545] → CONFIRMED
- H3 Dinos > LSTM: −3818 [−14253, +3877] → NOT confirmed (testing stops here; H4-H6 descriptive)
- H4 V.2 − Dinos +1634 [−8556, +13538] · H5 V.2 − LSTM+transfer −2854 [−9869, +3039] · H6 Dinos − LSTM+transfer −4487 [−13022, +12]
Descriptive: V.2 − transfer_mlp −2050 [−6333, +2929]; Dinos − mlp_a1h +3945 [−6366, +11963]; LSTM+transfer − DERP +27637; LSTM − DERP +26968;
ratios V.2/DERP 1.39×, /enhanced 1.26×; V.2/oracle_Dstep 0.99; closeness 34% (descriptive). Order check 3/7 ✓.
Learning curve (descriptive, val by checkpoint): V.2 has the highest AUC (85.3k) and jumpstart at 1k (73.2k) and reaches 95% of its best at 5k
ticks; LSTM+transfer AUC 84.3k (95% at 10k); LSTM 69.5k; Dinos 66.4k; DERP 47.1k.
## Reading (revised after Codex review)
CONFIRMED (pre-registered H1-H2 only): Dinos+Transfer V.2 > DERP and > enhanced DERP on spike300_d5 (test10, 5 fresh seeds).
NOT confirmed: H3 Dinos > LSTM; testing stopped, so H4-H6 and everything else (LSTM/MLP/LSTM+transfer vs DERP, HPA rows, ratios, oracle,
closeness) are DESCRIPTIVE. The dev advantage of V.2/Dinos with n-step did not replicate; this is consistent with selection optimism (configs
selected on seeds 53-57) and seed variability (Dinos sd 11k, seed 62 −23.6k vs LSTM), but the failed replication does not isolate either cause.
Learning curve (exploratory, no uncertainty, source pretraining excluded): V.2 has the highest observed validation AUC and jumpstart and the
earliest own-best (95%) threshold crossing; this establishes neither overall training speed nor a confirmed sample-efficiency advantage.
Closeness 34% from closeness.py (descriptive). Review: Codex gpt-6.1-sol medium — numbers verified; wording revised as above.
