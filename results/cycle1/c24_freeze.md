# FREEZE (c24, test11) — 2026-10-23 ~18:00 UTC; written before any test11 run
Basis: c23 package NOT ADOPTED (.lab/tuning/cycles/c23_screen.md, rule in .lab/council/C-20261023-01/card.md) → incumbent recipe of
c21_freeze.md unchanged: arch a1h, n-step 3 on every compatible learning arm, no transfer technique, transfer recipes as c21 amendment 1.
- Arms: Dinos+Transfer V.2 (PRIMARY) transfer_a1h+n3; Dinos dinos_a1h+n3; LSTM+transfer transfer_orig_orig_d0+n3; LSTM lstm_scratch+n3;
  MLP mlp_hist10+n3; matched controls mlp_a1h+n3, transfer_mlp_a1h+n3; DERP, enhanced DERP (derp_hist10) without n-step; baselines HPA-demand,
  his HPA (hpa_rule), fixed_best, oracle_Dstep (heuristic reference).
- Training seeds 80-84 (fresh; make_c18 --confirm checks). Split val + test11 (12000-12019), single use (per-split guard). n=5 as test10
  (budget: test10 used 58.0 CPU-h for 46 runs; v12 has 91.4 left).
- Checkpoint: across-seed locked rule on val, per arm (sum_c10). No reselection after test11.
- Inference (order from the council card C-20261023-01 line 14: H1/H2 vs DERP first, then H-a, H-b; extended after H-b with the
  two remaining c21 contrasts; differs from the c21/test10 order, disclosed): fixed sequence at α = 0.05 (crossed 95% CI lower bound > 0 on
  test11, sum_c10 2000 resamples); a claim is confirmed only if it and all earlier ones pass; testing stops at the first failure, the rest
  descriptive: H1 V.2 > DERP · H2 V.2 > enhanced DERP · H3 (H-a) Dinos > LSTM · H4 (H-b) V.2 > LSTM+transfer · H5 V.2 > Dinos ·
  H6 Dinos > LSTM+transfer. IQM reported alongside means.
- Failure handling as c21 amendment 2: no seed replacement, no reruns, no checkpoint fallback; incomplete arm → its hypotheses not confirmed.
- Relation to test10: test11 is a fresh replication of the same frozen recipe (test10 H1/H2 confirmed, H3 failed:
  c21_test10_result.md). Both are reported; neither is dropped.
- Explicit recipe (from the test10 entries c18c_entries.json; only seed/split/tag change): all learning arms --total-ticks or
  --finetune-ticks 20000, --checkpoints 1000,2000,5000,10000,20000, --eps-schedule linear10k, --reward-div 10, --obs-scale on, --threads 1,
  --n-step 3 (not DERP/enhanced DERP); optimiser = each arm's own default (no --lr, no --grad-clip), --pending-maturity off (7 inputs);
  matched MLPs = ordinary mlp_a1h / transfer_mlp_a1h / mlp_hist10 (no _pm). V.2 and transfer_mlp_a1h: pretrain orig_d0 20k,
  --source-checkpoints 1000,2000,5000,10000,20000 (val-best source), --finetune-eps restart. LSTM+transfer: pretrain orig_d0 10k, no source
  search (asymmetric recipes kept and disclosed, c21 amendment 1).
- Code: dinos-tuning 5eae972 (c23 options present but OFF: no --lr/--grad-clip/--pending-maturity flags in any entry).
Command:
  python3 .lab/tuning/make_c18.py --confirm --confirm-split test11 --arch-search --arches a1h --transfer-v2 a1h --tag c24c \
    --targets spike300_d5 --seeds 80,81,82,83,84 --ddqn n3 --techniques-on-existing --cpu-affinity 8-39 --min-wall-hours 5 \
    --out .lab/tuning/c24c_entries.json
  then .lab/tuning/drive_chunks.py .lab/tuning/c24c_entries.json c24c R-20261023-001 8-39 code
Report: sum_c10 --split test11, order_check.py --split test11, convergence.py; by 24/10 21:59 UTC.
Lab file hashes (sha256, 12 chars) at freeze:
d980af0c4322 .lab/tuning/make_c18.py
6c1ac3e8eb50 .lab/tuning/sum_c10.py   (state_size patch for c23 runs only; c24c runs have 7 inputs)
6bb7dc0f8253 .lab/tuning/order_check.py
9844de087a9f .lab/tuning/convergence.py
ac1ebb3517dc code-tuning/tuning/closeness.py
Generated 17:39 UTC: 46 entries (est. 50.5 CPU-h), seeds 80-84, splits val,test11, no c23 flags, n-step 3 on all learning arms except
DERP/enhanced DERP (checked by script). c24c_entries.json e1449d3ad207; manifest c24c_confirm_manifest.json sha256 2b15540df59e….
Codex review of c23_screen + this freeze (gpt-6.1-sol, thread 01a169d7): rejection correct, numbers match, sum_c10 patch correct (7+MATURITY_K 5);
asked for the explicit recipe and the card's hypothesis order → both added above before generation.
LAUNCHED 17:42 UTC on R-20261023-001 (driver .lab/tuning/drive_c24c.log; first chunk 12 runs, the driver adds the rest as the budget reservation frees).
