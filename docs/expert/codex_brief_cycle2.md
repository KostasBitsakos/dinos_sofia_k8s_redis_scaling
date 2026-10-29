# Blind question to Codex — cycle 2 (Poisson) design (J12, R-20261028-001, M-20261029-017)
Answer independently (do not look for a Claude plan; there is none in the repo yet). English, structured, <400 words.

Context: DInos autoscaler study in simulation (his env + pod delay D=5, spike300_d5 workload; cycle-1 results in
`.lab/tuning/cycles/c24_test11_result.md`, recipe `c24_freeze.md`). Researcher drops cycle-1 numbers as targets, keeps lessons.
Cycle 2: all our models (LSTM, LSTM+transfer, Dinos, Dinos+Transfer V.2, DERP, enhanced DERP, HPA) + an adapted
published baseline; NEW noisy Poisson microservice workload (correct process) + env exposing latency/CPU/memory;
Claude and Codex tune on separate branches; deadline 2/11/2026 (4 days from now); CPU budget 160 CPU-h / 7 days
(R-20261029-001); cycle-1 cost ≈1.26 CPU-h per 20k-tick learning run (test10: 58 CPU-h / 46 runs). Researcher wants
order V.2 > LSTM+transfer > Dinos > LSTM > paper model > DERP > HPA; we may tune anything but fairly.

Questions:
1. External baseline for CCGrid-type reviewers: DRe-SCale LSTM-PPO only, or also Gwydion (JNCA 2025, 104067)? Why?
2. Workload: exact arrival model (homogeneous Poisson / NHPP via thinning / MMPP / mixture), noise, and how to make
   val/test traces and a held-out "shifted" workload.
3. Env: minimal queueing model for latency, CPU, memory; reward; what must be common to all agents (obs window,
   action set, delay, horizon).
4. Tuning budget allocation that is equal across arms and fits 160 CPU-h with ≥10 seeds on a single-use test split;
   how to merge the two branches' candidates without test leakage.
5. Statistics/reporting (metrics, hypotheses order, CIs) and the main risk of the whole cycle + a falsifying check.
