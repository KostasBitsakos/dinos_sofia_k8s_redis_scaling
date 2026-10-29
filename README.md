# DInos autoscaler — Kubernetes + Redis, Deep RL (QRL-Lab)

Research code and results for a Deep-RL horizontal autoscaler (DInos) for a Redis workload on Kubernetes,
and its planned hybrid quantum–classical extension (PhD work of Kostas Bitsakos). All results so far are from
**simulation**; the real-cluster calibration is paused.

## Branches
| Branch | Content |
|---|---|
| `main` | this README, cycle-1 results, lessons, locked cycle-1 checkpoints, cycle-2 plan |
| `cycle1-claude` | research code of cycle 1, Claude branch (`dinos-tuning`), snapshot without run logs (`*.log`, `real/experiments/lstmtransfer/log*`) and live-object dumps |
| `cycle1-codex` | research code of cycle 1, Codex branch (`dinos-tuning-astra`), snapshot without run logs (`*.log`, `real/experiments/lstmtransfer/log*`) and live-object dumps |

## Cycle 1 (19–24/10/2026): DInos tuning loop in simulation
Goal: reproduce the DInos picture with a fair protocol (same target training budget of 20k ticks, validation-only
model selection, single-use test splits, 5 seeds, paired traces). Pretraining recipes of the transfer arms are
asymmetric (V.2 20k with source-checkpoint selection, LSTM+transfer 10k without); this is disclosed in
`results/cycle1/c24_freeze.md`. Frozen recipe and protocol: `results/cycle1/c24_freeze.md`.
Final confirmation on the single-use split test11 (tag c24c, seeds 80–84): `results/cycle1/c24_test11_result.md`;
earlier split test10: `results/cycle1/c21_test10_result.md`. Per-seed numbers: `results/cycle1/per_seed_test1*.json`;
full summary: `results/cycle1/c24c_summary.json`; figures: `results/cycle1/figures/`.
The cycle-1 numbers are **not** targets for the paper; what we keep is in `LESSONS.md`.

## Models (`models/cycle1/`)
PyTorch state files of the locked checkpoint (across-seed rule: the checkpoint with the largest mean validation
return over the training seeds) for the four arms that saved weights, one file per seed 80–84:
| File prefix | Arm | Locked checkpoint |
|---|---|---|
| `spike300_d5-dinos_a1h-…` | Dinos (LSTM only in the first layer, n-step 3) | 20k ticks |
| `spike300_d5-transfer_a1h-…` | Dinos+Transfer V.2 | 10k ticks (fine-tuning) |
| `spike300_d5-mlp_a1h-…` | parameter-matched MLP control | 20k ticks |
| `spike300_d5-transfer_mlp_a1h-…` | parameter-matched MLP + transfer control | 10k ticks |

LSTM, LSTM+transfer, MLP(hist10), DERP and enhanced DERP ran through the legacy runner without weight export,
so no checkpoints exist for them. Load with `torch.load(path, map_location="cpu")`; the network definitions are in
`tuning/` of the `cycle1-*` branches.

## How to run (cycle-1 code)
Check out `cycle1-claude`, create a Python venv with PyTorch, then e.g.
`python tuning/legacy/run_legacy.py --agent lstm_scratch --agent-variant a1h --env-family orig --orig-variant spike300_d5 --n-step 3 --seed 80 --eval-split val --out out.json`
(`--help` lists all options; the exact commands of each confirmation run are recorded in the result files).
Note: `tuning/legacy/run_legacy.py` imports the original agents from a hard-coded path
(`LEGACY_P = /home/ubuntu/simulations/redisk8s/timeseries/Pytorch`); point it to the `sim/timeseries/Pytorch`
copy in the branch (or your own checkout) before running elsewhere.

## Cycle 2 (planned): Poisson microservice workload
Plan: `docs/cycle2_plan.md`; background notes (RL evaluation, DRL tuning, Poisson workloads, autoscaling
baselines): `docs/expert/`. Nothing of cycle 2 has run yet.
