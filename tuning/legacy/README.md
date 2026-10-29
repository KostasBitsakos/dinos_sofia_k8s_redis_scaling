# Track L: faithful port of the legacy DInos/DERP simulation

P = `~/simulations/redisk8s/timeseries/Pytorch` (original code, read-only, never modified).

## Files
| port | original | imported by (original scripts) |
|---|---|---|
| `env_dense.py` | `P/dense/environment.py` (= `P/HPA/environment.py`, byte-identical) | dense/doubledeep.py, doubledeepfinal.py; HPA/hpa_final.py |
| `env_dense2.py` | `P/dense/environment2.py` | dense/doubledeep2.py |
| `env_cpu.py` | `P/environmentCPU.py` | doubledeepCPU.py, doubledeepCPUfinal.py |
| `env_cpu2.py` | `P/environmentCPU2.py` (prints every step) | doubledeepCPU2.py |
| `env_transfer.py` (`--env lstm_transfer`) | `P/lstm_transfer/environment.py` | lstm_transfer/lstm_transfer.py |
| `network_dense.py`, `ddqnagent.py` | `P/dense/network.py`, `P/dense/ddqnagent.py` | DERP |
| `networks_lstm.py`, `agent_huber.py` | `P/networks.py`, `P/agentHuber.py` | doubledeepCPU*.py |
| `agent_huber_transfer.py` | `P/lstm_transfer/agentHuber.py` | lstm_transfer.py |
| `simpleagent.py` | `P/HPA/simpleagent.py` | HPA/hpa_final.py |
| `env_orig.py` (cycle 9) | `P/environment.py` | doubledeep3_pytorch.py, doubledeep4_pytorch.py, deeprl.py |

Each port is the original byte-for-byte plus a 4-line header (source path:lines, sha256). The only edits are the
import lines that are made package-relative. `port_sources.py` regenerates the ports, and `tests/test_legacy.py`
asserts byte equality with the originals. `P/environment2.py` is not ported: its `step()` returns a 3-tuple, and it
belongs to older scripts (doubledeep3-5_pytorch.py, deeprl.py). `P/environment.py` (also a 3-tuple) is ported as
`env_orig.py` for cycle 9 and used only through `orig_variants.py`.

**Per-agent defaults** (`TRAIN_ENV` / `EVAL_ENV` in run_legacy.py) are the environments of the paper's final
evaluation. They come from two independent provenance checks, `~/qrl-lab/.lab/tuning/deep/claude_opus_2.md` and
`~/qrl-lab/.lab/tuning/deep/codex_astra_2.md`, which agree and were relayed by the coordinator on 2026-10-19. The md5
values below were re-verified here.

| agent | train env | eval env | original weights |
|---|---|---|---|
| derp | dense/environment.py (md5 5ee774f0) | dense | dense/modelCPU_N.pth |
| lstm_scratch ("RS") | 5ee774f0 (= dense; at the time the file environmentCPU.py) | environmentCPU.py (a91f7d77) | P/modelCPU_N.pth |
| dinos_t | pretrain 5ee774f0 (dense), the RS weights | lstm_transfer/environment.py (6f8be85e): fine-tune + eval | lstm_transfer/modelCPU_N.pth, byte-identical to RS; tuned weights 2/5/10/20 tensor-identical to base |
| hpa_rule | – | HPA/environment.py (byte-identical to dense) | – |
| k8s_hpa, always_down | – | dense (no original script) | – |

- The reward is the same in all of these environments: 3L − 5lat − 10pods − 0.5cpu − 0.5mem. Only the capacity
  observation (state[1]) differs.
- `--train-env` and `--env` select any variant. `dense2` and `cpu2` have a different reward (normalised, plus 10000
  per tick).
- Paper Fig 6 is the dense model_1 eval plot and the lstm_transfer model_5 eval plot.
- Fig 5 and the 3.8e6 have no log source (the values were hand-typed). The parity tests therefore target the logged
  sums, not 3.8e6.

Modification times: `P/environmentCPU.py` and `P/agentHuber.py` (2025-02-03) and `P/dense/environment.py`
(2025-01-29) are newer than the weights that were trained with them (`P/modelCPU_*.pth` 2024-07/08,
`P/dense/modelCPU_*.pth` 2024-07-25). So the current files may differ from the ones that trained the weights. The
history is unknown, and only the original author's history or a backup would settle it.

## Run: `run_legacy.py`
```
python3 tuning/legacy/run_legacy.py --agent {derp,lstm_scratch,dinos_t,hpa_rule,k8s_hpa,always_down} \
   --train-ticks {1000,2000,5000,10000,20000} --seed S --protocol {orig,fair,both} \
   [--train-env dense|dense2|cpu|cpu2|lstm_transfer] [--env ...] [--hpa-target 50] --out r.json
```
- **Training.** N ticks run as N/1000 chunks of 1000 ticks on one env that is never reset, as the original "N steps"
  models. The original architectures and hyperparameters are used unchanged.
- **dinos_t.** Pretrains `agent_huber` on the train env (default dense 5ee774f0, as doubledeepCPU.py produced the
  `modelCPU_N.pth` that lstm_transfer.py loads) for `--train-ticks`. It then loads the weights into `DDQNAgentLSTM`
  with ε 0.2 and fine-tunes for 1000 ticks on the target env. The JSON reports `pretrain_ticks` and `finetune_ticks`
  separately.
- **P-orig** ("reproduction of figure protocol"):
  - **derp, lstm_scratch.** A fresh agent with ε=1 loads the weights, so its actions are uniform random. The env is
    never reset, so the five runs last 5000+1+1+1+1 ticks.
  - **dinos_t.** ε-greedy. The env is shared between fine-tuning and evaluation and is reset before each of 5×5000
    runs. The shared-env quirk is reproduced for `--train-ticks` ≠ 1000: fine-tuning stops after 1 tick.
  - **Rule agents.** Follow hpa_final.py: reset, then 5×5000.
- **P-fair.** Greedy, with no exploration draw. The env is reset, and numpy/random/torch are reseeded with 1000..1004
  before each of 5 episodes of 5000 ticks. The LSTM hidden state is zeroed at the start of each episode. Within one
  env class the exogenous noise is identical across agents. The reward law is identical in dense, cpu and
  lstm_transfer (only the capacity observation differs), but the noise streams differ between those classes.
- Both protocols start from the same post-training RNG state.
- **Output.** 5-episode sum, per-episode returns, reward components (3L, −5lat, −10N, −0.5cpu, −0.5mem), mean pods,
  CPU seconds, and the telemetry files `$QRL_RUN_DIR/metrics.jsonl` and `progress.json`.
- **Seeds.** `--seed` must be in 0–19999 and outside 1000–1004. When `QRL_SANDBOX_SEEDS` is set, every seed used
  (including 1000–1004 for fair) must lie in it, as in `sim_v2/env.py` `_sandbox_seed_guard`.
- **Deviations that change no number:**
  - no plots, tqdm or file writes;
  - stdout is silenced (cpu2/dense2 print every step);
  - the LSTM target network forward runs under `torch.no_grad()`. The original leaks an ever-growing autograd graph
    through the stateful `myLSTM.hidden`; the values are identical, and a test checks this.
- **k8s_hpa** is our addition: the `sim_v2/policies.py` K8sHPA formula applied to the legacy cpu observation, one
  ±1 move per tick. The legacy env makes cpu independent of pods, so it is noise-driven.

## Parity (tests/test_legacy.py, seed 0 unless stated; logged = sum of P/.../model_N/rewards.txt)
| check | logged / reference | port |
|---|---|---|
| HPA rule, 5-run sum | logged −1.0366/−1.0321/−1.0350/−1.0315/−1.0354M (mean −1.0341M) | −1.0289M (−0.5%), mean pods 19.68 |
| always-down ceiling (derived, not logged) | 3.641M | 3.6410M orig / 3.6410M fair |
| DERP protocol = random policy | logged 286.7/259.4/218.4/197.4/152.2k (mean 222.8k); recompute 252k SD 57k | 254.4k, SD 54.6k, SE 8.6k (seeds 200–239); the logged mean is within 2 SE of the difference, and every logged value is within ±3 SD |
| DInos model 1, saved fine-tuned weights, ε 0.0996 | 3.2809M | 3.2954M (+0.44%); seeds 0–4: 3.250–3.308M |
| DInos model 2 / 5, ε 0.2 | 3.6176M / 3.6166M | 3.6205M / 3.6205M (+0.08% / +0.11%) |
| DInos model 10 | 2.8221M | 2.8193M (−0.10%) |
| DInos model 20 | 3.2285M | 3.2664M (**+1.17%, not reproduced within 1%**); seeds 0–4: 3.255–3.284M, so the gap is systematic. The cause is unknown (torch-version numerics of near-tied Q values or the stateful hidden state are candidates). The test bound is 2%. |

## Legacy targets for `tuning/closeness.py --target legacy-orig|legacy-fair` (`targets.json`)
Written by `recompute_targets.py` from the original trained weights (DInos orig: model 1 = saved fine-tuned weights). The target is the mean over models
1/2/5/10/20.

| protocol | DERP | DInos |
|---|---|---|
| **legacy-orig** | 251.6k (random policy, 100 seeds; logged mean 222.8k) | 3.324M (per model 3.30/3.62/3.62/2.82/3.27M; logged mean 3.313M) |
| **legacy-fair** | 2.042M (per model 3.63/3.64/−0.09/−0.61/3.64M) | 3.478M (per model 3.62/3.64/3.64/2.88/3.62M; model 1 = saved fine-tuned weights) |

The paper targets (2.2e5 / 3.8e6) remain the default.

## Cost (1k-tick smoke, `--protocol both`, seed 0, 1 torch thread, taskset 1-7 with 6 runs in parallel)
| agent | CPU-s total | of which training | orig / fair 5-ep sum |
|---|---|---|---|
| derp | 44 | 38 | 233k / 2.424M |
| lstm_scratch | 224 | 182 | 322k / 2.392M |
| dinos_t | 620 | 186 (pretrain) + 2 × ~210 (two 1000-tick fine-tunes) | 1.942M / 2.391M |
| hpa_rule / k8s_hpa / always_down | 2–3 | 0 | −1.029M / −1.034M; −1.096M / −1.095M; 3.641M / 3.641M |

LSTM training costs about 0.18 CPU-s per tick and DERP about 0.04 CPU-s per tick (replay of 16 single samples per
tick). Both are expected to scale linearly with ticks; this is not measured beyond 2k (DERP 2k on cpu2: 74 CPU-s).
Fair evaluation costs about 42 CPU-s for an LSTM and 6 CPU-s for DERP. P-orig evaluation is cheap, except for
dinos_t at 1k, whose fine-tune runs for real.

## Cycle 4: coupled load-sharing variant (`coupled_env.py`, `--env-variant coupled`)
Design: `~/qrl-lab/.lab/tuning/cycles/c3_codex.md` and `c3_synthesis.md`. This is a synthetic mechanism experiment, not a
calibrated simulator. The ported env files are unchanged, and `CoupledEnv` wraps them. After the action sets the pods
n (±1, 1–20, start 6), the wrapped `step()` runs unchanged, so it makes the same random draws. The wrapper then applies
`g = 6L/(70n)` to the legacy draws: `c = min(100, g·c0)`, `m = min(100, g·m0)`, `l = g·l0`. The reward is
`3L − 5l − 10n − 0.5c − 0.5m`. Observations keep the layout `[L, capacity, c, m, l, n]` and carry the coupled c/m/l. At
L = 70 and n = 6, g = 1 and every number equals the legacy one (tested bit for bit). Inverse sharing and linear latency
are new assumptions.

- **Envs.** derp uses dense. lstm_scratch uses lstm_transfer for both training and evaluation. dinos_t pretrains B − 1000
  ticks on the **legacy** dense source env and then fine-tunes 1000 ticks on the coupled lstm_transfer target. The rule
  agents use dense.
- **Training.** Under coupled, training resets the env when it reports done (t ≥ 5000). Legacy training never resets.
- **Evaluation.** Only greedy evaluation is allowed (`--protocol fair`, the default under coupled). `--eval-split` takes
  a comma list of `fair` (1000–1004), `val` (1100–1104) and `test` (2000–2019). Each episode is 5000 ticks with a
  reset. Every split evaluates the same trained weights; dinos_t is fine-tuned once and that agent is shared by all
  splits. In stats mode (coupled, or any split other than `fair` alone), each episode also reports `p95_latency` and
  `mean_latency`, and each split reports `mean_ret`, `sd_ret`, `components_mean` and `p95_latency_mean`. `sum5` is then
  the sum over all of the split's episodes.
- **Legacy default.** `--env-variant legacy --eval-split fair` gives JSON identical to the previous code, apart from
  timings (checked for all six agents, 1k ticks, seed 3).
- **Example run.**
  `python3 tuning/legacy/run_legacy.py --agent dinos_t --train-ticks 5000 --seed 0 --env-variant coupled --eval-split val,test --out r.json`
- **Baselines.** `coupled_baselines.py` covers always_down, the verbatim HPA rule at (50, 50), the HPA rule tuned on val
  over {10, 20, …, 90}², and fixed-N for N in 1..20 in two variants: ramped from 6 by ±1, and preprovisioned. Selection
  is on val only, and the selected policies are then run on test. It is single-threaded.

## Cycle 5: common coupled target, transfer controls, checkpoints (`c5_protocol.py`)
Design: `~/qrl-lab/.lab/tuning/cycles/c4_codex.md` and `c4_synthesis.md`. Selected in `run_legacy.py` by any of
`--source-variant/--pretrain-ticks/--finetune-ticks/--total-ticks/--checkpoints/--target-env`; needs
`--env-variant coupled`; the legacy and cycle-4 paths are untouched (JSON identical apart from timings, checked).

- **Target.** All arms (DERP included) train and evaluate on the coupled `lstm_transfer` env (`--target-env`). DERP's
  network input stays 6 (the lstm_transfer state has the same layout; capacity is 10n there), architecture unchanged.
- **Arms.** `derp`/`lstm_scratch --total-ticks T`; `dinos_t --source-variant legacy|coupled --pretrain-ticks P
  --finetune-ticks F` (agentHuber pretrained on dense legacy / dense coupled, weights into DDQNAgentLSTM, epsilon
  0.2 -> 0.1, lr 5e-5, fresh memory/optimiser); `--source-variant none --finetune-ticks F` = the same transfer agent from
  random weights. Epsilon x0.995 per replay as the originals (DERP 1 -> 0.01, scratch 1 -> 0.05).
- **Training.** Episodes of 5000 ticks with reset on done; the window starts from the reset observation (also at the
  start of every phase). LSTM hidden state is not reset during training (as the originals).
- **RNG.** Every env has a private numpy stream (`StreamEnv`): target `[seed,5,1]`, source `[seed,5,2]`, eval episode
  seed s = `np.random.seed(s)` (same traces as `coupled_baselines.py`). Target traces are identical across arms.
- **Checkpoints.** `--checkpoints 1000,10000,30000` (target ticks): greedy eval of a deep copy on val+test, global RNG
  restored; `<out>.ck<T>.json` per checkpoint plus the full `--out`.
- **Example.** `python3 tuning/legacy/run_legacy.py --agent dinos_t --source-variant coupled --pretrain-ticks 10000
  --finetune-ticks 30000 --checkpoints 1000,10000,30000 --env-variant coupled --eval-split val,test --seed 0 --out r.json`

## Cycle 6: fixed LSTM variant, diagnostics, test2 (`agent_lstm_fixed.py`, `c6_diag.py`)
Design: `~/qrl-lab/.lab/tuning/cycles/c5_codex.md` and `c5_synthesis.md`. Both modules are new (not ports).
- **`--agent-variant verbatim|fixed`** (lstm_scratch, dinos_t; the dinos_t source pretraining uses the same variant).
  fixed = zero hidden state per 10x6 window (acting and replay), detached Double-DQN target (online argmax, target
  network evaluates), target sync every `--target-sync` replays (default 250). Network, inputs, reward, Huber, lrs,
  epsilon schedules and the 16 sequential updates per replay are as in the verbatim agents. DERP stays verbatim.
- **`--diag N`**: observers only (no change to training, tested): every N replays loss, grad norm, parameter delta vs
  phase start, TD-error quantiles, spurious non-action error, Q-margin and greedy histogram on 64 fixed probe windows,
  action histogram, replay-order probe (carried vs reset hidden state) -> `<out>.diag.json`.
- **`test2`** split = seeds 3000-3019 (fresh); `coupled_baselines.py --report-splits test,test2`.
- Verbatim path: JSON identical to the pre-cycle-6 code apart from timings and four metadata keys on the LSTM arms
  (`tests/data/c5_verbatim_ref_s3.json`, `tests/test_c6.py`).

## Cycle 8: conditioning options and test4 (`c5_protocol.py`)
Design: `~/qrl-lab/.lab/tuning/cycles/c7_codex.md` and `c7_synthesis.md`. All c5 arms (derp, lstm_scratch, dinos_t,
both phases); defaults reproduce cycles 5-7 exactly (`tests/data/c6_fixed_ref_s3.json` from commit fd42de8,
`tests/test_c8.py`); every JSON records `reward_div`, `obs_scale`, `obs_scale_divisors`, `eps_schedule`, `eps_linear_ticks`.
- **`--reward-div D`**: replay memory stores r/D, so targets are r/D + gamma*Q' and Q is learned in units of D from
  initialisation (incl. source pretraining). Reported returns are unscaled.
- **`--obs-scale on`**: agent observations (training, evaluation, diag probes) divided by [70, 60, 100, 100, 1.2, 6]
  for the env layout [load, capacity, cpu %, mem %, latency, pods] (checked against env_transfer/env_dense).
- **`--eps-schedule linear10k`**: epsilon linear from the agent's own initial value to its floor over 10k training
  ticks of the phase (scratch 1 -> 0.05, DERP 1 -> 0.01, transfer 0.2 -> 0.1); the x0.995 per-replay decay is off.
- **`test4`** split = seeds 5000-5019 (run_legacy.SPLITS; also `c7_oracles.py --split test4` and
  `coupled_baselines.py --report-splits test4`, same traces).

## Cycle 9: the original env P/environment.py (`env_orig.py`, `orig_variants.py`, `orig_baselines.py`)
Design: `~/qrl-lab/.lab/tuning/cycles/c9_design.md`. `env_orig.py` is the verbatim port of `P/environment.py`
(2024-07-16, md5 2096a045; reward min(capacity, load) − 3·VMs, step() returns a 3-tuple), checked byte for byte.
- **`orig_variants.py`** runs the unchanged port source in a private namespace whose `np.random` is the calling
  adapter's own RandomState (the global numpy stream is untouched). Variant `orig` with RandomState(s) equals the
  verbatim env under `np.random.seed(s)` exactly (all 5001 states and rewards, `tests/test_c9.py`). Variants are
  parameters only: `noise{K}` (load noise sd K, same draw), `spike300` (load spike amplitude 300), `multispike`
  (4 spikes of the original shape at stratified positions drawn per reset from a separate stream `[seed, 9]`;
  spike k starts in [1250k+50, 1250k+1100]), `noise15_multispike`. step() → (state, reward, done, info), info = served
  min(capacity, load) and vm_cost −3·VMs. `--obs-scale on` divisors [100, 100, 60, 70, 2.4, 20].
- **c5 protocol**: `--env-family orig --orig-variant V` (derp, lstm_scratch; no dinos_t, no --diag); target env
  `orig:V`, the same stream seeds as cycle 5. Eval episodes add `frac_const_pods` and `pods_changes`. The coupled path is
  unchanged (`tests/test_c8.py` reference JSON). New split **test5** = 6000-6019.
- **`orig_baselines.py --orig-variant V`**: fixed N 1..20 (ramped, best on val), demand HPA (u in 0.6..1.0 on val, band
  0.2; council addition), the verbatim HPA rule, always_down, causal, causal_be (break-even read 0.3; addition),
  causal_trend (k = 5; council addition) and a clairvoyant one-step oracle (peeks via `OrigEnv.clone()`). Per-trace
  returns on val and test5 (same traces as the agents' evaluation).
- Entries: `.lab/tuning/make_c9a.py` → `c9a_entries.json`; summary: `.lab/tuning/sum_c9.py`.
