---
description: DRL hyperparameter tuning practice for DQN/DRQN/PPO arms under an equal per-arm budget (cycle 2)
---
# DRL tuning under an equal budget
## Sources
- Eimer, Lindauer, Raileanu, "Hyperparameters in Reinforcement Learning and How To Tune Them", ICML 2023,
  arXiv 2306.01324 — tuning choices change rankings; report the tuning protocol; seeds of tuning ≠ seeds of test. (B)
- Andrychowicz et al., "What Matters in On-Policy RL? A Large-Scale Empirical Study", ICLR 2021, arXiv 2006.05990 —
  PPO: observation normalisation, advantage normalisation, lr ~3e-4, separate actor/critic nets matter most. (B)
- Huang et al., "The 37 Implementation Details of Proximal Policy Optimization", ICLR Blog Track 2022. (B)
- Hessel et al., "Rainbow", AAAI 2018, arXiv 1710.02298 — n-step returns help (we confirmed n-step 3: +6.7k test9,
  `c19_screen.md`); dueling/PER did not help us (STATE §6). (B for paper; V for our result)
- Hausknecht & Stone, "Deep Recurrent Q-Learning for POMDPs", arXiv 1507.06527 — recurrent state reset/burn-in. (B)
- Raffin et al., "Stable-Baselines3", JMLR 22 (2021) — `sb3_contrib.RecurrentPPO` used by DRe-SCale
  (`docs/related_work_sota_autoscaler.md:20-23`). (B)
## Practice
1. **Equal budget = equal CPU-h and equal number of configs per arm**, same seeds for tuning (not the final seeds).
   Transfer arms pay their pretraining ticks from the same budget.
2. **Search.** Small random search or successive halving over a pre-declared space (lr, γ, n-step, target update,
   ε schedule, batch, net width; PPO: lr, n_steps, clip, ent_coef, GAE λ). Same number of trials per arm.
   Prefer short-horizon screening (5k ticks) → top-k extended (20k), identical for all arms.
3. **Fix the knobs that our cycle 1 already settled** (reward-div 10, obs-scale on, n-step 3 for DQN-family,
   linear10k ε; `c24_freeze.md`) as the *default* config, so tuning starts from the incumbent.
4. **Selection rule** written before tuning: across-seed mean on val at the locked checkpoint rule (as sum_c10).
5. **Recurrent arms**: reset hidden state at episode start; same observation window for all arms or report both
   (DRe-SCale used single-step obs + LSTM state; `related_work_sota_autoscaler.md:239-240`).
6. **Do not** tune on the test split, tune one arm more than another, or drop seeds after seeing results.
