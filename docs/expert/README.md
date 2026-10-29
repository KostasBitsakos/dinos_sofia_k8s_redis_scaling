---
description: Expert knowledge pack for cycle 2 (Poisson) - RL evaluation, DRL tuning, autoscaling baselines, workload generation (J12)
---
# Expert pack — cycle 2 (J12, 2026-10-29, R-20261028-001, M-20261029-017)
Short, practical notes for engineer/runner/analyst agents. Each source carries a verification level:
**V** = verified in this repo or online (pointer given) · **B** = bibliographic from memory, check DOI/venue with the
`scholar` agent before citing in the paper. Never cite a B entry in the paper unchecked.

| File | Use it for |
|---|---|
| `rl_evaluation.md` | seeds, splits, CIs, IQM, what to report |
| `drl_tuning.md` | how to spend an equal tuning budget per arm (DQN-family, recurrent, PPO) |
| `autoscaling_baselines.md` | DRe-SCale vs Gwydion decision; which baselines reviewers expect |
| `poisson_workloads.md` | correct Poisson / NHPP / MMPP generators, noise, trace splits, queue/latency model |
Skills encoding these: `.claude/skills/rl-eval-protocol`, `.claude/skills/drl-tuning`, `.claude/skills/poisson-workload`.
Plan: `docs/cycle2_plan.md`.
