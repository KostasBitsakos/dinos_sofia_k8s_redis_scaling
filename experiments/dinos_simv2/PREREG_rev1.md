# Pre-registration: Sim v2.1 protocol rev 1 (R-20261017-012, option 1)

Written and committed BEFORE any rev-1 run. Machine-readable twin: `gate_rev1.json` (both are in the freeze manifest).
Revised once before any run after the Codex review of 03832b4/b24b7e7 (burned seeds, wording, pins, extra registrations).

## Why
The original PLAN 8.1 preflight (run 20261017-140714-R-20261017-004-e0-a1, `preflight.json` sha256
`60023e4c6460dd4ffd2ce3c68190bca056345441026650b5760ea2c8e1bc8ccf`, pinned in `gate_rev1.json` and verified at load)
FAILED: the tuned threshold (40/25/0) did not beat `fixed_best` (c = 22). That policy starts with 22 ready pods while
every other policy starts at 6, so the comparison mixed policy quality with a start-up advantage. The diagnose run
(20261017-145935-R-20261017-010-e0-a1) looked at this; rev 1 equalises the start. **The original FAIL (R-20261017-004)
stands and is reported alongside any rev-1 result.** Rev 1 is a protocol amendment made after seeing those results; it
is not a re-run of the same test. The run asserts the original gate is recorded as not passed and fixed c = 22.

## Protocol
1. Common initial state c0 = `SimParams.init_pods` = 6 (the env default, "legacy env starts at 6 VMs"); fixed in
   advance, independent of the realised lambda(0). Threshold, static_ramp, always_min, always_max and the agents start
   there. (Oracle replay also starts there; it is clairvoyant by design.)
2. `static_ramp(c)`: starts at c0 and moves one pod per step toward c (up or down; startup delay and rebalancing
   apply), then holds c. c is selected on tune seeds 1000-1004 by mean return over all integers c in 1..30 (same grid
   and method as the `fixed_best` selection). **Tie-breaking (deterministic): the smallest c among candidates with
   exactly equal mean return.**
3. Threshold: the 40/25/0 selected in the original preflight is kept, no re-tuning.
4. The original preprovisioned `fixed_best` (c = 22, starts with c ready pods) is reported as a separate comparator
   and is not part of the gate.
5. Evaluation: default config, workload W0, n = 10 paired episodes, FRESH validation seeds **1200-1209**, no
   reselection of anything on them.

## Seeds (audit)
- 1100-1109 was the first choice for the gate. A smoke test of the first rev-1 commit (03832b4) evaluated 200-step
  episodes on them (its seed override was a no-op), so **1100-1109 are BURNED** and not used for the gate.
- 1200-1209: not named in any code, test, config, smoke, run output or doc before this revision (grep), and no code path
  iterates over a range containing them (selection uses 1000-1004 only; train seeds are drawn from 0-999; the Sim v2
  gate suite uses 1000-1004 and held-out 10000+). Tests and smokes cannot evaluate them: the test `conftest.py` raises
  on any simulator reset with a gate or burned seed, and `train_eval.guard_gate_block` refuses the block unless the run
  is launched frozen (`--freeze`); smokes use the unfrozen `--rev1-eval-seeds` override with development seeds 1000-1004.
- Final-test seeds 20000-20999 are untouched by rev 1 and stay reserved for the final evaluation.

## Fixed simulator conventions (default parameters unchanged)
- Scoring horizon: 5000 control steps per episode (`SimParams.episode_steps`), reward summed over all steps.
- Terminal backlog: work still queued at the end of the episode is settled as failed (counted once as bad) in the reward
  of the last step (`sim_v2/env.py`).
- Oracle: certified upper bound / replay of the DP's maximising action sequence in the real dynamics; the sanity
  criterion uses an absolute tolerance of 1e-9 on the mean returns (`oracle_tolerance_abs`).

## Gate (pass iff all hold)
- mean paired return difference (threshold - static_ramp_best) > 0 with a paired 95% percentile bootstrap CI (10,000
  resamples, numpy `default_rng` seed 20261017) whose lower end is > 0;
- threshold beats always_min and always_max (paired mean difference > 0);
- mean oracle replay return >= mean threshold return - 1e-9.

Reported, not gated: comparison with the preprovisioned fixed_best, startup (first 150 steps) vs rest split, win counts.
Sensitivity policy: the gate uses the default config only. Sensitivity configs are reported afterwards, descriptively and
non-gating, for (a) the frozen controls (threshold 40/25/0, static_ramp_best c, fixed_best c = 22) and (b) controls
re-selected per config on seeds 1000-1004 and labelled as such. (Not implemented in the `preflight_rev1` agent; any
such report is a separate, labelled analysis.)

## Consequences
No further amendments based on seeds 1000-1004 or 1200-1209. Agents may be run (`run_agent`) only with a verified passing
rev-1 artifact (`--rev1-artifact PATH --rev1-sha256 HEX`, checked: protocol, gate passed, seeds, no reselection, pinned
preflight sha256) or an explicit approved-amendment id (`--amendment-id`, recorded in `rev1_authorization.json`). If rev 1
fails, agents run only after an explicit, approved amendment of PLAN 8.1. A pass licenses only the sanity claim "the
simulator rewards a reactive controller over a static ramp on W0"; it says nothing about quantum or learned policies.
