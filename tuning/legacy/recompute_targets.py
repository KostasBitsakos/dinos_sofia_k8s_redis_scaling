#!/usr/bin/env python3
"""Recompute the legacy reference numbers used by tuning/closeness.py --target legacy-orig|legacy-fair, from the
ORIGINAL trained weights (read-only, P = ~/simulations/redisk8s/timeseries/Pytorch) run through the verbatim port.

  orig (reproduction of figure protocol):
    derp   P/dense/doubledeepfinal.py: fresh agent, epsilon 1.0 -> uniform random actions whatever the weights;
           mean over seeds 0..N-1 of the 5-run sum (no reset: 5000 + 4 x 1 ticks)
    dinos  P/lstm_transfer/lstm_transfer.py replay of the evaluated policies (run_legacy.dinos_orig_logged_model):
           model 1 = saved fine-tuned weights at the post-fine-tune epsilon, models 2..20 = base weights at 0.2; seed 0
  fair (greedy, reset, eval seeds 1000-1004):
    derp   P/dense/modelCPU_{N}.pth on the dense env
    dinos  the policies the original pipeline actually evaluated: P/lstm_transfer/final_metrics_lstm/modelCPU_1_fine_tuned.pth
           for model 1, the base P/lstm_transfer/modelCPU_{N}.pth for N = 2..20 (their fine-tuning was a no-op)
Target per protocol and agent = mean over the 5 models. Writes tuning/legacy/targets.json.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from tuning.legacy import ddqnagent, env_dense, env_transfer, port_sources, run_legacy as RL  # noqa: E402

P = port_sources.LEGACY_ROOT
MODELS = (1, 2, 5, 10, 20)


def derp_orig_random(n_seeds):
    out = []
    for s in range(n_seeds):
        RL.seed_all(s)
        a = ddqnagent.DDQNAgent(6, 3)
        out.append(sum(e["ret"] for e in RL.eval_noreset(env_dense.Environment(), lambda x: a.act(x, verbose=0), False)))
    return out


def fair(agent_kind, path):
    if agent_kind == "derp":
        a = ddqnagent.DDQNAgent(6, 3)
        a.online_model.load_state_dict(torch.load(path))
        eps = RL.eval_reset(env_dense.Environment(), RL.greedy_fn(a, False), False, True, RL.EVAL_SEEDS)
    else:
        a = RL.make_transfer(torch.load(path))
        hid = lambda: setattr(a.online_model, "hidden", a.online_model.init_hidden())  # noqa: E731
        eps = RL.eval_reset(env_transfer.Environment(), RL.greedy_fn(a, True), True, False, RL.EVAL_SEEDS, hid)
    return RL.summarize(eps)["sum5"]


def main(n_seeds=100):
    torch.set_num_threads(1)
    with open(os.devnull, "w") as dn:
        import contextlib
        with contextlib.redirect_stdout(dn):
            rnd = derp_orig_random(n_seeds)
            d_orig = {m: RL.dinos_orig_logged_model(m, 0)["sum5"] for m in MODELS}
            de_fair = {m: fair("derp", f"{P}/dense/modelCPU_{m}.pth") for m in MODELS}
            di_fair = {m: fair("dinos", f"{P}/lstm_transfer/final_metrics_lstm/modelCPU_1_fine_tuned.pth" if m == 1
                               else f"{P}/lstm_transfer/modelCPU_{m}.pth") for m in MODELS}
    res = {
        "_doc": __doc__,
        "logged_sums_models_1_2_5_10_20": {k: list(v) for k, v in RL.LOGGED.items()},
        "legacy-orig": {"derp": float(np.mean(rnd)), "dinos": float(np.mean(list(d_orig.values()))),
                        "derp_random_seeds": n_seeds, "derp_random_sd": float(np.std(rnd, ddof=1)),
                        "dinos_per_model": {str(k): v for k, v in d_orig.items()}},
        "legacy-fair": {"derp": float(np.mean(list(de_fair.values()))), "dinos": float(np.mean(list(di_fair.values()))),
                        "derp_per_model": {str(k): v for k, v in de_fair.items()},
                        "dinos_per_model": {str(k): v for k, v in di_fair.items()}},
    }
    with open(os.path.join(HERE, "targets.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "_doc"}, indent=1))


if __name__ == "__main__":
    main()
