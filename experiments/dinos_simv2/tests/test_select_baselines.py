import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def test_select_tie_break_is_smallest_candidate():
    import select_baselines_dinos as sb   # imported per test: the manifest test drops it from sys.modules
    assert sb.select([(40, -1.0), (30, -1.0), (50, -2.0)]) == (30, [30, 40])
    assert sb.select([((60, 20, 1), -5.0), ((60, 20, 0), -5.0)])[0] == (60, 20, 0)


def test_budget_caps_every_family_deterministically():
    import select_baselines_dinos as sb   # imported per test: the manifest test drops it from sys.modules
    from params import SimParams
    a = sb.candidates(SimParams(), budget=3)
    b = sb.candidates(SimParams(), budget=3)
    assert a == b and all(len(v) <= 3 for v in a.values())
    full = sb.candidates(SimParams())
    assert {k: len(v) for k, v in full.items()}["k8s_hpa"] == 18
    assert all(dn < up for _, (up, dn, cd) in full["threshold"])


def test_dev_only_small_run(tmp_path):
    import select_baselines_dinos as sb   # imported per test: the manifest test drops it from sys.modules
    out = tmp_path / "sel.json"
    res = sb.main(["--reward-mode", "dinos", "--budget", "2", "--jobs", "2", "--out", str(out)])
    on_disk = json.load(open(out))
    assert on_disk["seeds"] == [1000, 1001, 1002, 1003, 1004] == res["seeds"]
    assert set(on_disk["selected"]) == {"k8s_hpa", "threshold", "fixed", "static_ramp"}
    assert on_disk["reward_mode"] == "dinos" and len(on_disk["reward_dinos_sha256"]) == 64
    assert all(n <= 2 for n in on_disk["n_candidates"].values())
