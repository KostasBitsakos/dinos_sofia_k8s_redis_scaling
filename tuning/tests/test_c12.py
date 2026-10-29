"""Cycle 12: fresh confirmation split test8 (9000-9019) and the c12a/c12b training seeds (20-27)."""
import importlib.util
import json
import os

import pytest

from tuning.legacy import orig_baselines as OB
from tuning.legacy import run_legacy as RL

TEST3 = set(range(4000, 4020))          # c7_oracles.TEST3_SEEDS (not in RL.SPLITS)
C12_TRAIN_SEEDS = set(range(20, 28))    # c12a 20-24, c12b 25-27
MAKE_C12 = os.path.expanduser("~/qrl-lab/.lab/tuning/make_c12.py")


def test_test8_split_fresh_and_disjoint():
    assert RL.SPLITS["test8"] == tuple(range(9000, 9020)) == RL.TEST8_SEEDS
    for k in ("fair", "val", "test", "test2", "test4", "test5", "test6", "test7"):
        assert not set(RL.SPLITS["test8"]) & set(RL.SPLITS[k]), k
    assert not set(RL.SPLITS["test8"]) & TEST3
    assert RL.parse_splits("val,test8") == ("val", "test8")


def test_c12_training_seeds_outside_all_splits():
    every = set().union(*(set(v) for v in RL.SPLITS.values())) | TEST3
    assert not C12_TRAIN_SEEDS & every
    for s in C12_TRAIN_SEEDS:
        RL.check_seeds(s, "fair", ("val", "test5", "test8"))
    with pytest.raises(SystemExit):
        RL.check_seeds(9003, "fair", ("val", "test8"))


def test_baselines_report_test8(monkeypatch, tmp_path):
    monkeypatch.setitem(RL.SPLITS, "val", (1100,))
    monkeypatch.setitem(RL.SPLITS, "test8", (9000,))
    out = tmp_path / "b.json"
    res = OB.main(["--orig-variant", "spike300_d5", "--report-splits", "test8", "--ticks", "40", "--out", str(out)])
    assert "test8" in res and json.load(open(out))["test8_seeds"] == [9000]


@pytest.mark.skipif(not os.path.exists(MAKE_C12), reason="lab generator not present")
def test_make_c12_excludes_every_split():
    spec = importlib.util.spec_from_file_location("make_c12", MAKE_C12)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    every = set().union(*(set(v) for v in RL.SPLITS.values())) | TEST3
    assert m.EVAL_SEEDS == every
    for part, n in (("a", 2 * 5 * 5 + 2), ("b", 3 * 4 + 1)):
        ents, _ = m.build(part, [int(x) for x in m.PARTS[part]["seeds"].split(",")])
        assert len(ents) == n and len({e["name"] for e in ents}) == n
        for e in ents:
            c = e["cmd"]
            if "--eval-split" in c:
                assert RL.parse_splits(c[c.index("--eval-split") + 1]) == tuple(m.PARTS[part]["splits"].split(","))
            else:
                assert c[c.index("--report-splits") + 1] == m.PARTS[part]["report"]
