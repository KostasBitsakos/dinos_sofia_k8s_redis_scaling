import pytest


@pytest.fixture(autouse=True)
def _allow_unfrozen(monkeypatch):
    """Tests run train_eval.main without a spec; real runs always pass --freeze (checked in the spec tests)."""
    monkeypatch.setenv("QRL_ALLOW_UNFROZEN", "1")


GATE_BLOCK = range(1200, 1210)   # rev-1 gate seeds (gate_rev1.json); 1100-1109 are burned
BURNED = range(1100, 1110)
EXPLORE_BLOCK = range(1300, 1320)   # exploratory W0 evaluation block (explore_w0.json)


@pytest.fixture(autouse=True)
def _never_evaluate_gate_seeds(monkeypatch):
    """Guard: no test or smoke may reset the simulator on the rev-1 gate block (or the burned first choice)."""
    import env
    orig = env.SimV2.reset

    def reset(self, seed=0, init_pods=None):
        if int(seed) in GATE_BLOCK or int(seed) in BURNED or int(seed) in EXPLORE_BLOCK:
            raise AssertionError(f"tests must never evaluate gate/burned seed {seed}")
        return orig(self, seed, init_pods=init_pods)
    monkeypatch.setattr(env.SimV2, "reset", reset)


@pytest.fixture(autouse=True)
def _lab_db(tmp_path, monkeypatch):
    """Unfrozen runs read amendment approvals from QRL_LAB_DB: a throw-away database with one approved test id,
    one open id and the real amendment id (approved), never the real lab database."""
    import sqlite3
    p = tmp_path / "lab_test.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE requests (id TEXT PRIMARY KEY, state TEXT, title TEXT, decided_at TEXT)")
    con.executemany("INSERT INTO requests VALUES (?, ?, ?, ?)", [
        ("R-TEST-000", "approved", "test", "2026-10-17"), ("R-20261017-099", "approved", "test", "2026-10-17"),
        ("R-TEST-OPEN", "open", "test", None), ("R-20261017-014", "approved", "test copy", "2026-10-17")])
    con.commit()
    con.close()
    monkeypatch.setenv("QRL_LAB_DB", str(p))
    return p
