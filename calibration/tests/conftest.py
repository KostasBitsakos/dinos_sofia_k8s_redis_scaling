import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest


@pytest.fixture(autouse=True)
def _run_dir(tmp_path, monkeypatch):
    """Keep events.jsonl / MEM_TRIPPED of every test out of the source tree."""
    if "QRL_RUN_DIR" not in os.environ:
        monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path / "rundir"))


@pytest.fixture(autouse=True)
def _rollback_paths(tmp_path, monkeypatch):
    """Tests never touch the real .last_rollback.json / .rollback.lock."""
    monkeypatch.setenv("QRL_LAST_ROLLBACK", str(tmp_path / "last_rollback.json"))
    monkeypatch.setenv("QRL_ROLLBACK_LOCK", str(tmp_path / "rollback.lock"))
