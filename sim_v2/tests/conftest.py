import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest  # noqa: E402

PROTECTED = (range(1100, 1110), range(1200, 1210), range(1300, 1320), range(20000, 21000))


@pytest.fixture(autouse=True)
def _never_evaluate_protected_seeds(monkeypatch):
    """Burned, rev-1 gate, exploratory W0 and final-test blocks are never simulated by a unit test."""
    import env
    orig = env.SimV2.reset

    def reset(self, seed=0, init_pods=None):
        if any(int(seed) in r for r in PROTECTED):
            raise AssertionError(f"tests must never evaluate protected seed {seed}")
        return orig(self, seed, init_pods=init_pods)
    monkeypatch.setattr(env.SimV2, "reset", reset)
