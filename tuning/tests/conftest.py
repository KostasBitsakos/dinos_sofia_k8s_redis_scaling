import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "experiments", "dinos_simv2"), os.path.join(ROOT, "sim_v2")):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running parity test (deselect with -m 'not slow')")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_confirm_grants():
    """confirm_guard grants are per process; no test may inherit one from another test."""
    from tuning.legacy import confirm_guard as CG
    CG._GRANTS.clear()
    yield
    CG._GRANTS.clear()
