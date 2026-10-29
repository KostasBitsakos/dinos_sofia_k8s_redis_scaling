"""Loader for the frozen Sim v2.1 sensitivity configurations (sensitivity_v21.json).

  names()            -> list of config names ("default" first)
  params(name, base) -> SimParams with that config's overrides applied on `base` (default SimParams())
Design choice: cpu_offset_pct changes the OBSERVED cpu_pct only (capacity and queue unchanged); a background
CPU consumer that steals capacity (fork stall, load_metrics_literature.md 2.3) needs a design decision and is
not part of Sim v2.1. latency_add_s / tail_scale change the observed p95 and the SLO test (via
SimParams.slo_eff_s), so the oracle stays a valid bound for them.
"""
import json
import os

from params import SimParams

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensitivity_v21.json")


def _load():
    with open(PATH) as f:
        return json.load(f)["configs"]


def names():
    return list(_load())


def overrides(name):
    return dict(_load()[name]["overrides"])


def params(name, base=None):
    return (base or SimParams()).with_(**overrides(name))
