"""Imports the researcher's agents from code/sim without copying them.

LSTM: code/sim/timeseries/Pytorch/lstm_transfer/agentHuber.py (DDQNAgentLSTM, default = fixed, non-legacy).
DERP: code/sim/timeseries/Pytorch/dense/ddqnagent.py (DDQNAgent, dense net on a single 6-vector, published
      per-sample replay with target sync after every replay; it has no window and no fixes, used as published).
"""
import importlib
import os
import sys

from adapter import N_ACTIONS, N_FEATURES, WINDOW

PYTORCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim",
                                        "timeseries", "Pytorch"))
LSTM_DIR = os.path.join(PYTORCH, "lstm_transfer")
DENSE_DIR = os.path.join(PYTORCH, "dense")


LOADED_FILES = set()   # source files of every module imported here (incl. the ones dropped from sys.modules)


def _import_from(directory, module, drop=()):
    """Import `module` with `directory` first on sys.path; remove the bare helper modules afterwards so that the
    LSTM and dense folders (which both define `environment`) cannot shadow each other."""
    sys.path.insert(0, directory)
    try:
        mod = importlib.import_module(module)
    finally:
        sys.path.remove(directory)
        for m in list(sys.modules.values()):
            f = getattr(m, "__file__", None)
            if f and os.path.abspath(f).startswith(directory + os.sep):
                LOADED_FILES.add(os.path.abspath(f))
        for d in drop:
            sys.modules.pop(d, None)
    return mod


def pretrained_path(k):
    p = os.path.join(LSTM_DIR, f"modelCPU_{int(k)}.pth")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return p


def make_lstm(pretrained=None):
    mod = _import_from(LSTM_DIR, "agentHuber", drop=("networks",))
    path = pretrained_path(pretrained) if pretrained is not None else None
    return mod.DDQNAgentLSTM(N_FEATURES, N_ACTIONS, WINDOW, pretrained_model=path)  # legacy=False


def make_derp():
    mod = _import_from(DENSE_DIR, "ddqnagent", drop=("environment", "network", "ddqnagent"))
    return mod.DDQNAgent(N_FEATURES, N_ACTIONS)


def agent_state(agent_name, window):
    """What each agent sees: the whole window for LSTM agents, the latest row for the dense DERP agent."""
    return window[-1] if agent_name == "derp" else window
