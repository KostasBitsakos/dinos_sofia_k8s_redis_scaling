"""Config-driven Q-networks (architecture JSON per model, models/configs/<name>.json).

The defaults in configs/ reproduce the existing classes bit-identically for the same torch seed:
  derp.json            == sim/timeseries/Pytorch/dense/network.py::MyDenseNet(6, 128, 3)
  dinos.json/dinos_t.json == sim/timeseries/Pytorch/lstm_transfer/networks.py::myLSTM(6, 128, 3, 3) (legacy=False)
Parameter creation order and attribute names (fc1.., lstm, fc) are the same, so state_dicts are interchangeable.

NOT wired into the agents yet: agents import the frozen classes (experiments/dinos_simv2/freeze.py FROZEN list).
Wiring needs an approved new freeze. torch is imported lazily so that validate_config works without it.
"""
import json
import os

ACTIVATIONS = ("relu", "tanh", "sigmoid", "leaky_relu", "gelu", "elu", "selu", "softplus", "linear")
KINDS = ("dense", "lstm")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


class ArchError(ValueError):
    pass


def _pos_int(v, what, hi=100000):
    if isinstance(v, bool) or not isinstance(v, int) or not 0 < v <= hi:
        raise ArchError(f"{what} must be an integer in 1..{hi}, got {v!r}")


def _layers(lst, what):
    if not isinstance(lst, list):
        raise ArchError(f"{what} must be a list")
    for i, l in enumerate(lst):
        if not isinstance(l, dict):
            raise ArchError(f"{what}[{i}] must be an object")
        _pos_int(l.get("units"), f"{what}[{i}].units")
        if l.get("activation") not in ACTIVATIONS:
            raise ArchError(f"{what}[{i}].activation {l.get('activation')!r} not in {ACTIVATIONS}")


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ArchError("config must be an object")
    if cfg.get("kind") not in KINDS:
        raise ArchError(f"kind must be one of {KINDS}")
    _pos_int(cfg.get("input_size"), "input_size")
    _pos_int(cfg.get("output_size"), "output_size")
    if cfg.get("output_activation", "linear") not in ACTIVATIONS:
        raise ArchError("output_activation unknown")
    if cfg["kind"] == "dense":
        _layers(cfg.get("layers"), "layers")
    else:
        _pos_int(cfg.get("hidden_size"), "hidden_size")
        _pos_int(cfg.get("num_layers"), "num_layers", 16)
        _layers(cfg.get("head", []), "head")
    return cfg


def _act(name):
    import torch.nn as nn
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU,
            "elu": nn.ELU, "selu": nn.SELU, "softplus": nn.Softplus, "linear": nn.Identity}[name]()


def build(cfg):
    """nn.Module for a validated config."""
    validate_config(cfg)
    import torch.nn as nn

    if cfg["kind"] == "dense":
        class ConfigDenseNet(nn.Module):
            def __init__(self):
                super().__init__()
                sizes = [cfg["input_size"]] + [l["units"] for l in cfg["layers"]] + [cfg["output_size"]]
                self.n = len(sizes) - 1
                for i in range(self.n):                       # fc1, fc2, ... as in MyDenseNet
                    setattr(self, f"fc{i + 1}", nn.Linear(sizes[i], sizes[i + 1]))
                self.acts = [_act(l["activation"]) for l in cfg["layers"]] + [_act(cfg.get("output_activation", "linear"))]

            def forward(self, x):
                for i in range(self.n):
                    x = self.acts[i](getattr(self, f"fc{i + 1}")(x))
                return x
        return ConfigDenseNet()

    class ConfigLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            h = cfg["hidden_size"]
            self.lstm = nn.LSTM(cfg["input_size"], h, cfg["num_layers"], batch_first=True)
            head = cfg.get("head", [])
            sizes = [h] + [l["units"] for l in head] + [cfg["output_size"]]
            self.fc = nn.Linear(sizes[0], sizes[1])           # `fc` is the first (or only) Linear, as in myLSTM
            self.more = nn.ModuleList(nn.Linear(sizes[i], sizes[i + 1]) for i in range(1, len(sizes) - 1))
            self.acts = [_act(l["activation"]) for l in head] + [_act(cfg.get("output_activation", "linear"))]

        def forward(self, x):
            out, _ = self.lstm(x)                              # zero initial (h, c): legacy=False behaviour
            x = self.fc(out[:, -1, :])
            x = self.acts[0](x)
            for lin, act in zip(self.more, self.acts[1:]):
                x = act(lin(x))
            return x
    return ConfigLSTM()


def load_config(name, config_dir=None):
    with open(os.path.join(config_dir or CONFIG_DIR, f"{name}.json"), encoding="utf-8") as f:
        return validate_config(json.load(f))
