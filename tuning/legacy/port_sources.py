#!/usr/bin/env python3
"""Generate the verbatim legacy ports in this directory from the original DInos/DERP simulation code (read-only).

Each port = provenance header + the original file byte-for-byte, except the import lines listed in IMPORT_SUBS
(flat-directory imports -> package-relative imports). Nothing else is changed. tests/test_legacy.py re-renders every
port from the original and asserts byte equality, so any accidental edit fails the tests.

  python3 tuning/legacy/port_sources.py            # (re)write the ports (only needed if the originals change)
"""
import hashlib
import os

LEGACY_ROOT = "/home/ubuntu/simulations/redisk8s/timeseries/Pytorch"
HERE = os.path.dirname(os.path.abspath(__file__))

# port file -> (original path relative to LEGACY_ROOT, [(old import line, new import line), ...], role)
PORTS = {
    "env_dense.py": ("dense/environment.py", [], "imported by dense/doubledeep.py, dense/doubledeepfinal.py, dense/ddqnagent.py; HPA/environment.py is byte-identical"),
    "env_cpu.py": ("environmentCPU.py", [], "imported by doubledeepCPU.py, doubledeepCPUfinal.py"),
    "env_transfer.py": ("lstm_transfer/environment.py", [], "imported by lstm_transfer/lstm_transfer.py"),
    "env_dense2.py": ("dense/environment2.py", [], "imported by dense/doubledeep2.py (normalised reward + 10000/tick baseline)"),
    "env_orig.py": ("environment.py", [], "P/environment.py (2024-07-16): reward min(capacity, load) - 3*VMs; step() returns a "
                                          "3-tuple; cycle 9 reference env, wrapped by orig_variants.py"),
    "env_cpu2.py": ("environmentCPU2.py", [], "imported by doubledeepCPU2.py (normalised reward + 10000/tick baseline; prints every step)"),
    "network_dense.py": ("dense/network.py", [], "DERP dense Q-network"),
    "ddqnagent.py": ("dense/ddqnagent.py",
                     [("from network import MyDenseNet  # Import the new dense network",
                       "from .network_dense import MyDenseNet  # Import the new dense network"),
                      ("from environment import Environment", "from .env_dense import Environment")],
                     "DERP DDQN agent"),
    "networks_lstm.py": ("networks.py", [], "LSTM Q-network (lstm_transfer/networks.py differs only in blank lines and "
                                            "one commented-out print)"),
    "agent_huber.py": ("agentHuber.py",
                       [("from networks import myLSTM  # Assuming myLSTM is defined in networks.py",
                         "from .networks_lstm import myLSTM  # Assuming myLSTM is defined in networks.py")],
                       "LSTM agent of doubledeepCPU.py / doubledeepCPUfinal.py"),
    "agent_huber_transfer.py": ("lstm_transfer/agentHuber.py",
                                [("from networks import myLSTM", "from .networks_lstm import myLSTM")],
                                "DDQNAgentLSTM of lstm_transfer/lstm_transfer.py (loads pretrained weights, fine_tune)"),
    "simpleagent.py": ("HPA/simpleagent.py",
                       [("from environment import Environment", "from .env_dense import Environment")],
                       "threshold rule of HPA/hpa_final.py"),
}


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def render(port):
    rel, subs, role = PORTS[port]
    with open(os.path.join(LEGACY_ROOT, rel), "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")
    lines = text.split("\n")
    changed = []
    for old, new in subs:
        idx = [i for i, ln in enumerate(lines) if ln == old]
        if len(idx) != 1:
            raise SystemExit(f"{rel}: expected exactly one line {old!r}, found {len(idx)}")
        lines[idx[0]] = new
        changed.append(idx[0] + 1)
    n_lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    header = (f"# LEGACY PORT (Track L) -- verbatim copy of {LEGACY_ROOT}/{rel}:1-{n_lines}\n"
              f"# original sha256 {sha256(raw)}\n"
              f"# role: {role}\n"
              f"# changes: {'import lines ' + ', '.join(map(str, changed)) + ' made package-relative' if changed else 'none'}"
              f"; nothing else. Regenerate/verify with tuning/legacy/port_sources.py and tests/test_legacy.py.\n")
    return header + "\n".join(lines)


def main():
    for port in PORTS:
        with open(os.path.join(HERE, port), "w") as f:
            f.write(render(port))
        print("wrote", port)


if __name__ == "__main__":
    main()
