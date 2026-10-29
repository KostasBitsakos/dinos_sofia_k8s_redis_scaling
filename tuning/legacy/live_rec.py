"""Opt-in observers for run_legacy.py / c5_protocol.py (dashboard M-20261021-008). Neither changes any number.

LiveRecorder (--live-file PATH): ring buffer of the last KEEP ticks of the current run (training and evaluation), one
row per env step: [t, phase, load, pods, pending, action, reward]. Every EVERY ticks (and at most once per
MIN_INTERVAL seconds of wall time) the buffer is written atomically (tmp file + os.replace) as one small JSON, so a
reader never sees a half-written file. t counts every env step of the run (pretrain, train and eval) from 1.
Trajectory (--record-traj N): per-tick lists of the first N greedy eval episodes of every checkpoint's last split,
stored in the result JSON under checkpoint_results[<T>][<split>]["trajectory"].

Both only read values the protocol already computed (state, action, reward); no RNG draw, no extra env step, no
change of the training/evaluation order (tuning/tests/test_live_rec.py checks bit-identity of the result JSON).
"""
import json
import os
import time
from collections import deque

COLS = ("t", "phase", "load", "pods", "pending", "action", "reward")
TRAJ_KEYS = ("load", "cpu", "mem", "latency", "pods", "pending", "reward", "action")
EVERY, KEEP, MIN_INTERVAL = 25, 600, 0.5


def _r(x, nd=3):
    return round(float(x), nd)


def atomic_write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


class LiveRecorder:
    def __init__(self, path, meta=None, every=EVERY, keep=KEEP, min_interval=MIN_INTERVAL, clock=time.time):
        self.path, self.meta = path, dict(meta or {})
        self.every, self.min_interval, self.clock = int(every), float(min_interval), clock
        self.buf = deque(maxlen=int(keep))
        self.t, self.label, self.last_write, self.started = 0, "", 0.0, clock()

    def set_label(self, label):
        self.label = str(label)

    def add(self, phase, state, action, reward):
        """state = the raw env observation after the step ([load, capacity, cpu, mem, latency, pods(, pending)])."""
        self.t += 1
        pend = int(round(float(state[6]))) if len(state) > 6 else 0
        self.buf.append([self.t, phase, _r(state[0]), _r(state[5], 2), pend, int(action), _r(reward)])
        if self.t % self.every == 0 and self.clock() - self.last_write >= self.min_interval:
            self.flush()

    def snapshot(self, done=False):
        return dict(self.meta, pid=os.getpid(), started=self.started, updated=self.clock(), done=bool(done), t=self.t,
                    label=self.label, phase=self.buf[-1][1] if self.buf else None, cols=list(COLS),
                    rows=[list(r) for r in self.buf])

    def flush(self, done=False):
        self.last_write = self.clock()
        try:
            atomic_write_json(self.path, self.snapshot(done))
        except OSError:
            pass      # a full disk or a vanished directory must never stop the experiment


class TrajRecorder:
    """Per-tick lists of one greedy eval episode."""

    def __init__(self):
        self.d = {k: [] for k in TRAJ_KEYS}

    def add(self, state, action, reward):
        d = self.d
        d["load"].append(_r(state[0]))
        d["cpu"].append(_r(state[2], 2))
        d["mem"].append(_r(state[3], 2))
        d["latency"].append(_r(state[4]))
        d["pods"].append(_r(state[5], 2))
        d["pending"].append(int(round(float(state[6]))) if len(state) > 6 else 0)
        d["reward"].append(_r(reward))
        d["action"].append(int(action))
