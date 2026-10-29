"""Input-feature mapping between the DInos pretraining inputs and the Sim v2 adapter features.

Pretraining (sim/timeseries/Pytorch/lstm_transfer/environment.py) fed the LSTM RAW, unscaled values in this order:
  slot 0 load      ~ 20 .. 120   (50 sin + 50 + 20, spike up to +80)
  slot 1 capacity  = 10 * VMs    (10 .. 200)
  slot 2 cpu       ~ 30 .. 78    (uniform 30..75, +3 in the spike)
  slot 3 memory    ~ 40 .. 68    (uniform 40..65, +3 in the spike)
  slot 4 latency   ~ 0 .. 3.5    (1.2 sin + 1.2 + noise, spike adds up to 2.4)
  slot 5 VMs       1 .. 20
The Sim v2 adapter (experiments/dinos_simv2/adapter.py) writes NORMALISED features (all about 0..1) in these slots:
  0 completed_ops/(mu*max_pods)  1 cpu/100  2 log10(1+p95_ms)/4  3 log1p(failed)/log1p(mu*max_pods)
  4 ready/max_pods               5 pending/4
So without mapping (a) the slot meanings differ (adapter slot 1 is CPU, the pretrained slot 1 is capacity ...) and
(b) the scales differ by 1-2 orders of magnitude. Loading modelCPU_k.pth is then only a weight initialisation.

mode "pretrain" rebuilds a pretraining-shaped window from the adapter window (semantic + scale mapping, ASSUMPTIONS,
untested whether it helps; the ranges above are read from environment.py, not measured):
  load     <- completed_ops_norm * 100       (so full capacity throughput ~ 100, mid-range of the pretrain load)
  capacity <- 10 * vms
  cpu      <- cpu_norm * 100                 (0..100 vs pretrain 30..75)
  memory   <- 52.5 (constant: Sim v2 has no memory signal; midpoint of 40..65)
  latency  <- p95_norm * 2.4                 (p95 10 s -> 2.4, the pretrain latency mid/upper range)
  vms      <- ready_norm * 20                (pretrain VMs 1..20; max_pods of Sim v2 mapped onto 20)
failed/s and pending are dropped (no pretraining counterpart).
mode "none": identity (current behaviour).
"""
import numpy as np

MODES = ("none", "pretrain", "adapter")
LOAD_SCALE, CPU_SCALE, LAT_SCALE, VM_SCALE, CAP_PER_VM, MEM_CONST = 100.0, 100.0, 2.4, 20.0, 10.0, 52.5


def map_window(window, mode="none"):
    """(..., 6) adapter-normalised array -> (..., 6) array in the input convention of the agent."""
    if mode in ("none", "adapter"):      # "adapter": raw sim v2 features; the agent owns a trainable input layer
        return window
    if mode != "pretrain":
        raise ValueError(f"input map mode must be one of {MODES}, got {mode!r}")
    w = np.asarray(window, dtype=np.float64)
    out = np.empty_like(w)
    vms = w[..., 4] * VM_SCALE
    out[..., 0] = w[..., 0] * LOAD_SCALE
    out[..., 1] = CAP_PER_VM * vms
    out[..., 2] = w[..., 1] * CPU_SCALE
    out[..., 3] = MEM_CONST
    out[..., 4] = w[..., 2] * LAT_SCALE
    out[..., 5] = vms
    return out


def affine_pretrain(d=6):
    """(W, b) with  map_window(x) == x @ W.T + b  for the 'pretrain' mode (initialisation of the trainable adapter)."""
    W = np.zeros((d, d))
    b = np.zeros(d)
    W[0, 0] = LOAD_SCALE
    W[1, 4] = CAP_PER_VM * VM_SCALE
    W[2, 1] = CPU_SCALE
    b[3] = MEM_CONST
    W[4, 2] = LAT_SCALE
    W[5, 4] = VM_SCALE
    return W, b
