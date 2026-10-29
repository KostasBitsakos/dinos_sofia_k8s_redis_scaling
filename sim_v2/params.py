"""SimParams: every Sim v2 parameter in one place, each labelled provisional with its source.

Nothing here is measured. Council C-20261016-01 (option 1, approval R-20261016-013) fixes these as
*provisional* starting values; the calibration pilot (separate approval) replaces them. Results computed
with these values are not paper results.
"""
from dataclasses import dataclass, field, fields, replace
import math

COUNCIL = "council C-20261016-01 card (provisional starting value, not measured)"
DINOS = "DInos convention (code/sim/timeseries/Pytorch/lstm_transfer/environment.py)"
ENG = "engineering choice for Sim v2 (R-20261016-013), not measured"


def _p(default, source, unit=""):
    return field(default=default, metadata={"provisional": True, "source": source, "unit": unit})


@dataclass(frozen=True)
class SimParams:
    # --- time ---
    control_step_s: float = _p(15.0, DINOS + "; 15 s control step", "s")
    dt_s: float = _p(1.0, ENG + "; internal fluid sub-step, convergence-checked against 15 s", "s")
    episode_steps: int = _p(5000, DINOS + "; 5000-step episodes", "steps")
    # --- pods and actions ---
    min_pods: int = _p(1, "task R-20261016-013: c in [1, 30]", "pods")
    max_pods: int = _p(30, "task R-20261016-013: c in [1, 30]; claude-r2 (30 pods ~ 6 of 32 vCPU)", "pods")
    init_pods: int = _p(6, DINOS + "; legacy env starts at 6 VMs", "pods")
    max_delta: int = _p(1, ENG + "; 1 = actions {0,1,2}; 2 adds 3 = -2 and 4 = +2", "pods/step")
    startup_delay_steps: int = _p(2, COUNCIL + "; d = 2 steps (30 s)", "steps")
    # --- capacity C(c) = mu * c**theta ---
    mu_ops_per_pod: float = _p(1000.0, ENG + "; mu = 1000 ops/s per pod (Redis pod limited to 200m CPU)", "ops/s")
    theta: float = _p(1.0, COUNCIL + "; linear capacity, sensitivity at 0.8 and 0.9", "")
    # --- rebalance loss ---
    rebalance_loss: float = _p(0.3, COUNCIL + "; delta = 0.3", "fraction")
    rebalance_steps: int = _p(1, COUNCIL + "; k = 1 step", "steps")
    rebalance_trigger: str = _p("action", ENG + "; 'action' = any accepted change (task text); "
                                "'topology' = when the ready set changes", "")
    # --- requests ---
    timeout_s: float = _p(60.0, ENG + "; client timeout 60 s, timed-out requests are failures", "s")
    slo_latency_s: float = _p(None, ENG + "; None -> 2 x p95 sojourn of one M/M/1 pod at rho = 0.5 "
                              "(= 4 ln20 / mu, 11.98 ms at mu = 1000)", "s")
    # --- Sim v2.1 sensitivity knobs (defaults reproduce Sim v2 exactly) ---
    latency_add_s: float = _p(0.0, "Sim v2.1 sensitivity (R-20261017-001); additive client/network latency on every "
                              "request, shifts observed p95 and the SLO test; 0 = Sim v2", "s")
    tail_scale: float = _p(1.0, "Sim v2.1 sensitivity (R-20261017-001); excess-tail factor kappa >= 1 multiplying the "
                           "sojourn time (li2014tales: real tails exceed queueing models); 1 = Sim v2", "")
    cpu_offset_pct: float = _p(0.0, "Sim v2.1 sensitivity (R-20261017-001); constant background CPU added to the "
                               "OBSERVED cpu_pct only, capacity unchanged; 0 = Sim v2", "%")
    # --- reward ---
    reward_mode: str = _p("cost", "R-20261017-017: 'cost' = Sim v2 reward (default, old specs); 'dinos' = DInos-style "
                          "reward adapter, constants in reward_dinos.json", "")
    dinos_variant: str = _p("default", "R-20261017-023: reward_dinos.json variant used when reward_mode='dinos'; 'default' = V2 "
                            "(mean latency incl. failed at timeout, offered-demand U); 'legacy_p95' = the old p95/served-U "
                            "definition (reproduces runs made before 2026-10-18)", "")
    alpha: float = _p(1.0, COUNCIL + "; alpha = 1 per billed pod-step", "")
    beta: float = _p(100.0, COUNCIL + "; beta = 100 per normalised bad-request volume", "")
    gamma: float = _p(0.5, COUNCIL + "; gamma = 0.5 per pod added or removed", "")
    cost_norm_s: float = _p(None, ENG + "; None -> control_step_s (pod-seconds -> pod-steps)", "s")
    sla_norm_requests: float = _p(None, ENG + "; None -> mu * control_step_s (one pod-step of work)", "requests")
    # --- telemetry noise (observations only, never dynamics) ---
    cpu_noise_pct: float = _p(2.0, ENG + "; Gaussian sd on CPU %", "%")
    latency_noise_log_sd: float = _p(0.05, ENG + "; lognormal sd on observed p95", "")

    # ---- derived values ----
    @property
    def n_sub(self):
        n = self.control_step_s / self.dt_s
        if abs(n - round(n)) > 1e-9 or round(n) < 1:
            raise ValueError("control_step_s must be an integer multiple of dt_s")
        return int(round(n))

    @property
    def slo_s(self):
        if self.slo_latency_s is not None:
            return float(self.slo_latency_s)
        # p95 sojourn of M/M/1 with service rate mu at rho: ln(20) / (mu (1 - rho)); SLO = 2x at rho = 0.5
        return 2.0 * math.log(20.0) / (self.mu_ops_per_pod * 0.5)

    @property
    def slo_eff_s(self):
        """SLO on the plain M/M/1 sojourn that is equivalent to latency_add_s + tail_scale * sojourn <= SLO.
        With the defaults this is exactly slo_s. env and oracle both use it, so they stay consistent."""
        if self.tail_scale < 1.0 or self.latency_add_s < 0.0:
            raise ValueError("tail_scale must be >= 1 and latency_add_s >= 0 (the oracle proof needs it)")
        eff = (self.slo_s - self.latency_add_s) / self.tail_scale
        if eff <= 0.0:
            raise ValueError("latency_add_s must be below the SLO")
        return eff

    @property
    def cost_norm(self):
        return self.control_step_s if self.cost_norm_s is None else float(self.cost_norm_s)

    @property
    def sla_norm(self):
        if self.sla_norm_requests is not None:
            return float(self.sla_norm_requests)
        return self.mu_ops_per_pod * self.control_step_s

    def capacity(self, ready, rebalancing):
        """Service capacity in ops/s of `ready` pods, reduced by (1 - delta) while rebalancing."""
        c = self.mu_ops_per_pod * float(ready) ** self.theta
        return c * (1.0 - self.rebalance_loss) if rebalancing else c

    def with_(self, **kw):
        return replace(self, **kw)

    @classmethod
    def provenance(cls):
        """[(name, default, unit, provisional, source)] for the README and the tests."""
        return [(f.name, f.default, f.metadata.get("unit", ""), f.metadata.get("provisional"),
                 f.metadata.get("source")) for f in fields(cls)]
