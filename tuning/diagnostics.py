"""Per-episode behaviour diagnostics: pod churn, reversals, mean pods, SLA violations (+ TD loss, mean |Q| via the caller)."""
import numpy as np


class EpisodeStats:
    """Accumulates one episode from sim.step() infos. Pod change = step with accepted_change != 0 (pods actually
    added / removed). Reversal = a pod change whose sign is opposite to the previous pod change."""

    def __init__(self):
        self.steps = 0
        self.ret = 0.0
        self.changes = 0
        self.reversals = 0
        self.up = self.down = 0
        self.pod_steps = 0.0
        self.bad = self.arrived = 0.0
        self.actions = [0, 0, 0]
        self._last = 0
        self.pods = []
        self.comp = dict(r_L=0.0, r_D=0.0, r_N=0.0, r_U=0.0, r_M=0.0)
        self.failed = 0.0
        self.lat_sum = 0.0
        self.lat_w = self.req_w = 0.0
        self.churn = 0.0

    def add(self, action, reward, info):
        self.steps += 1
        self.ret += reward
        self.actions[int(action)] += 1
        d = info["accepted_change"]
        self.churn += abs(d)
        if d != 0:
            self.changes += 1
            self.up += d > 0
            self.down += d < 0
            s = 1 if d > 0 else -1
            if self._last and s != self._last:
                self.reversals += 1
            self._last = s
        self.pod_steps += info["ready"] + info["pending"]
        self.pods.append(int(info["ready"] + info["pending"]))
        self.bad += info["bad_requests"]
        self.arrived += info["arrived"]
        self.failed += info.get("failed", 0.0)
        c = info.get("dinos")
        if c:
            for k in self.comp:
                self.comp[k] += c[k]
            self.lat_sum += c["p95_all_s"]
            n_req = info.get("served", 0.0) + info.get("failed", 0.0)
            self.lat_w += c["p95_all_s"] * n_req
            self.req_w += n_req

    def pod_trajectory(self):
        """Total pods (ready+pending) after the 1st step, after step 101, 501 and after the last step."""
        p = self.pods
        at = lambda i: p[min(i, len(p) - 1)] if p else None
        return dict(pods_t0=at(0), pods_t100=at(100), pods_t500=at(500), pods_end=at(len(p) - 1))

    def summary(self):
        n = max(self.steps, 1)
        return dict(**self.pod_trajectory(), steps=self.steps, ret=float(self.ret), pod_changes=int(self.changes), reversals=int(self.reversals),
                    scale_up=int(self.up), scale_down=int(self.down), mean_pods=self.pod_steps / n,
                    sla_violation_rate=self.bad / max(self.arrived, 1e-12), bad_requests=float(self.bad),
                    action_counts=list(self.actions),
                    sum_3L=self.comp["r_L"], sum_neg5D=self.comp["r_D"], sum_neg10N=self.comp["r_N"],
                    sum_neg05U=self.comp["r_U"], sum_neg05M=self.comp["r_M"], failed_requests=float(self.failed),
                    mean_latency_s=self.lat_sum / n,
                    # cycle 2: latency weighted by resolved requests (served+failed, failed at timeout), failures/arrivals, churn
                    req_weighted_latency_s=self.lat_w / max(self.req_w, 1e-12),
                    failed_over_arrived=self.failed / max(self.arrived, 1e-12), churn_pods=float(self.churn))
