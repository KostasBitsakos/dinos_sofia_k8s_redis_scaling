---
description: Correct Poisson/NHPP/MMPP workload generation, noise, trace splits and a minimal queueing latency model (cycle 2)
---
# Poisson microservice workloads — how to generate them correctly
## Sources
- Lewis & Shedler, "Simulation of nonhomogeneous Poisson processes by thinning", Naval Research Logistics Quarterly 26(3),
  1979. (B)
- Fischer & Meier-Hellstern, "The Markov-modulated Poisson process (MMPP) cookbook", Performance Evaluation 18(2), 1993. (B)
- Gan et al., "DeathStarBench" (open-source microservice benchmark), ASPLOS 2019 — microservice request chains. (B)
- DRe-SCale's generator samples Poisson-distributed *gaps*, which is not a Poisson arrival process (Codex r2,
  `docs/related_work_sota_codex_r2.md:9`). (V)
## Correct generators (numpy Generator, one seed per trace)
- **Homogeneous Poisson, rate λ:** inter-arrival times ~ Exponential(mean 1/λ); cumulative sum; count per tick.
  Equivalent and faster per tick: N_t ~ Poisson(λ·Δt). Both are correct; the gap version is needed for request-level sim.
- **NHPP λ(t)** (diurnal/sinusoid + spikes): thinning — draw candidates at rate λ_max, accept each with prob λ(t)/λ_max.
  Per-tick shortcut: N_t ~ Poisson(∫_tick λ(s)ds) (exact for piecewise-constant λ per tick).
- **MMPP-2 (bursty)**: CTMC with states {normal, burst}, rates λ_n < λ_b, switching rates r_nb, r_bn; exponential sojourns;
  Poisson arrivals at the current state's rate. Gives over-dispersion (variance > mean) seen in real microservice traffic.
- **Noise.** Put noise in λ (e.g. log-normal multiplicative per minute, AR(1)), not by adding Gaussian noise to counts
  (counts must stay non-negative integers and the process stays Poisson given λ). Measurement noise on observed CPU/mem
  is separate and goes into the observation only.
- **Sanity tests (must be unit tests):** for homogeneous λ: mean≈variance of per-tick counts (index of dispersion ≈1),
  inter-arrival KS-test vs Exponential (p>0.01 at n≥10k); for MMPP: dispersion >1; reproducible per seed.
## Microservice queue → latency/CPU/memory (minimal, defendable)
- Each pod: c parallel workers, service rate μ per worker (requests/tick); pods = ready replicas after delay D.
- Per tick: arrivals + backlog served up to capacity pods·c·μ; leftover queues (bounded queue → drops count as SLO miss).
- Latency: per-tick mean and p95 from the queue (request-level sim or M/M/c Erlang-C approximation for waiting time);
  SLO = p95 ≤ L_slo. CPU util = served/capacity (+ small measurement noise); memory = base_per_pod + k·in-flight.
- F1 check: served load and latency must depend on pods (PLAN §2 F1).
