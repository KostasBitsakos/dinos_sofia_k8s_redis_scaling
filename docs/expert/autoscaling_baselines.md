---
description: Which external autoscaling RL baseline(s) cycle 2 uses - DRe-SCale vs Gwydion decision and justification
---
# External baselines — decision for cycle 2
## Candidates (verified)
- **DRe-SCale** — Agarwal, Rodriguez, Buyya, "A Deep Recurrent-Reinforcement Learning Method for Intelligent AutoScaling
  of Serverless Functions", IEEE TSC 17(5) 1899-1910, 2024, DOI 10.1109/TSC.2024.3387661; arXiv 2308.05937;
  github.com/SidAg26/DRe-SCale, Apache-2.0; LSTM-PPO (RecurrentPPO), Poisson arrivals in its evaluation; code needs a live
  cluster (OpenFaaS/K8s/Prometheus) → we re-implement the agent in our sim. (V: `docs/related_work_sota_autoscaler.md:83-101`,
  `docs/related_work_sota_codex_r2.md:4-6`)
- **Gwydion** — J. Santos, E. Reppas, T. Wauters, B. Volckaert, F. De Turck, "Gwydion: Efficient auto-scaling for complex
  containerized applications in Kubernetes through Reinforcement Learning", J. Network and Computer Applications 234 (2025)
  104067, DOI 10.1016/j.jnca.2024.104067; github.com/jpedro1992/gwydion; extension of gym-hpa (Santos et al., NOMS 2023);
  Redis among the apps; dataset-replay simulation; **non-commercial research licence**. (V bibliographic: imec repository
  https://imec-publications.be/handle/20.500.12860/44979 and `related_work_sota_codex_r2.md:7`. Method details and numbers:
  NOT verified by me — do not cite them until the scholar opens the PDF.)
- Discuss, do not port: AWARE (USENIX ATC 2023), ASTRA (TSE 2025), FIRM (Qiu et al., OSDI 2020, B), Autopilot (Rzadca et
  al., EuroSys 2020, B). Survey: Garí et al., "RL-based application autoscaling in the cloud: a survey", Eng. Appl. AI 102
  (2021) (B).
## Decision (J12): DRe-SCale LSTM-PPO as THE ported baseline; Gwydion cited and discussed; port Gwydion only as stretch
Why, for reviewers:
1. **Algorithmic contrast.** All our learners are value-based (DQN/DDQN, LSTM variants). Reviewers' standard question is
   "why not a policy-gradient / PPO method?" DRe-SCale answers it with a peer-reviewed *recurrent* PPO — the direct
   competitor of our LSTM arms. Gwydion is a gym-hpa (PPO/A2C) extension: same family as DRe-SCale, so a second port adds
   less contrast per CPU-h.
2. **Venue fit and licence.** IEEE TSC + Buyya group = Cloud/Grid community (CCGrid); Apache-2.0 lets us publish our port.
   Gwydion's non-commercial licence is fine for research but limits redistribution of a derived port.
3. **Time/budget.** Deadline 2/11 and 160 CPU-h (R-20261029-001): one external arm × 10 seeds is ≈13 CPU-h at cycle-1 cost
   (1.26 CPU-h/run, test10: 58 CPU-h/46 runs); a second external arm takes budget from tuning the arms that matter.
4. **Where Gwydion matters:** it is Redis/Kubernetes-specific, close to our *real-cluster* phase. Recommend porting it for the
   real-cluster paper/extension, and citing it now as the closest application-level related work.
Caveat to state in the paper: "adapted DRe-SCale agent (architecture + hyperparameters) in our simulator", not a
reproduction of their numbers; disclose commit hash and every deviation.
Also required by reviewers: a **tuned load-based HPA** (our HPA-demand, tuned on val) in addition to his HPA, and a
no-transfer DDQN (= Dinos/LSTM arms).
