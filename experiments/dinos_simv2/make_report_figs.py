#!/usr/bin/env python3
"""Report figures that mirror the DInos (DEXA 2025) result figures, from the Sim v2.1 runs already done.

Read-only on the runs; no simulator call, no training, no new seeds (only the stored traces of selection seeds
1000-1004 and the stored per-episode returns of gate seeds 1200-1209 are used).
Everything produced is EXPLORATORY / PROVISIONAL (gates failed, simulator not calibrated).

Usage:  ~/qrl-lab/.venv/bin/python make_report_figs.py [--out DIR]
Writes: PNGs, report.html (images base64-embedded, Greek prose), report.md (relative image links).
"""
import argparse
import base64
import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / ".lab" / "runs"
R_PRE = RUNS / "20261017-140714-R-20261017-004-e0-a1"
R_DIAG = RUNS / "20261017-145935-R-20261017-010-e0-a1"
R_REV1 = RUNS / "20261017-152657-R-20261017-013-e0-a1"
HPA_DEV = Path(__file__).resolve().parent / "hpa_selection_dev.json"
PAPER = REPO / "docs" / "papers" / "dinos_dexa_paper2025.pdf"
SEEDS = [1000, 1001, 1002, 1003, 1004]       # selection seeds (in-sample for threshold / fixed(22))
COL = {"threshold": "tab:red", "static_ramp_best": "tab:orange", "fixed_best": "tab:blue",
       "oracle": "tab:green", "always_max": "tab:gray"}


def load_json(p):
    with open(p) as f:
        return json.load(f)


def load_trace(policy, seed):
    with open(R_DIAG / "traces" / f"trace_seed{seed}_{policy}.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def save(fig, out, name):
    fig.savefig(out / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return name


# ------------------------------------------------------------------ DInos figure extraction (pypdf + Pillow)
def extract_dinos(out):
    from PIL import Image
    import pypdf
    r = pypdf.PdfReader(str(PAPER))
    want = {"fig4": (5, 0), "fig5": (9, 0), "fig6_left": (10, 0), "fig6_right": (10, 1),
            "fig7_left": (11, 0), "fig7_right": (11, 1), "fig8": (12, 0)}
    got = {}
    for key, (page, idx) in want.items():
        try:
            data = r.pages[page].images[idx].data
            im = Image.open(io.BytesIO(data)).convert("RGB")
            if im.width > 900:
                im = im.resize((900, round(im.height * 900 / im.width)))
            name = f"dinos_{key}.png"
            im.save(out / name)
            got[key] = name
        except Exception as e:  # keep the report usable without the reference image
            print("could not extract", key, e)
    return got


# ------------------------------------------------------------------ data
def gather():
    D = {}
    D["tr"] = {p: {s: load_trace(p, s) for s in SEEDS} for p in ("threshold", "fixed_best", "oracle")}
    D["diag"] = load_json(R_DIAG / "diagnose.json")
    D["rev1"] = load_json(R_REV1 / "preflight_rev1.json")
    D["pre"] = load_json(R_PRE / "preflight.json")
    D["hpa"] = load_json(HPA_DEV)
    # consistency: sum of per-step reward == stored per-episode return (diagnose.json accounting)
    for p in D["tr"]:
        tot = np.mean([D["tr"][p][s]["reward"].sum() for s in SEEDS])
        ref = D["diag"]["accounting_mean"][p]["all"]["ret"]
        assert abs(tot - ref) < 1.0, (p, tot, ref)
    return D


# ------------------------------------------------------------------ figures
def fig4(D, out):
    t = D["tr"]["fixed_best"][1000]
    x = np.arange(len(t["t"]))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    spec = [(ax[0, 0], t["lambda_ops"], "Load (offered ops/s)", "tab:blue"),
            (ax[0, 1], t["obs_cpu_used"], "CPU Usage (% , fixed 22 pods)", "tab:orange"),
            (ax[1, 0], t["backlog"], "Backlog (ops, fixed 22 pods)", "tab:green"),
            (ax[1, 1], t["p95_ms"], "Latency (p95 ms, fixed 22 pods)", "tab:purple")]
    for a, y, lab, c in spec:
        a.plot(x, y, color=c, lw=0.7)
        a.set_xlabel("Time"); a.set_ylabel(lab); a.grid(alpha=0.3, ls="--")
    ax[1, 0].set_title("(no Memory metric in Sim v2.1; backlog shown instead)", fontsize=8)
    ax[1, 1].set_yscale("log")
    fig.suptitle("Sim v2.1, workload W0, seed 1000 (validation) - input metrics over 5000 timesteps", fontsize=10)
    fig.tight_layout()
    return save(fig, out, "fig4_input_metrics.png")


def fig5(D, out):
    rv = D["rev1"]["per_episode_return"]
    order = ["threshold", "static_ramp_best", "fixed_best", "always_max", "oracle"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
    ax[0].axis("off")
    ax[0].text(0.5, 0.5, "DERP / DInos at 1k, 2k, 5k, 10k, 20k steps\nNOT YET AVAILABLE\n(no agent trained yet;\nneeds W0 run R-20261017-015)",
               ha="center", va="center", fontsize=11, color="firebrick",
               bbox=dict(boxstyle="round", fc="#fff3f3", ec="firebrick"))
    ax[0].set_title("DERP (Dense) / DInos")
    m = [np.mean(rv[p]) for p in order]
    ax[1].bar(range(len(order)), m, color=[COL[p] for p in order])
    for i, p in enumerate(order):
        ax[1].scatter(np.full(len(rv[p]), i) + np.linspace(-0.2, 0.2, len(rv[p])), rv[p], c="k", s=9, zorder=3)
    ax[1].set_xticks(range(len(order)))
    ax[1].set_xticklabels(order, rotation=30, ha="right")
    ax[1].set_ylabel("Total Accumulated Reward")
    ax[1].grid(alpha=0.3, ls="--", axis="y")
    ax[1].set_title("Baselines only (Sim v2.1 W0, 10 episodes, seeds 1200-1209)\nalways_min (-3.5e6) omitted from the axis")
    lo = min(min(rv[p]) for p in order)
    ax[1].set_ylim(lo * 1.05, 0)
    fig.tight_layout()
    return save(fig, out, "fig5_checkpoints_analogue.png")


def four_panel_pods(tr, out, name, title):
    x = tr["t"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    spec = [(ax[0, 0], tr["lambda_ops"], "Load (ops/s)", "tab:blue", "Load Over Time"),
            (ax[0, 1], tr["backlog"], "Backlog (ops)", "tab:green", "Backlog Over Time (no Memory in Sim v2.1)"),
            (ax[1, 0], tr["obs_cpu_used"], "CPU Usage (%)", "orange", "CPU Usage Over Time"),
            (ax[1, 1], tr["p95_ms"], "Latency (p95 ms)", "tab:purple", "Latency Over Time")]
    for a, y, lab, c, ttl in spec:
        a.plot(x, y, color=c, lw=0.6, alpha=0.8)
        a.set_xlabel("Time"); a.set_ylabel(lab, color=c); a.set_title(ttl, fontsize=10)
        b = a.twinx()
        b.scatter(x, tr["ready"], s=4, c="red", alpha=0.35)
        b.set_ylabel("Number of Pods (ready)", color="red")
        b.set_ylim(0, 31)
    ax[1, 1].set_yscale("log")
    ax[0, 1].set_yscale("symlog", linthresh=1e3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return save(fig, out, name)


def fig6(D, out):
    a = four_panel_pods(D["tr"]["threshold"][1000], out, "fig6_left_threshold.png",
                        "Reactive baseline: threshold 40/25/0 (stand-in for the reactive DERP panel) - Sim v2.1 W0, seed 1000")
    b = four_panel_pods(D["tr"]["oracle"][1000], out, "fig6_right_oracle.png",
                        "Clairvoyant oracle replay (reference bound, NOT a trained DInos) - Sim v2.1 W0, seed 1000")
    return a, b


def fig7_left(D, out):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for p in ("threshold", "fixed_best", "oracle"):
        cs = np.mean([np.cumsum(D["tr"][p][s]["reward"]) for s in SEEDS], axis=0)
        ax.plot(cs, color=COL[p], label=f"{p} (final {cs[-1]:,.0f})")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Cumulative Reward (mean of seeds 1000-1004)")
    ax.set_title("Cumulative reward over one 5000-step episode (Sim v2.1 W0, baselines only)")
    ax.grid(alpha=0.3, ls="--"); ax.legend()
    return save(fig, out, "fig7_left_cumulative_reward.png")


def fig7_right(D, out):
    tr = D["tr"]["threshold"][1000]
    n = 2000
    x = tr["t"][:n]
    fig, ax = plt.subplots(2, 2, figsize=(11, 6))
    spec = [(ax[0, 0], tr["lambda_ops"], "Load", "Load Over Time"),
            (ax[0, 1], tr["obs_cpu_used"], "CPU Usage", "CPU Usage Over Time"),
            (ax[1, 0], tr["backlog"], "Backlog (no Memory in Sim v2.1)", "Backlog Over Time"),
            (ax[1, 1], tr["p95_ms"], "Latency (p95 ms)", "Latency Over Time")]
    for a, y, lab, ttl in spec:
        a.plot(x, y[:n], color="blue", lw=0.7)
        a.set_xlabel("Timesteps"); a.set_ylabel(lab); a.set_title(ttl, fontsize=10)
        b = a.twinx()
        b.step(x, tr["ready"][:n], color="red", ls="--", lw=1)
        b.set_ylabel("Number of Pods")
    ax[1, 1].set_yscale("log")
    fig.suptitle("SIMULATOR trace, first 2000 steps, threshold baseline (the DInos figure is a real Kubernetes/Redis run)", fontsize=9)
    fig.tight_layout()
    return save(fig, out, "fig7_right_sim_timeseries.png")


def table_stats(D):
    """Fig. 8 analogue: per policy, mean over seeds 1000-1004 of return, CPU, p95 and pods."""
    rows = []
    for p in ("threshold", "fixed_best", "oracle"):
        ts = [D["tr"][p][s] for s in SEEDS]
        rows.append(dict(method=p, ret=np.mean([t["reward"].sum() for t in ts]),
                         cpu=np.mean([t["obs_cpu_used"].mean() for t in ts]),
                         p95=np.mean([t["p95_ms"].mean() for t in ts]),
                         pods=np.mean([t["ready"].mean() for t in ts])))
    return rows


def fig8(D, out, rows):
    cells = [["DERP", "Simulation (Sim v2.1)", "not yet available", "-", "-", "-"],
             ["DInos (with transfer)", "Simulation (Sim v2.1)", "not yet available", "-", "-", "-"]]
    for r in rows:
        cells.append([r["method"] + " (baseline)", "Simulation (Sim v2.1)", f"{r['ret']:,.0f}",
                      f"{r['cpu']:.1f}", f"{r['p95']:.1f}", f"{r['pods']:.1f}"])
    cells.append(["any agent", "Real cluster", "not available", "-", "-", "-"])
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=["Method", "Environment", "Accumulated Reward", "mean CPU %",
                                              "mean p95 ms", "mean pods"], loc="center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.5)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_width([0.22, 0.24, 0.18, 0.12, 0.12, 0.12][c_])
    return save(fig, out, "fig8_summary_table.png"), cells


def extra_gate(D, out):
    g = D["rev1"]["gate"]["controls"]["static_ramp_best"]
    d = np.array(g["per_episode_diff"])
    seeds = D["rev1"]["eval_seeds"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(d)), d, color="tab:red")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(g["ci_lo"], g["ci_hi"], color="gold", alpha=0.3, label=f"95% CI of mean [{g['ci_lo']:,.0f}, {g['ci_hi']:,.0f}]")
    ax.set_xticks(range(len(d))); ax.set_xticklabels(seeds, rotation=45)
    ax.set_xlabel("evaluation seed"); ax.set_ylabel("return(threshold) - return(static_ramp)")
    ax.set_title("Rev-1 gate: threshold minus static_ramp per episode (needs > 0 to pass)")
    ax.legend(); ax.grid(alpha=0.3, ls="--", axis="y")
    return save(fig, out, "extra_rev1_gate.png")


def extra_hpa(D, out):
    h = D["hpa"]
    g, v = zip(*h["table"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g, v, "o-")
    ax.axvline(h["selected_target_pct"], color="r", ls="--", label=f"selected target {h['selected_target_pct']}%")
    ax.set_ylim(-400000, -100000)
    ax.set_xlabel("K8sHPA target CPU utilisation (%)"); ax.set_ylabel("Mean return, dev seeds 1000-1004")
    ax.set_title("K8sHPA baseline tuning (dev seeds only; target >=60% falls off the axis)")
    ax.legend(); ax.grid(alpha=0.3, ls="--")
    return save(fig, out, "extra_hpa_tuning.png")


def extra_sens(D, out):
    s = D["diag"]["sensitivity"]
    names = [k for k in s]
    thr = [s[k]["mean_return"]["threshold"] for k in names]
    fx = [s[k]["mean_return"]["fixed_best"] for k in names]
    orc = [s[k]["mean_return"]["oracle"] for k in names]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.27
    X = np.arange(len(names))
    ax.bar(X - w, thr, w, label="threshold", color=COL["threshold"])
    ax.bar(X, fx, w, label="fixed_best (22)", color=COL["fixed_best"])
    ax.bar(X + w, orc, w, label="oracle", color=COL["oracle"])
    ax.set_xticks(X); ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylim(-400000, 0)
    ax.set_ylabel("Mean return (seeds 1000-1004)")
    ax.set_title("Sim v2.1 sensitivity, one knob at a time (baselines NOT re-tuned; bars clipped at -4e5)")
    ax.legend(); ax.grid(alpha=0.3, ls="--", axis="y")
    return save(fig, out, "extra_sensitivity.png")


# ------------------------------------------------------------------ report text
def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def build_report(out, D, ref, figs, rows, cells):
    rv = D["rev1"]["mean_return"]
    g = D["rev1"]["gate"]["controls"]["static_ramp_best"]
    pre = D["pre"]["summary"]
    pre_ret = {p: pre[p]["ret"] for p in pre}
    sens = D["diag"]["sensitivity"]
    hpa = D["hpa"]
    hsel = dict(hpa["table"])[hpa["selected_target_pct"]]
    fx_dev = D["diag"]["accounting_mean"]["fixed_best"]["all"]["ret"]
    thr_dev = D["diag"]["accounting_mean"]["threshold"]["all"]["ret"]
    orc_dev = D["diag"]["accounting_mean"]["oracle"]["all"]["ret"]
    k = lambda v: f"{v/1000:,.1f}k"

    top = [
        ("Τι αλλάξαμε", [
            "Sim v2 -> v2.1: νέα knobs ευαισθησίας (latency_add, tail_scale, cpu_offset, rebalance k/delta, theta) με παγωμένες τιμές στο `code/sim_v2/sensitivity_v21.json`. Οι default τιμές δίνουν bit-identical αποτελέσματα με το v2.",
            "Πρωτόκολλο dinos_simv2: freeze (hash + git), preflight, diagnose, rev-1 gate με seeds 1200-1209, παγωμένος baseline K8sHPA (`docs/hpa_spec.md`).",
            "ΟΛΑ τα παράμετρα του simulator είναι προσωρινά και μη βαθμονομημένα.",
        ]),
        ("Τι τρέξαμε", [
            "R-20261017-004 (preflight): threshold vs fixed(22), 5 seeds (1000-1004).",
            "R-20261017-010 (diagnose): traces ανά βήμα, accounting startup/rest, sensitivity (ίδια 5 seeds, χωρίς re-tuning).",
            "R-20261017-013 (rev-1 gate): c0=6, 10 seeds (1200-1209), threshold vs static_ramp(22), fixed_best, oracle.",
            "Δεν έχει εκπαιδευτεί κανένας RL agent (DERP, DInos, LSTM) ακόμη. Το W0 run R-20261017-015 δεν έχει τρέξει.",
        ]),
        ("Τι δοκιμάσαμε", [
            "Αν ο threshold (40/25/0) ξεπερνά απλούς ελεγκτές: static_ramp(22) (ξεκινά από 6 pods και ανεβαίνει), fixed(22) προ-παρασχεμένο, always_min/max, και clairvoyant oracle.",
            "Αν η κατάταξη αντέχει σε αλλαγές των παραμέτρων (sensitivity, 10 configs).",
        ]),
        ("Τι βγήκε", [
            f"Preflight (R-004): threshold {k(pre_ret['threshold'])} vs fixed(22) {k(pre_ret['fixed_best'])}, νίκες 0/5. FAIL.",
            f"Rev-1 (R-013): threshold {k(rv['threshold'])} vs static_ramp {k(rv['static_ramp_best'])}, διαφορά {g['mean_diff']/1000:,.1f}k, 95% CI [{g['ci_lo']/1000:,.1f}k, {g['ci_hi']/1000:,.1f}k], νίκες {g['wins']}/{g['n']}. FAIL. Oracle {k(rv['oracle'])}, fixed_best {k(rv['fixed_best'])}, always_max {k(rv['always_max'])}, always_min {rv['always_min']/1e6:,.2f}M.",
            f"K8sHPA (dev seeds 1000-1004): καλύτερο target {hpa['selected_target_pct']}% με {k(hsel)} (threshold {k(thr_dev)}, fixed(22) {k(fx_dev)} στα ίδια seeds). Μόνο dev, in-sample.",
            "Συμπέρασμα: τα gates απέτυχαν, ο simulator δεν είναι βαθμονομημένος. Όλα τα παρακάτω είναι διερευνητικά (exploratory) και προσωρινά.",
        ]),
    ]

    # mapping table: (dinos fig, our file, status)
    mapping = [
        ("Fig 4", "Input metrics (load, CPU, memory, latency, 5000 steps)", figs["fig4"], "Αντίστοιχο (αναλογία): load/CPU/latency από το sim. Η μνήμη δεν μοντελοποιείται, δείχνουμε backlog."),
        ("Fig 5", "DERP/DInos reward ανά checkpoint (1k-20k)", figs["fig5"], "ΔΕΝ ΥΠΑΡΧΕΙ ακόμη (χωρίς agents). Δείχνουμε baselines. Χρειάζεται R-20261017-015."),
        ("Fig 6 left", "DERP: metrics + pods, 5000 steps", figs["fig6l"], "Αναλογία: αντιδραστικός threshold baseline, όχι DERP."),
        ("Fig 6 right", "DInos: metrics + pods, 5000 steps", figs["fig6r"], "Αναλογία: clairvoyant oracle (όριο), ΟΧΙ DInos."),
        ("Fig 7 left", "Real cluster: αθροιστικό reward 2000 steps, DInos vs LSTM-only", figs["fig7l"], "Μερική αναλογία: αθροιστικό reward baselines στο sim. Real run ΔΕΝ υπάρχει."),
        ("Fig 7 right", "Real cluster: metrics + Redis pods", figs["fig7r"], "Μερική αναλογία: sim trace (ίδιο στυλ). Real run ΔΕΝ υπάρχει."),
        ("Fig 8", "Πίνακας reward/CPU/μνήμη/latency, DERP vs DInos, sim και real", figs["fig8"], "Μερικός πίνακας: baselines στο sim. DERP/DInos/real: not available."),
    ]

    # per-figure explanation (Greek, 2-4 sentences)
    thr_r, fx_r, orc_r = rows[0], rows[1], rows[2]
    expl = {
        "fig4": f"Δείχνει τα μεγέθη εισόδου του simulator σε ένα επεισόδιο 5000 βημάτων (seed 1000): offered load (ημιτονοειδές με spike), CPU, backlog και p95 latency με 22 σταθερά pods. Στο DInos το load έχει επίσης ημίτονο, θόρυβο και spike στα 1500-1600. Στο δικό μας W0 το spike διαρκεί 100 βήματα σε τυχαία θέση ανά seed (`code/sim_v2/workloads.py`), και η φάση του ημιτόνου είναι επίσης τυχαία ανά seed. Οι κλίμακες διαφέρουν (ops/s, ms) και δεν υπάρχει μνήμη, άρα η σύγκριση είναι μόνο ποιοτική.",
        "fig5": f"Το DInos δείχνει reward ανά αριθμό βημάτων εκπαίδευσης για DERP και DInos. Εμείς δεν έχουμε εκπαιδευμένους agents, άρα το αριστερό πάνελ είναι κενό (not yet available). Δεξιά βλέπεις το συνολικό reward των baselines σε 10 επεισόδια (seeds 1200-1209, rev-1): threshold {k(rv['threshold'])}, static_ramp {k(rv['static_ramp_best'])}, fixed_best {k(rv['fixed_best'])}, always_max {k(rv['always_max'])}, oracle {k(rv['oracle'])}. Το reward είναι αρνητικό (κόστος), όχι θετικό όπως στο DInos, άρα οι τιμές δεν συγκρίνονται άμεσα.",
        "fig6l": f"Στο DInos το αριστερό διάγραμμα δείχνει τον DERP που αντιδρά μετά το γεγονός. Εδώ χρησιμοποιούμε τον threshold baseline (όχι DERP) ως αντιδραστικό ελεγκτή: ανεβαίνει σε 30 pods (μέγιστο) κατά την εκκίνηση και μετά ταλαντεύεται γύρω από το φορτίο. Το reward έχει ποινή SLA, άρα η αρχική φάση (startup) κοστίζει. Μην διαβάσεις το σχήμα ως αποτέλεσμα DERP.",
        "fig6r": f"Στο DInos το δεξί διάγραμμα είναι ο DInos με λίγα pods. Εδώ βάζουμε τον oracle replay, που ξέρει το μέλλον και είναι άνω όριο, όχι εκπαιδευμένος agent. Χρησιμοποιεί λιγότερα pods (μέσος όρος {orc_r['pods']:.1f} έναντι {thr_r['pods']:.1f} του threshold) και έχει return {orc_dev/1000:,.1f}k (seeds 1000-1004) έναντι {thr_dev/1000:,.1f}k. Δείχνει πόσο περιθώριο υπάρχει, όχι ότι το DInos θα το πετύχει.",
        "fig7l": f"Το DInos δείχνει αθροιστικό reward 2000 βημάτων σε πραγματικό cluster (308,000 vs 56,000). Εμείς δεν έχουμε real-cluster run αυτού του τύπου. Δείχνουμε το αθροιστικό reward των baselines στο sim (μέσος όρος seeds 1000-1004, 5000 βήματα): τελικά threshold {thr_dev:,.0f}, fixed(22) {fx_dev:,.0f}, oracle {orc_dev:,.0f}. Ο threshold ξεκινά από 5-6 pods και ανεβαίνει σταδιακά, ενώ το fixed(22) είναι προ-παρασχεμένο, άρα η εκκίνηση (πρώτα 150 βήματα) είναι ένα μέρος της διαφοράς (accounting στο diagnose: startup threshold -18.4k έναντι fixed -3.5k).",
        "fig7r": "Στο DInos είναι ο χρονικός ρυθμός metrics και pods από ζωντανό Kubernetes/Redis. Εδώ δείχνουμε τα πρώτα 2000 βήματα ενός trace του simulator (threshold baseline) στο ίδιο στυλ (μπλε γραμμή, κόκκινα διακεκομμένα pods). Δεν είναι πραγματική μέτρηση και δεν υπάρχει agent. Οι calibration runs του Οκτωβρίου μετρούν το cluster, δεν τρέχουν autoscaler.",
        "fig8": f"Ο πίνακας του DInos συγκρίνει reward, CPU, μνήμη και latency. Εδώ ο πίνακας έχει μόνο baselines στο sim (seeds 1000-1004, mean): threshold {thr_r['ret']:,.0f}, fixed(22) {fx_r['ret']:,.0f}, oracle {orc_r['ret']:,.0f}. Οι γραμμές DERP, DInos και real cluster είναι κενές (not available). Η μνήμη δεν υπάρχει στο sim, το CPU είναι το παρατηρούμενο (με θόρυβο), το latency είναι ο μέσος p95 σε ms. Ο μέσος p95 του threshold κυριαρχείται από το startup (αρχικό backlog). Τα threshold/fixed(22) είχαν επιλεγεί στα ίδια seeds (in-sample).",
    }

    extras = [
        ("extra_rev1", "Έξτρα A: rev-1 gate, διαφορά ανά επεισόδιο", figs["xg"],
         f"Κάθε μπάρα είναι return(threshold) μείον return(static_ramp) στο ίδιο seed. Όλες είναι αρνητικές: 0/10 νίκες, μέση διαφορά {g['mean_diff']:,.0f}, CI [{g['ci_lo']:,.0f}, {g['ci_hi']:,.0f}]. Άρα ο threshold δεν περνά το gate. Πηγή: {R_REV1.name}/preflight_rev1.json."),
        ("extra_hpa", "Έξτρα B: ρύθμιση K8sHPA (dev seeds)", figs["xh"],
         f"Το mean return του K8sHPA ανά target CPU %. Βέλτιστο στο {hpa['selected_target_pct']}% ({hsel:,.0f}), εσωτερικό σημείο του grid. Μόνο dev seeds, δεν είναι αποτέλεσμα αξιολόγησης. Πηγή: code/experiments/dinos_simv2/hpa_selection_dev.json."),
        ("extra_sens", "Έξτρα Γ: ευαισθησία του simulator", figs["xs"],
         f"Mean return ανά config (ένα knob τη φορά, χωρίς re-tuning). Ο συσχετισμός threshold/fixed αλλάζει: π.χ. θ=0.9 threshold {k(sens['theta_0.9']['mean_return']['threshold'])} vs fixed {k(sens['theta_0.9']['mean_return']['fixed_best'])}, rebalance_asm_delta_0.05 threshold {k(sens['rebalance_asm_delta_0.05']['mean_return']['threshold'])} vs fixed {k(sens['rebalance_asm_delta_0.05']['mean_return']['fixed_best'])}. Άρα η κατάταξη εξαρτάται από παραμέτρους που δεν έχουμε μετρήσει. θ=0.8 είναι εκτός κλίμακας (clip στο -4e5). Πηγή: {R_DIAG.name}/diagnose.json."),
    ]
    return top, mapping, expl, extras


CSS = """body{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#fafafa;color:#222;margin:0}
main{max-width:1100px;margin:0 auto;padding:12px 16px}
h1{font-size:1.5em}h2{font-size:1.25em;border-bottom:1px solid #ccc;padding-bottom:4px;margin-top:2em}
.warn{background:#fff3cd;border:1px solid #e0c36a;padding:8px 12px;border-radius:6px}
.cols{display:flex;gap:12px;flex-wrap:wrap}.col{flex:1 1 360px;min-width:0}
.col img{max-width:100%;height:auto;border:1px solid #ddd;background:#fff}
.cap{font-size:.85em;color:#555}table{border-collapse:collapse;width:100%;font-size:.9em}
td,th{border:1px solid #ccc;padding:4px 6px;text-align:left;vertical-align:top}th{background:#eee}
.na{color:#b00020;font-weight:600}"""

DINOS_CAP = {
    "fig4": "Fig 4 (DInos): input metrics from the simulation (load, CPU, memory, latency) over 5000 timesteps; sinusoid + Gaussian noise + spike at 1500-1600.",
    "fig5": "Fig 5 (DInos): total accumulated reward of DERP (left) and DInos (right) at 1k, 2k, 5k, 10k, 20k training steps.",
    "fig6l": "Fig 6 left (DInos): DERP runtime metrics and pod count over 5000 timesteps; reactive, over/under-provisions.",
    "fig6r": "Fig 6 right (DInos): DInos runtime metrics and pod count over 5000 timesteps; anticipates load.",
    "fig7l": "Fig 7 left (DInos): real Kubernetes/Redis, cumulative reward over 2000 timesteps, DInos 308,000 vs LSTM-only 56,000.",
    "fig7r": "Fig 7 right (DInos): real cluster, load/CPU/memory/latency and number of Redis pods over 2000 timesteps.",
    "fig8": "Fig 8 (DInos): summary table, DERP vs DInos, simulation and real world (reward 220k vs 3.8M; 56k vs 308k).",
}
REFKEY = {"fig4": ["fig4"], "fig5": ["fig5"], "fig6l": ["fig6_left"], "fig6r": ["fig6_right"],
          "fig7l": ["fig7_left"], "fig7r": ["fig7_right"], "fig8": ["fig8"]}
DINOS_FIGPATH = {"fig4": "fig4", "fig5": "fig5", "fig6l": "fig6_left", "fig6r": "fig6_right",
                 "fig7l": "fig7_left", "fig7r": "fig7_right", "fig8": "fig8"}


def write_html_md(out, top, mapping, expl, extras, ref, figs, rows):
    DISC = ("ΔΙΕΡΕΥΝΗΤΙΚΟ / ΠΡΟΣΩΡΙΝΟ: τα gates (preflight, rev-1) απέτυχαν και ο simulator δεν είναι βαθμονομημένος. "
            "Τα νούμερα δεν είναι αποτέλεσμα για paper. Κανένας RL agent δεν έχει εκπαιδευτεί σε αυτά τα runs.")
    keys = ["fig4", "fig5", "fig6l", "fig6r", "fig7l", "fig7r", "fig8"]
    h, m = [], []
    h.append(f"<!doctype html><html lang='el'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>QRL-Lab: σύνοψη Sim v2.1</title><style>{CSS}</style></head><body><main>")
    h.append("<h1>QRL-Lab: σύνοψη υλοποιήσεων και πειραμάτων (Sim v2.1)</h1>")
    h.append(f"<p class='warn'>{DISC}</p>")
    m.append("# QRL-Lab: σύνοψη υλοποιήσεων και πειραμάτων (Sim v2.1)\n\n> " + DISC + "\n")
    for title, bullets in top:
        h.append(f"<h2>{title}</h2><ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
        m.append(f"\n## {title}\n\n" + "\n".join(f"- {b}" for b in bullets) + "\n")
    # mapping table
    h.append("<h2>Αντιστοίχιση σχημάτων DInos ↔ δικά μας</h2><table><tr><th>DInos</th><th>Τι δείχνει</th><th>Δικό μας σχήμα</th><th>Κατάσταση</th></tr>")
    m.append("\n## Αντιστοίχιση σχημάτων DInos ↔ δικά μας\n\n| DInos | Τι δείχνει | Δικό μας σχήμα | Κατάσταση |\n|---|---|---|---|")
    fnames = {"fig4": "fig4_input_metrics.png", "fig5": "fig5_checkpoints_analogue.png",
              "fig6l": "fig6_left_threshold.png", "fig6r": "fig6_right_oracle.png",
              "fig7l": "fig7_left_cumulative_reward.png", "fig7r": "fig7_right_sim_timeseries.png",
              "fig8": "fig8_summary_table.png"}
    for (fg, what, fn, status) in mapping:
        cls = " class='na'" if ("ΔΕΝ" in status or "not available" in status) else ""
        h.append(f"<tr><td>{fg}</td><td>{what}</td><td>{fn}</td><td{cls}>{status}</td></tr>")
        m.append(f"| {fg} | {what} | {fn} | {status} |")
    h.append("</table><p class='cap'>Τα Fig 1-3 του DInos είναι αρχιτεκτονικά διαγράμματα, όχι αποτελέσματα, και δεν αναπαράγονται.</p>")
    m.append("\nΤα Fig 1-3 του DInos είναι αρχιτεκτονικά διαγράμματα, όχι αποτελέσματα, και δεν αναπαράγονται.\n")

    for key in keys:
        fg = [x for x in mapping if figs[{"fig4": "fig4", "fig5": "fig5", "fig6l": "fig6l", "fig6r": "fig6r",
                                          "fig7l": "fig7l", "fig7r": "fig7r", "fig8": "fig8"}[key]] == x[2]][0]
        title = f"{fg[0]}: {fg[1]}"
        ref_name = ref.get(DINOS_FIGPATH[key])
        h.append(f"<h2>{title}</h2><div class='cols'><div class='col'>")
        m.append(f"\n## {title}\n")
        h.append(f"<p class='cap'>{DINOS_CAP[key]}</p>")
        m.append(f"*DInos:* {DINOS_CAP[key]}\n")
        if ref_name:
            h.append(f"<img alt='DInos {fg[0]}' src='data:image/png;base64,{b64(out / ref_name)}'>")
            m.append(f"![DInos {fg[0]}]({ref_name})\n")
        h.append(f"</div><div class='col'><p class='cap'>Δικό μας ({fg[3]})</p>")
        m.append(f"*Δικό μας ({fg[3]}):*\n")
        h.append(f"<img alt='ours {fg[0]}' src='data:image/png;base64,{b64(out / fg[2])}'></div></div>")
        m.append(f"![ours {fg[0]}]({fg[2]})\n")
        h.append(f"<p>{expl[key]}</p>")
        m.append(expl[key] + "\n")
        if key == "fig8":
            cells = ["Method", "Environment", "Accumulated Reward", "mean CPU %", "mean p95 ms", "mean pods"]
            h.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
            for r in fig8_cells:
                h.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
            h.append("</table>")
            m.append("\n| " + " | ".join(cells) + " |\n|---|---|---|---|---|---|\n" +
                     "\n".join("| " + " | ".join(r) + " |" for r in fig8_cells) + "\n")
    h.append("<h2>Επιπλέον σχήματα (χωρίς αντίστοιχο στο DInos)</h2>")
    m.append("\n## Επιπλέον σχήματα (χωρίς αντίστοιχο στο DInos)\n")
    for _, title, fn, text in extras:
        h.append(f"<h3>{title}</h3><div class='col'><img src='data:image/png;base64,{b64(out / fn)}' alt='{title}'></div><p>{text}</p>")
        m.append(f"\n### {title}\n\n![{title}]({fn})\n\n{text}\n")
    note = ("Σημείωση παραγωγής: τα σχήματα φτιάχνονται από `code/experiments/dinos_simv2/make_report_figs.py` με ανάγνωση "
            "των αποθηκευμένων traces (seeds 1000-1004) και των per-episode returns (seeds 1200-1209). Δεν έτρεξε simulator, "
            "εκπαίδευση ή νέα seeds. Τα seeds 1300+ και 20000+ δεν αγγίχθηκαν. Τα traces των 1000-1004 είναι in-sample "
            "(threshold και fixed(22) επιλέχθηκαν εκεί). Το static_ramp δεν έχει traces ανά βήμα, μόνο returns.")
    h.append(f"<p class='cap'>{note}</p></main></body></html>")
    m.append("\n" + note + "\n")
    (out / "report.html").write_text("\n".join(h), encoding="utf-8")
    (out / "report.md").write_text("\n".join(m), encoding="utf-8")


fig8_cells = []


def main():
    global fig8_cells
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "reports" / "2026-10-17-sim-summary"))
    ap.add_argument("--seed", type=int, default=0, help="unused: deterministic, reads stored outputs only")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = gather()
    ref = extract_dinos(out)
    rows = table_stats(D)
    f6l, f6r = fig6(D, out)
    f8, fig8_cells = fig8(D, out, rows)
    figs = dict(fig4=fig4(D, out), fig5=fig5(D, out), fig6l=f6l, fig6r=f6r, fig7l=fig7_left(D, out),
                fig7r=fig7_right(D, out), fig8=f8, xg=extra_gate(D, out), xh=extra_hpa(D, out),
                xs=extra_sens(D, out))
    top, mapping, expl, extras = build_report(out, D, ref, figs, rows, fig8_cells)
    write_html_md(out, top, mapping, expl, extras, ref, figs, rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
