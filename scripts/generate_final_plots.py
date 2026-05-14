"""
generate_final_plots.py
=======================
Generates 4 publication-quality thesis plots.
Dependencies: csv, numpy, matplotlib only (no pandas).

If ALL_SEEDS_combined_metrics.csv / ALL_SEEDS_combined_summary.csv are absent
the script auto-combines the standalone IC and SI run CSVs:
    results/benchmark_op_ic/ALL_SEEDS_combined_metrics.csv
    results/benchmark_op_si/ALL_SEEDS_combined_metrics.csv  (same for summary)

Outputs saved to the current working directory:
    ttr_comparison_both_conditions.png
    phase_reward_comparison.png
    supervisor_activation_decay.png
    recovery_rate_summary.png
"""

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# ── File paths ────────────────────────────────────────────────────────────────
METRICS_CSV = "ALL_SEEDS_combined_metrics.csv"
SUMMARY_CSV = "ALL_SEEDS_combined_summary.csv"
IC_METRICS  = "results/benchmark_op_ic/ALL_SEEDS_combined_metrics.csv"
IC_SUMMARY  = "results/benchmark_op_ic/ALL_SEEDS_combined_summary.csv"
SI_METRICS  = "results/benchmark_op_si/ALL_SEEDS_combined_metrics.csv"
SI_SUMMARY  = "results/benchmark_op_si/ALL_SEEDS_combined_summary.csv"

# ── Experiment constants ──────────────────────────────────────────────────────
SEEDS    = list(range(42, 52))
SAB_EP   = 75
EVAL_EP  = 125
TOTAL_EP = 150
NR_H     = 75

# ── Strict colour scheme ──────────────────────────────────────────────────────
C_HPD   = "#1565C0"
C_VAN   = "#5C5C5C"
C_P1_BG = "#E0F7F4"
C_P2_BG = "#FFD6D6"
C_P3_BG = "#D4EDDA"
C_NR    = "#FFD6D6"
C_SAB   = "#D32F2F"
C_EVAL  = "#2E7D32"
C_HPD_L = "#90CAF9"

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":             9,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.spines.left":      True,
    "axes.spines.bottom":    True,
    "axes.linewidth":        0.8,
    "axes.grid":             True,
    "axes.grid.axis":        "y",
    "grid.linestyle":        "-",
    "grid.linewidth":        0.4,
    "grid.alpha":            0.3,
    "axes.labelsize":        10,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       8.5,
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
    "lines.linewidth":       1.5,
})

LABEL_HPD = "HPD Framework"
LABEL_VAN = "Vanilla DQN"
SAB_DISPLAY = {
    "inverted_controls": "Inverted Controls",
    "sensor_inversion":  "Sensor Inversion",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def _read_csv(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_or_combine(primary: str, fa: str, fb: str) -> List[Dict]:
    if os.path.exists(primary):
        return _read_csv(primary)
    if os.path.exists(fa) and os.path.exists(fb):
        print(f"  '{primary}' not found — combining standalone run CSVs.")
        return _read_csv(fa) + _read_csv(fb)
    sys.exit(f"ERROR: Cannot find '{primary}' or the standalone run CSVs. Aborting.")


def load_data():
    metrics = _load_or_combine(METRICS_CSV, IC_METRICS, SI_METRICS)
    summary = _load_or_combine(SUMMARY_CSV, IC_SUMMARY, SI_SUMMARY)

    # Parse numeric fields in metrics
    for r in metrics:
        r["seed"]      = int(r["seed"])
        r["episode"]   = int(r["episode"])
        r["reward"]    = float(r["reward"])
        r["llm_calls"] = float(r["llm_calls"])

    # Parse summary; ttr_num = float or None
    for r in summary:
        r["seed"] = int(r["seed"])
        raw = r.get("ttr_episodes", "not_recovered")
        r["ttr_num"] = None if raw == "not_recovered" else int(raw)
        r["mean_reward"] = float(r["mean_reward"])

    # Print summary
    sab_types = sorted({r["sabotage_type"] for r in summary})
    print("\n=== Data summary ===")
    print(f"  Metrics rows : {len(metrics):,}")
    print(f"  Summary rows : {len(summary):,}")
    for sab in sab_types:
        sub = [r for r in summary if r["sabotage_type"] == sab]
        print(f"\n  [{sab}]")
        for agent in ["hpd", "vanilla"]:
            a = [r for r in sub if r["agent"] == agent]
            nr = sum(1 for r in a if r["ttr_num"] is None)
            ttrs = [r["ttr_num"] for r in a if r["ttr_num"] is not None]
            mean_ttr = np.mean(ttrs) if ttrs else float("nan")
            mean_rew = np.mean([r["mean_reward"] for r in a])
            print(f"    {agent:<8}  seeds={len(a)}  NR={nr}  "
                  f"mean_TTR={mean_ttr:.1f}  mean_reward={mean_rew:.1f}")
    print()
    return metrics, summary


def _sab_types(data: List[Dict]) -> List[str]:
    order = ["inverted_controls", "sensor_inversion"]
    found = {r["sabotage_type"] for r in data}
    return [s for s in order if s in found]


# ── Shared helpers ────────────────────────────────────────────────────────────

def add_phase_backgrounds(ax):
    ax.axvspan(0,       SAB_EP,   color=C_P1_BG, zorder=0)
    ax.axvspan(SAB_EP,  EVAL_EP,  color=C_P2_BG, zorder=0)
    ax.axvspan(EVAL_EP, TOTAL_EP, color=C_P3_BG, zorder=0)


def add_phase_vlines(ax):
    ax.axvline(SAB_EP,  color=C_SAB,  linestyle="--", linewidth=1.6,
               zorder=4, label="Sabotage (ep 75)")
    ax.axvline(EVAL_EP, color=C_EVAL, linestyle=":",  linewidth=1.8,
               zorder=4, label="Evaluation (ep 125)")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — TTR Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_ttr_comparison(summary: List[Dict]):
    present = _sab_types(summary)
    if not present:
        print("WARNING: No data for TTR plot. Skipping."); return

    bar_w = 0.36
    x = np.arange(len(SEEDS))

    for sab in present:
        fig, ax = plt.subplots(figsize=(10, 6))
        sub = [r for r in summary if r["sabotage_type"] == sab]
        hpd_map = {r["seed"]: r["ttr_num"] for r in sub if r["agent"] == "hpd"}
        van_map = {r["seed"]: r["ttr_num"] for r in sub if r["agent"] == "vanilla"}

        for i, (ttr_map, color, label) in enumerate([
            (hpd_map, C_HPD, LABEL_HPD),
            (van_map, C_VAN, LABEL_VAN),
        ]):
            xpos = x + (i - 0.5) * bar_w
            for j, (seed, xp) in enumerate(zip(SEEDS, xpos)):
                v = ttr_map.get(seed)
                is_nr = (v is None)
                h = NR_H if is_nr else int(v)
                ax.bar(xp, h, width=bar_w,
                       color=C_NR if is_nr else color,
                       alpha=0.55 if is_nr else 0.88,
                       edgecolor="white", linewidth=0.8,
                       hatch="////" if is_nr else "",
                       label=label if j == 0 else "_")
                if is_nr:
                    ax.text(xp, h + 1.0, "NR", ha="center", va="bottom",
                            fontsize=8.5, fontweight="bold", color=C_SAB)
                else:
                    ax.text(xp, h + 1.0, str(h), ha="center", va="bottom",
                            fontsize=8.5, color="#222222")

        # Mean TTR lines (recovered seeds only)
        hpd_vals = [v for v in hpd_map.values() if v is not None]
        van_vals = [v for v in van_map.values() if v is not None]
        if hpd_vals:
            hm = np.mean(hpd_vals)
            ax.axhline(hm, color=C_HPD, linestyle="--", linewidth=1.4,
                       alpha=0.8, label=f"HPD Mean TTR = {hm:.1f}")
        if van_vals:
            vm = np.mean(van_vals)
            ax.axhline(vm, color=C_VAN, linestyle="--", linewidth=1.4,
                       alpha=0.8, label=f"Vanilla Mean TTR = {vm:.1f} (recovered only)")

        ax.set_title(f"Time-to-Recovery — {SAB_DISPLAY.get(sab, sab)}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Random Seed", fontsize=11)
        ax.set_ylabel("Episodes to Recovery  (lower = better)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in SEEDS])
        ax.set_ylim(0, NR_H + 16)
        ax.legend(fontsize=8.5, framealpha=0.9)

        fig.text(0.5, -0.02,
                 "NR = Not Recovered within benchmark window (75-episode limit)",
                 ha="center", fontsize=9, color="#555555", style="italic")
        plt.tight_layout()
        out = f"ttr_comparison_{sab}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Phase Reward Comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase_reward_comparison(metrics: List[Dict]):
    present = _sab_types(metrics)
    if not present:
        print("WARNING: No data for phase reward plot. Skipping."); return

    phases = [
        ("Nominal\n(ep 0–74)",        0,       SAB_EP,   C_P1_BG),
        ("Recovery\n(ep 75–124)",      SAB_EP,  EVAL_EP,  C_P2_BG),
        ("Evaluation\n(ep 125–149)",   EVAL_EP, TOTAL_EP, C_P3_BG),
    ]

    bar_w = 0.30
    x = np.arange(len(phases))

    for sab in present:
        fig, ax = plt.subplots(figsize=(10, 6))
        sub = [r for r in metrics if r["sabotage_type"] == sab]

        # Phase background bands behind each group
        for pi, (_, ep_lo, ep_hi, bg) in enumerate(phases):
            ax.axvspan(pi - 0.48, pi + 0.48, color=bg, zorder=0)

        hpd_means = []
        all_stds  = []

        for j, (agent, color, label) in enumerate([
            ("hpd",     C_HPD, LABEL_HPD),
            ("vanilla", C_VAN, LABEL_VAN),
        ]):
            agent_rows = [r for r in sub if r["agent"] == agent]
            means, stds = [], []
            for _, ep_lo, ep_hi, _ in phases:
                vals = [r["reward"] for r in agent_rows
                        if ep_lo <= r["episode"] < ep_hi]
                means.append(float(np.mean(vals)) if vals else 0.0)
                stds.append( float(np.std(vals))  if vals else 0.0)

            xpos = x + (j - 0.5) * bar_w
            ax.bar(xpos, means, width=bar_w, color=color, alpha=0.88,
                   edgecolor="white", linewidth=0.8, zorder=2,
                   yerr=stds, capsize=4,
                   error_kw={"elinewidth": 1.3, "capthick": 1.3,
                              "ecolor": "#444444"},
                   label=label)

            if agent == "hpd":
                hpd_means = means[:]
            else:
                van_means = means[:]
            all_stds.extend(stds)

        # Percentage difference annotation
        max_std = max(all_stds) if all_stds else 0
        for pi, (hm, vm) in enumerate(zip(hpd_means, van_means)):
            if vm > 0:
                pct = (hm - vm) / vm * 100
                top = max(hm, vm) + max_std + 6
                ax.text(pi, top, f"{'+' if pct >= 0 else ''}{pct:.0f}%",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color=C_HPD if pct >= 0 else C_SAB)

        for pi in [0.5, 1.5]:
            ax.axvline(pi, color="#CCCCCC", linewidth=0.9, zorder=1)

        ax.set_title(f"Per-Phase Mean Reward — {SAB_DISPLAY.get(sab, sab)}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Training Phase", fontsize=11)
        ax.set_ylabel("Mean Episode Reward", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([p[0] for p in phases], fontsize=10)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        ax.legend(fontsize=9.5, framealpha=0.9)

        plt.tight_layout()
        out = f"phase_reward_comparison_{sab}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Supervisor Activation Decay
# ══════════════════════════════════════════════════════════════════════════════

def plot_supervisor_activation(metrics: List[Dict]):
    present = _sab_types(metrics)
    if not present:
        print("WARNING: No data for supervisor plot. Skipping."); return

    for sab in present:
        fig, ax = plt.subplots(figsize=(10, 6))
        hpd_rows = [r for r in metrics
                    if r["sabotage_type"] == sab and r["agent"] == "hpd"]

        # Aggregate llm_calls per episode across seeds
        ep_vals: Dict[int, List[float]] = defaultdict(list)
        for r in hpd_rows:
            ep_vals[r["episode"]].append(r["llm_calls"])

        eps  = np.array(sorted(ep_vals))
        mean = np.array([np.mean(ep_vals[e]) for e in eps])
        std  = np.array([np.std(ep_vals[e])  for e in eps])

        add_phase_backgrounds(ax)
        ax.fill_between(eps, mean - std, mean + std,
                        color=C_HPD_L, alpha=0.35, zorder=2)
        ax.plot(eps, mean, color=C_HPD, linewidth=2.2,
                label=f"{LABEL_HPD} LLM Calls", zorder=3)
        add_phase_vlines(ax)

        trans = ax.get_xaxis_transform()
        ax.text((0 + SAB_EP) / 2,       0.95, "Dormant",
                ha="center", va="top", fontsize=8.5, color="#00695C",
                fontweight="bold", style="italic", transform=trans, alpha=0.8)
        ax.text((SAB_EP + EVAL_EP) / 2, 0.95, "Active",
                ha="center", va="top", fontsize=8.5, color=C_HPD,
                fontweight="bold", transform=trans, alpha=0.9)
        ax.text((EVAL_EP + TOTAL_EP)/2,  0.95, "Dormant",
                ha="center", va="top", fontsize=8.5, color="#1B5E20",
                fontweight="bold", style="italic", transform=trans, alpha=0.8)

        ax.set_title(f"Supervisor Activation Decay — {SAB_DISPLAY.get(sab, sab)}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("LLM Calls per Episode", fontsize=11)
        ax.set_xlim(0, TOTAL_EP)
        ax.set_ylim(-0.05, 1.10)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=9.5, loc="lower left", framealpha=0.9, edgecolor="white")

        plt.tight_layout()
        out = f"supervisor_activation_decay_{sab}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Recovery Rate Summary
# ══════════════════════════════════════════════════════════════════════════════

def plot_recovery_rate(summary: List[Dict]):
    present = _sab_types(summary)
    if not present:
        print("WARNING: No data for recovery rate plot. Skipping."); return

    x_labels = [SAB_DISPLAY.get(s, s) for s in present]
    x        = np.arange(len(present))
    bar_w    = 0.32
    n_seeds  = len(SEEDS)

    fig, ax = plt.subplots(figsize=(10, 5))

    for j, (agent, color, label) in enumerate([
        ("hpd",     C_HPD, LABEL_HPD),
        ("vanilla", C_VAN, LABEL_VAN),
    ]):
        counts = [
            sum(1 for r in summary
                if r["sabotage_type"] == sab
                and r["agent"] == agent
                and r["ttr_num"] is not None)
            for sab in present
        ]
        xpos = x + (j - 0.5) * bar_w
        bars = ax.bar(xpos, counts, width=bar_w, color=color, alpha=0.88,
                      edgecolor="white", linewidth=0.8, label=label)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{cnt}/{n_seeds}",
                    ha="center", va="bottom",
                    fontsize=13, fontweight="bold",
                    color=C_HPD if agent == "hpd" else C_VAN)

    ax.axhline(n_seeds, color="#888888", linestyle="--", linewidth=1.4,
               label=f"All seeds  (n = {n_seeds})")

    ax.set_title("Recovery Rate — HPD Framework vs Vanilla DQN\n"
                 "Seeds Successfully Recovered within Benchmark Window",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Seeds Successfully Recovered (out of 10)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylim(0, 11)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out = "recovery_rate_summary.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    metrics, summary = load_data()
    plot_ttr_comparison(summary)
    plot_phase_reward_comparison(metrics)
    plot_supervisor_activation(metrics)
    plot_recovery_rate(summary)
    print("\nDone. All plots saved to the current directory.")
