"""
Generates 4 independent, colourful, thesis-quality figures.

Colour language (consistent with architecture diagrams):
  Blue  (#1976D2)  — HPD Framework, active/positive
  Grey  (#546E7A)  — Vanilla DQN, neutral
  Red   (#E57373)  — NR bars, sabotage, Phase 2
  Cyan  (#E0F7FA)  — Phase 1 (Nominal) background
  Pink  (#FFEBEE)  — Phase 2 (Recovery) background
  Green (#E8F5E9)  — Phase 3 (Evaluation) background

Outputs (saved to thesis_plots/):
  plot_01_supervisor_activation.png
  plot_02_ttr_distribution_si.png
  plot_03_recovery_rate_si.png
  plot_04_phase_reward_comparison.png
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SI_METRICS = "results/benchmark_op_si/ALL_SEEDS_combined_metrics.csv"
SI_SUMMARY = "results/benchmark_op_si/ALL_SEEDS_combined_summary.csv"
IC_METRICS = "results/benchmark_op_ic/ALL_SEEDS_combined_metrics.csv"
OUT_DIR    = "thesis_plots"
SEEDS      = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
SAB_EP     = 75
EVAL_EP    = 125
TOTAL_EP   = 150

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_HPD     = "#1976D2"   # blue  — HPD / active
C_VAN     = "#546E7A"   # dark slate grey — Vanilla
C_NR      = "#E57373"   # red-pink — NR / sabotage
C_GREEN   = "#2E7D32"   # dark green — eval marker
C_RED_MK  = "#C62828"   # dark red — sabotage marker

# Phase backgrounds (axvspan)
C_P1_BG   = "#E0F7FA"   # light cyan   Phase 1
C_P2_BG   = "#FFEBEE"   # light pink   Phase 2
C_P3_BG   = "#E8F5E9"   # light green  Phase 3

# Phase bar fill colours (more saturated for bars)
C_P1_BAR  = "#80DEEA"   # cyan
C_P2_BAR  = "#EF9A9A"   # pink-red
C_P3_BAR  = "#A5D6A7"   # green

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.4,
})

# ---------------------------------------------------------------------------
# Loaders / helpers
# ---------------------------------------------------------------------------

def load_metrics(path: str) -> List[Dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "seed":      int(r["seed"]),
                "agent":     r["agent"],
                "episode":   int(r["episode"]),
                "reward":    float(r["reward"]),
                "llm_calls": float(r["llm_calls"]),
            })
    return rows


def load_summary(path: str) -> List[Dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ttr = r["ttr_episodes"]
            rows.append({
                "seed":        int(r["seed"]),
                "agent":       r["agent"],
                "ttr":         None if ttr == "not_recovered" else int(ttr),
                "mean_reward": float(r["mean_reward"]),
            })
    return rows


def agg_by_episode(rows: List[Dict], metric: str) -> Dict[str, Dict]:
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[r["agent"]][r["episode"]].append(r[metric])
    out = {}
    for agent, ep_map in grouped.items():
        eps = sorted(ep_map)
        out[agent] = {
            "episodes": np.array(eps),
            "mean":     np.array([np.mean(ep_map[e]) for e in eps]),
            "std":      np.array([np.std(ep_map[e])  for e in eps]),
        }
    return out


def phase_mean(rows, agent, ep_lo, ep_hi):
    vals = [r["reward"] for r in rows if r["agent"] == agent and ep_lo <= r["episode"] < ep_hi]
    return float(np.mean(vals)) if vals else 0.0


def phase_std(rows, agent, ep_lo, ep_hi):
    vals = [r["reward"] for r in rows if r["agent"] == agent and ep_lo <= r["episode"] < ep_hi]
    return float(np.std(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Plot 01 — Supervisor Activation Decay
# ---------------------------------------------------------------------------

def plot_supervisor_activation():
    rows = load_metrics(SI_METRICS)
    data = agg_by_episode(rows, "llm_calls")
    hpd  = data["hpd"]
    eps, mean, std = hpd["episodes"], hpd["mean"], hpd["std"]

    fig, ax = plt.subplots(figsize=(11, 5))

    # Phase background bands
    ax.axvspan(0,        SAB_EP,   color=C_P1_BG, zorder=0)
    ax.axvspan(SAB_EP,   EVAL_EP,  color=C_P2_BG, zorder=0)
    ax.axvspan(EVAL_EP,  TOTAL_EP, color=C_P3_BG, zorder=0)

    # Phase labels at top
    for mid, label, clr in [
        ((0 + SAB_EP) / 2,       "Phase 1\nNominal Operation",  "#00838F"),
        ((SAB_EP + EVAL_EP) / 2, "Phase 2\nRecovery",           "#C62828"),
        ((EVAL_EP + TOTAL_EP)/2, "Phase 3\nEvaluation",          "#2E7D32"),
    ]:
        ax.text(mid, 1.08, label, ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=clr, transform=ax.get_xaxis_transform())

    # LLM calls line + shading
    ax.fill_between(eps, mean - std, mean + std, color=C_HPD, alpha=0.18, zorder=2)
    ax.plot(eps, mean, color=C_HPD, linewidth=2.5, label="HPD Supervisor LLM Calls", zorder=3)

    # Boundary markers
    ax.axvline(SAB_EP,  color=C_RED_MK, linestyle="--", linewidth=1.8,
               label="Sabotage (ep 75)", zorder=4)
    ax.axvline(EVAL_EP, color=C_GREEN,  linestyle=":",  linewidth=1.8,
               label="Evaluation (ep 125)", zorder=4)

    # Annotations for flat zeros
    ax.annotate("Dormant\n(0 calls)", xy=(37, 0.01), fontsize=9.5,
                ha="center", color="#006064", style="italic")
    ax.annotate("Dormant\n(0 calls)", xy=(137, 0.01), fontsize=9.5,
                ha="center", color="#1B5E20", style="italic")
    ax.annotate("Active\n(≈1 call/ep)", xy=(100, 0.82), fontsize=9.5,
                ha="center", color=C_HPD, fontweight="bold")

    ax.set_title("Supervisor Activation Across Training Phases\n"
                 "HPD Framework — Sensor Inversion (Mean ± Std, 10 Seeds)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("LLM Calls per Episode", fontsize=11)
    ax.set_xlim(0, TOTAL_EP)
    ax.set_ylim(-0.05, 1.25)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=10, loc="upper right", framealpha=0.92)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "plot_01_supervisor_activation.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 02 — TTR Distribution, Sensor Inversion
# ---------------------------------------------------------------------------

NR_H = 75

def plot_ttr_distribution_si():
    summary = load_summary(SI_SUMMARY)
    hpd_map = {r["seed"]: r["ttr"] for r in summary if r["agent"] == "hpd"}
    van_map = {r["seed"]: r["ttr"] for r in summary if r["agent"] == "vanilla"}

    x     = np.arange(len(SEEDS))
    bar_w = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (agent_map, base_color, label) in enumerate([
        (hpd_map, C_HPD, "HPD Framework"),
        (van_map, C_VAN, "Vanilla DQN"),
    ]):
        heights, hatches, nr_flags, colors = [], [], [], []
        for s in SEEDS:
            v = agent_map.get(s)
            if v is None:
                heights.append(NR_H)
                hatches.append("////")
                nr_flags.append(True)
                colors.append(C_NR)
            else:
                heights.append(v)
                hatches.append("")
                nr_flags.append(False)
                colors.append(base_color)

        xpos = x + (i - 0.5) * bar_w
        for j, (h, hatch, is_nr, clr, xp) in enumerate(zip(heights, hatches, nr_flags, colors, xpos)):
            bar = ax.bar(xp, h, width=bar_w, color=clr, alpha=0.82 if not is_nr else 0.55,
                         edgecolor="white", linewidth=0.9, hatch=hatch,
                         label=label if j == 0 else "_nolegend_")
            lbl_y = h + 0.9
            if is_nr:
                ax.text(xp, lbl_y, "NR", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=C_NR)
            else:
                ax.text(xp, lbl_y, str(h), ha="center", va="bottom",
                        fontsize=9, color="#222222")

    # Mean TTR lines
    hpd_ttrs = [v for v in hpd_map.values() if v is not None]
    van_ttrs = [v for v in van_map.values() if v is not None]
    hpd_mean = np.mean(hpd_ttrs)
    van_mean = np.mean(van_ttrs)

    ax.axhline(hpd_mean, color=C_HPD, linestyle="--", linewidth=1.6, alpha=0.7,
               label=f"HPD Mean TTR = {hpd_mean:.1f}")
    ax.axhline(van_mean, color=C_VAN, linestyle="--", linewidth=1.6, alpha=0.7,
               label=f"Vanilla Mean TTR = {van_mean:.1f} (recovered only)")

    ax.set_title("Time-to-Recovery Distribution — Sensor Inversion\n"
                 "HPD Framework vs Vanilla DQN (10 Seeds, NR = Not Recovered)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Random Seed", fontsize=11)
    ax.set_ylabel("Episodes to Recovery  (lower = better)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_ylim(0, NR_H + 16)
    ax.legend(fontsize=10, framealpha=0.92)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "plot_02_ttr_distribution_si.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 03 — Recovery Rate, Sensor Inversion
# ---------------------------------------------------------------------------

def plot_recovery_rate_si():
    summary  = load_summary(SI_SUMMARY)
    hpd_rate = sum(1 for r in summary if r["agent"] == "hpd"     and r["ttr"] is not None) / len(SEEDS) * 100
    van_rate = sum(1 for r in summary if r["agent"] == "vanilla"  and r["ttr"] is not None) / len(SEEDS) * 100

    fig, ax = plt.subplots(figsize=(6.5, 6))

    agents = ["HPD Framework", "Vanilla DQN"]
    rates  = [hpd_rate, van_rate]
    colors = [C_HPD, C_VAN]

    bars = ax.bar(agents, rates, color=colors, alpha=0.88,
                  edgecolor="white", linewidth=1.0, width=0.45)

    for bar, val, agent in zip(bars, rates, agents):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2.0,
                f"{val:.0f}%", ha="center", va="bottom",
                fontsize=18, fontweight="bold",
                color=C_GREEN if agent == "HPD Framework" else "#333333")

    # Green tick annotation on HPD bar
    ax.annotate("✓ 9 / 10 seeds", xy=(0, hpd_rate),
                xytext=(0.28, hpd_rate - 12),
                fontsize=11, color=C_GREEN, fontweight="bold",
                ha="center")
    ax.annotate("✗ 5 / 10 seeds", xy=(1, van_rate),
                xytext=(1.28, van_rate - 12),
                fontsize=11, color=C_NR, fontweight="bold",
                ha="center")

    ax.axhline(100, color="#AAAAAA", linestyle=":", linewidth=1.2)
    ax.set_title("Recovery Rate — Sensor Inversion\n"
                 "HPD Framework vs Vanilla DQN  (10 Seeds)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Seeds Successfully Recovered (%)", fontsize=11)
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 25, 50, 75, 100])

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "plot_03_recovery_rate_si.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 04 — Phase Reward Comparison (Sensor Inversion)
# ---------------------------------------------------------------------------

def plot_phase_reward_comparison():
    rows = load_metrics(SI_METRICS)

    phases = [
        ("Phase 1\nNominal",   0,        SAB_EP,   C_P1_BAR, C_P1_BG),
        ("Phase 2\nRecovery",  SAB_EP,   EVAL_EP,  C_P2_BAR, C_P2_BG),
        ("Phase 3\nEvaluation",EVAL_EP,  TOTAL_EP, C_P3_BAR, C_P3_BG),
    ]

    x     = np.arange(len(phases))
    bar_w = 0.28
    fig, ax = plt.subplots(figsize=(10, 6))

    # Phase background bands
    for i, (_, ep_lo, ep_hi, _, bg) in enumerate(phases):
        ax.axvspan(i - 0.48, i + 0.48, color=bg, zorder=0, alpha=0.9)

    # Bars per phase
    for i, (phase_label, ep_lo, ep_hi, phase_color, _) in enumerate(phases):
        for j, (agent, agent_color, agent_label) in enumerate([
            ("hpd",     C_HPD, "HPD Framework"),
            ("vanilla", C_VAN, "Vanilla DQN"),
        ]):
            mean = phase_mean(rows, agent, ep_lo, ep_hi)
            std  = phase_std(rows,  agent, ep_lo, ep_hi)
            xpos = i + (j - 0.5) * bar_w

            ax.bar(xpos, mean, width=bar_w, color=agent_color, alpha=0.88,
                   edgecolor="white", linewidth=0.8, zorder=2,
                   label=agent_label if i == 0 else "_nolegend_")
            ax.errorbar(xpos, mean, yerr=std, fmt="none",
                        ecolor="#333333", elinewidth=1.4, capsize=4, zorder=3)
            ax.text(xpos, mean + std + 3, f"{mean:.0f}",
                    ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold", color="#111111", zorder=4)

    # Phase dividers
    ax.axvline(0.5,  color="#BBBBBB", linestyle="-", linewidth=1.0, zorder=1)
    ax.axvline(1.5,  color="#BBBBBB", linestyle="-", linewidth=1.0, zorder=1)

    ax.set_title("Per-Phase Mean Reward — Sensor Inversion\n"
                 "HPD Framework vs Vanilla DQN  (10 Seeds, ± Std Dev)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in phases], fontsize=12)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.92)

    # Phase colour legend patches
    phase_patches = [
        mpatches.Patch(color=C_P1_BAR, alpha=0.7, label="Phase 1 — Nominal"),
        mpatches.Patch(color=C_P2_BAR, alpha=0.7, label="Phase 2 — Recovery"),
        mpatches.Patch(color=C_P3_BAR, alpha=0.7, label="Phase 3 — Evaluation"),
    ]
    leg1 = ax.get_legend()
    ax.legend(handles=ax.get_legend_handles_labels()[0] + phase_patches,
              labels=ax.get_legend_handles_labels()[1] + [p.get_label() for p in phase_patches],
              fontsize=9.5, loc="upper left", framealpha=0.92, ncol=2)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "plot_04_phase_reward_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_supervisor_activation()
    plot_ttr_distribution_si()
    plot_recovery_rate_si()
    plot_phase_reward_comparison()
    print(f"\nAll 4 thesis plots saved to: {OUT_DIR}/")
