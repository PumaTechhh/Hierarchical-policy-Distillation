"""
Generates two thesis-quality figures from the standalone 10-seed benchmark runs.

Figure 1 — Combined reward curves (2x2):
  Top-left:  Reward curves, Inverted Controls
  Top-right: Reward curves, Sensor Inversion
  Bottom-left:  Supervisor calls, Inverted Controls
  Bottom-right: Supervisor calls, Sensor Inversion

Figure 2 — Summary statistics (1x3):
  Left:   TTR per seed, both sabotage types
  Middle: Recovery rate (bar)
  Right:  Mean episode reward (bar)
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IC_METRICS  = "results/benchmark_op_ic/ALL_SEEDS_combined_metrics.csv"
IC_SUMMARY  = "results/benchmark_op_ic/ALL_SEEDS_combined_summary.csv"
SI_METRICS  = "results/benchmark_op_si/ALL_SEEDS_combined_metrics.csv"
SI_SUMMARY  = "results/benchmark_op_si/ALL_SEEDS_combined_summary.csv"
OUTPUT_DIR  = "thesis_figures"
SABOTAGE_EP = 75
EVAL_EP     = 125
SEEDS       = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
C_HPD  = "#1565C0"
C_VAN  = "#5C5C5C"
C_SAB  = "#D32F2F"
C_EVAL = "#2E7D32"
LABELS = {"hpd": "HPD Framework", "vanilla": "Vanilla DQN"}
LLM_AGENTS = {"hpd", "pure_llm"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metrics(path: str) -> List[Dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "seed":         int(r["seed"]),
                "agent":        r["agent"],
                "episode":      int(r["episode"]),
                "reward":       float(r["reward"]),
                "llm_calls":    float(r["llm_calls"]),
            })
    return rows


def load_summary(path: str) -> List[Dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ttr_raw = r["ttr_episodes"]
            rows.append({
                "seed":        int(r["seed"]),
                "agent":       r["agent"],
                "ttr":         None if ttr_raw == "not_recovered" else int(ttr_raw),
                "mean_reward": float(r["mean_reward"]),
            })
    return rows


def aggregate(rows: List[Dict], metric: str) -> Dict[str, Dict]:
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


# ---------------------------------------------------------------------------
# Figure 1 — 2×2 reward + supervisor calls
# ---------------------------------------------------------------------------

def plot_reward_panel(ax, reward_data, sabotage_label):
    for agent, s in reward_data.items():
        c = C_HPD if agent == "hpd" else C_VAN
        ax.plot(s["episodes"], s["mean"], color=c, linewidth=1.8,
                label=LABELS.get(agent, agent))
        ax.fill_between(s["episodes"], s["mean"] - s["std"],
                        s["mean"] + s["std"], color=c, alpha=0.18)
    ax.axvline(SABOTAGE_EP, color=C_SAB,  linestyle="--", linewidth=1.4, label="Sabotage")
    ax.axvline(EVAL_EP,     color=C_EVAL, linestyle=":",  linewidth=1.6, label="Evaluation")
    ax.set_title(f"Reward — {sabotage_label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Total Reward", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)


def plot_calls_panel(ax, llm_data, sabotage_label):
    for agent, s in llm_data.items():
        if agent not in LLM_AGENTS:
            continue
        c = C_HPD if agent == "hpd" else C_VAN
        ax.plot(s["episodes"], s["mean"], color=c, linewidth=1.8,
                label=f"{LABELS.get(agent, agent)} LLM Calls")
        ax.fill_between(s["episodes"], s["mean"] - s["std"],
                        s["mean"] + s["std"], color=c, alpha=0.18)
    ax.axvline(SABOTAGE_EP, color=C_SAB,  linestyle="--", linewidth=1.4, label="Sabotage")
    ax.axvline(EVAL_EP,     color=C_EVAL, linestyle=":",  linewidth=1.6, label="Evaluation")
    ax.set_title(f"Supervisor Calls — {sabotage_label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("LLM Calls per Episode", fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)


def figure1_combined_curves():
    ic_rows = load_metrics(IC_METRICS)
    si_rows = load_metrics(SI_METRICS)

    ic_reward = aggregate(ic_rows, "reward")
    ic_calls  = aggregate(ic_rows, "llm_calls")
    si_reward = aggregate(si_rows, "reward")
    si_calls  = aggregate(si_rows, "llm_calls")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        "HPD Framework vs Vanilla DQN — 10-Seed Benchmark (Mean ± Std)",
        fontsize=14, fontweight="bold", y=1.01
    )

    plot_reward_panel(axes[0][0], ic_reward, "Inverted Controls")
    plot_reward_panel(axes[0][1], si_reward, "Sensor Inversion")
    plot_calls_panel(axes[1][0],  ic_calls,  "Inverted Controls")
    plot_calls_panel(axes[1][1],  si_calls,  "Sensor Inversion")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig1_reward_curves.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Summary statistics (TTR per seed + recovery rate + mean reward)
# ---------------------------------------------------------------------------

NR_HEIGHT = 75  # sentinel bar height for not-recovered


def figure2_summary():
    ic_sum = load_summary(IC_SUMMARY)
    si_sum = load_summary(SI_SUMMARY)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        "Recovery Summary — HPD Framework vs Vanilla DQN (10 Seeds)",
        fontsize=14, fontweight="bold"
    )

    # ── Panel A: TTR per seed (grouped bars, both sabotage types stacked left/right) ──
    _plot_ttr_bars(axes[0], ic_sum, si_sum)

    # ── Panel B: Recovery rate ──
    _plot_recovery_rate(axes[1], ic_sum, si_sum)

    # ── Panel C: Mean episode reward ──
    _plot_mean_reward(axes[2], ic_sum, si_sum)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig2_summary_stats.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _ttr_vals(summary, agent):
    return {r["seed"]: r["ttr"] for r in summary if r["agent"] == agent}


def _plot_ttr_bars(ax, ic_sum, si_sum):
    bar_w = 0.2
    x = np.arange(len(SEEDS))

    configs = [
        ("inverted_controls", "IC",  ic_sum, -1.5),
        ("sensor_inversion",  "SI",  si_sum, +0.5),
    ]
    agents = [("hpd", C_HPD, "HPD"), ("vanilla", C_VAN, "Vanilla")]

    handles = []
    for sab_key, sab_short, summary, grp_off in configs:
        for i, (agent, color, alabel) in enumerate(agents):
            ttr_map = _ttr_vals(summary, agent)
            heights, hatches, nr_flags = [], [], []
            for seed in SEEDS:
                v = ttr_map.get(seed)
                heights.append(NR_HEIGHT if v is None else v)
                hatches.append("////" if v is None else "")
                nr_flags.append(v is None)

            xpos = x + (grp_off + i) * bar_w
            bars = ax.bar(xpos, heights, width=bar_w, color=color, alpha=0.82,
                          edgecolor="white", linewidth=0.6,
                          label=f"{alabel} ({sab_short})")
            handles.append(bars[0])

            for j, (bar, hatch, is_nr) in enumerate(zip(bars, hatches, nr_flags)):
                bar.set_hatch(hatch)
                label_y = bar.get_height() + 0.5
                ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                        "NR" if is_nr else str(heights[j]),
                        ha="center", va="bottom", fontsize=6.5,
                        fontweight="bold" if is_nr else "normal",
                        color="#B71C1C" if is_nr else "#222222")

    ax.set_title("Time-to-Recovery per Seed", fontsize=12, fontweight="bold")
    ax.set_xlabel("Random Seed", fontsize=10)
    ax.set_ylabel("Episodes to Recovery (lower = better)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_ylim(0, NR_HEIGHT + 12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.legend(handles=handles, fontsize=8, ncol=2)
    ax.annotate("NR = Not Recovered within benchmark window",
                xy=(0.01, -0.18), xycoords="axes fraction",
                fontsize=7.5, color="#666666", va="top")


def _plot_recovery_rate(ax, ic_sum, si_sum):
    sab_labels = ["Inverted\nControls", "Sensor\nInversion"]
    summaries  = [ic_sum, si_sum]
    bar_w = 0.3
    x = np.arange(len(sab_labels))

    for i, (agent, color) in enumerate([("hpd", C_HPD), ("vanilla", C_VAN)]):
        rates = []
        for summary in summaries:
            recovered = sum(1 for r in summary if r["agent"] == agent and r["ttr"] is not None)
            rates.append(recovered / len(SEEDS) * 100)
        bars = ax.bar(x + (i - 0.5) * bar_w, rates, width=bar_w,
                      color=color, alpha=0.85, edgecolor="white",
                      label=LABELS[agent])
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Recovery Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("% Seeds Recovered", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(sab_labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="#999", linestyle=":", linewidth=1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.legend(fontsize=9)


def _plot_mean_reward(ax, ic_sum, si_sum):
    sab_labels = ["Inverted\nControls", "Sensor\nInversion"]
    summaries  = [ic_sum, si_sum]
    bar_w = 0.3
    x = np.arange(len(sab_labels))

    for i, (agent, color) in enumerate([("hpd", C_HPD), ("vanilla", C_VAN)]):
        means, stds = [], []
        for summary in summaries:
            rewards = [r["mean_reward"] for r in summary if r["agent"] == agent]
            means.append(np.mean(rewards))
            stds.append(np.std(rewards))
        bars = ax.bar(x + (i - 0.5) * bar_w, means, width=bar_w,
                      color=color, alpha=0.85, edgecolor="white",
                      yerr=stds, capsize=4, error_kw={"linewidth": 1.2},
                      label=LABELS[agent])
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) + 2,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Mean Episode Reward", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Reward (150 episodes)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(sab_labels, fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15 if ax.get_ylim()[1] > 0 else 250)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figure1_combined_curves()
    figure2_summary()
    print(f"\nAll thesis figures saved to: {OUTPUT_DIR}/")
