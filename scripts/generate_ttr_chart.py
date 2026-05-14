import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_summary(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ttr_raw = row["ttr_episodes"]
            ttr: Optional[int] = None if ttr_raw == "not_recovered" else int(ttr_raw)
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "agent": row["agent"],
                    "sabotage_type": row["sabotage_type"],
                    "ttr": ttr,
                    "num_episodes": int(row["num_episodes"]),
                    "sabotage_episode": int(row["sabotage_episode"]),
                }
            )
    return rows


def generate_ttr_chart(
    summary_csvs: List[str],
    output_path: str = "ttr_comparison",
) -> None:
    all_rows: List[Dict] = []
    for csv_path in summary_csvs:
        all_rows.extend(load_summary(csv_path))

    sabotage_types = sorted({r["sabotage_type"] for r in all_rows})
    seeds = sorted({r["seed"] for r in all_rows})

    sabotage_label_map = {
        "inverted_controls": "Inverted Controls",
        "sensor_inversion": "Sensor Inversion",
        "high_gravity": "High Gravity",
        "actuator_delay": "Actuator Delay",
        "sensor_noise": "Sensor Noise",
    }
    color_map = {"hpd": "#1565C0", "vanilla": "#5C5C5C"}
    agents = ["hpd", "vanilla"]

    n_sab = len(sabotage_types)
    fig, axes = plt.subplots(1, n_sab, figsize=(6 * n_sab, 5), sharey=False)
    if n_sab == 1:
        axes = [axes]

    # Maximum TTR bar height sentinel for "not_recovered" bars
    NR_SENTINEL = 75  # num_episodes - sabotage_episode

    bar_width = 0.32
    seed_positions = np.arange(len(seeds))

    for ax, sabotage_type in zip(axes, sabotage_types):
        sab_rows = [r for r in all_rows if r["sabotage_type"] == sabotage_type]
        ttr_by_agent_seed: Dict[str, Dict[int, Optional[int]]] = defaultdict(dict)
        for r in sab_rows:
            ttr_by_agent_seed[r["agent"]][r["seed"]] = r["ttr"]

        offsets = {"hpd": -bar_width / 2, "vanilla": bar_width / 2}

        for agent in agents:
            color = color_map[agent]
            agent_ttr = ttr_by_agent_seed.get(agent, {})
            heights = []
            hatches = []
            nr_flags = []
            for seed in seeds:
                val = agent_ttr.get(seed, None)
                if val is None:
                    heights.append(NR_SENTINEL)
                    hatches.append("////")
                    nr_flags.append(True)
                else:
                    heights.append(val)
                    hatches.append("")
                    nr_flags.append(False)

            xpos = seed_positions + offsets[agent]
            bars = ax.bar(
                xpos,
                heights,
                width=bar_width,
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.8,
                label=f"HPD Framework" if agent == "hpd" else "Vanilla DQN",
            )

            # Apply hatch and value labels
            for i, (bar, hatch, is_nr) in enumerate(zip(bars, hatches, nr_flags)):
                bar.set_hatch(hatch)
                label_y = bar.get_height() + 0.8
                if is_nr:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        label_y,
                        "NR",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color="#B71C1C",
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        label_y,
                        str(heights[i]),
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        color="#222222",
                    )

        sab_label = sabotage_label_map.get(sabotage_type, sabotage_type.replace("_", " ").title())
        ax.set_title(f"TTR — {sab_label}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Random Seed", fontsize=11)
        ax.set_ylabel("Episodes to Recovery (lower = better)", fontsize=10)
        ax.set_xticks(seed_positions)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_ylim(0, NR_SENTINEL + 10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)

        # "NR = Not Recovered" footnote (below x-axis label)
        ax.annotate(
            "NR = Not Recovered within benchmark window",
            xy=(0.01, -0.18),
            xycoords="axes fraction",
            fontsize=8,
            color="#666666",
            va="top",
        )

    plt.tight_layout()
    output_file = f"{output_path}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved TTR chart to: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TTR bar chart from benchmark summaries.")
    parser.add_argument(
        "--summary-csvs",
        nargs="+",
        required=True,
        help="Paths to ALL_SEEDS_combined_summary.csv files (one per sabotage type).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ttr_comparison",
        help="Output path (without .png extension).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_ttr_chart(summary_csvs=args.summary_csvs, output_path=args.output)


if __name__ == "__main__":
    main()
