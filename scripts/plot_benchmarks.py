import argparse
import csv
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def _load_metrics(metrics_csv: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(metrics_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "agent": row["agent"],
                    "episode": int(row["episode"]),
                    "reward": float(row["reward"]),
                    "llm_calls": float(row["llm_calls"]),
                    "sabotage_type": row.get("sabotage_type", "inverted_controls"),
                }
            )
    return rows


def _aggregate_mean_std(rows: List[Dict], metric: str) -> Dict[str, Dict[str, np.ndarray]]:
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["agent"]][row["episode"]].append(row[metric])

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for agent, episode_map in grouped.items():
        episodes = sorted(episode_map.keys())
        means = np.array([np.mean(episode_map[ep]) for ep in episodes], dtype=float)
        stds = np.array([np.std(episode_map[ep]) for ep in episodes], dtype=float)
        out[agent] = {
            "episodes": np.array(episodes, dtype=int),
            "mean": means,
            "std": stds,
        }
    return out


def plot_from_csv(
    metrics_csv: str,
    output_path: str = "benchmark_hpd_vs_vanilla",
    sabotage_episode: int = 75,
    eval_episode: int = 125,
) -> None:
    rows = _load_metrics(metrics_csv)
    sabotage_values = sorted({r["sabotage_type"] for r in rows})

    color_map = {
        "hpd": "#1565C0",
        "vanilla": "#5C5C5C",
        "pure_llm": "#D97706",
    }
    label_map = {
        "hpd": "HPD Framework",
        "vanilla": "Vanilla DQN",
        "pure_llm": "Pure LLM",
    }
    sabotage_title_map = {
        "inverted_controls": "Inverted Controls",
        "sensor_inversion": "Sensor Inversion",
        "high_gravity": "High Gravity",
        "actuator_delay": "Actuator Delay",
        "sensor_noise": "Sensor Noise",
    }
    # Only agents that can produce LLM calls appear in the supervisor panel
    llm_agents = {"hpd", "pure_llm"}

    for sabotage_type in sabotage_values:
        scoped_rows = [r for r in rows if r["sabotage_type"] == sabotage_type]
        reward_data = _aggregate_mean_std(scoped_rows, metric="reward")
        llm_data = _aggregate_mean_std(scoped_rows, metric="llm_calls")

        sabotage_label = sabotage_title_map.get(sabotage_type, sabotage_type.replace("_", " ").title())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

        for agent, series in reward_data.items():
            episodes = series["episodes"]
            mean = series["mean"]
            std = series["std"]
            color = color_map.get(agent, None)
            label = label_map.get(agent, agent.upper())

            ax1.plot(episodes, mean, color=color, linewidth=1.8, label=label)
            ax1.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.18)

        ax1.axvline(sabotage_episode, color="#D32F2F", linestyle="--", linewidth=1.4, label="Sabotage")
        ax1.axvline(eval_episode, color="#2E7D32", linestyle=":", linewidth=1.6, label="Evaluation")
        ax1.set_title(f"Reward Comparison: {sabotage_label} (Mean ± Std)", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.grid(True, linestyle="--", alpha=0.55)
        ax1.legend(fontsize=10)

        for agent, series in llm_data.items():
            if agent not in llm_agents:
                continue
            episodes = series["episodes"]
            mean = series["mean"]
            std = series["std"]
            color = color_map.get(agent, None)
            label = f"{label_map.get(agent, agent.upper())} LLM Calls"

            ax2.plot(episodes, mean, color=color, linewidth=1.8, label=label)
            ax2.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.18)

        ax2.axvline(sabotage_episode, color="#D32F2F", linestyle="--", linewidth=1.4, label="Sabotage")
        ax2.axvline(eval_episode, color="#2E7D32", linestyle=":", linewidth=1.6, label="Evaluation")
        ax2.set_title(f"Supervisor Calls: {sabotage_label} (Mean ± Std)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("LLM Calls per Episode")
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.grid(True, linestyle="--", alpha=0.55)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        output_file = f"{output_path}_{sabotage_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved benchmark plot to: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark curves from CSV metrics.")
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="benchmark_outputs/ALL_SEEDS_combined_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_outputs/benchmark_hpd_vs_vanilla",
    )
    parser.add_argument("--sabotage-episode", type=int, default=75)
    parser.add_argument("--eval-episode", type=int, default=125)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_from_csv(
        metrics_csv=args.metrics_csv,
        output_path=args.output,
        sabotage_episode=args.sabotage_episode,
        eval_episode=args.eval_episode,
    )


if __name__ == "__main__":
    main()
