import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.environment import SabotagedCartPole, SabotagedLunarLander
from src.supervisor import LocalLLaMASupervisor
from src.worker import DQNWorker


@dataclass
class RunConfig:
    num_episodes: int = 150
    sabotage_episode: int = 75
    eval_episode: int = 125
    sabotage_types: List[str] = None
    model_name: str = "qwen2.5:7b"
    env_name: str = "cartpole"
    recovery_threshold: float = 150.0
    sustain_episodes: int = 4

    def __post_init__(self) -> None:
        if self.sabotage_types is None:
            self.sabotage_types = ["inverted_controls"]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_ttr(
    rewards: List[float],
    sabotage_episode: int,
    recovery_threshold: float,
    sustain_episodes: int,
) -> Optional[int]:
    """
    Time-to-Recovery: episodes after sabotage until reward crosses threshold
    for `sustain_episodes` consecutive episodes.
    """
    max_start = len(rewards) - sustain_episodes + 1
    for idx in range(sabotage_episode, max_start):
        window = rewards[idx : idx + sustain_episodes]
        if all(val >= recovery_threshold for val in window):
            return idx - sabotage_episode
    return None


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_agent(seed: int, agent_type: str, cfg: RunConfig) -> Dict[str, object]:
    set_global_seed(seed)

    all_episode_rows: List[Dict] = []
    all_summary_rows: List[Dict] = []

    for sabotage_type in cfg.sabotage_types:
        if cfg.env_name == "lunarlander":
            env = SabotagedLunarLander(sabotage_type=sabotage_type, render_mode=None)
        else:
            env = SabotagedCartPole(render_mode=None)
        env.reset(seed=seed)
        env.action_space.seed(seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        worker = DQNWorker(state_dim, action_dim)
        supervisor = (
            LocalLLaMASupervisor(model_name=cfg.model_name)
            if agent_type in ["hpd", "pure_llm"]
            else None
        )

        active_context = cfg.sabotage_types[0] if cfg.sabotage_types else "inverted_controls"
        stability_threshold = 1.0
        episode_rows: List[Dict] = []
        episode_rewards: List[float] = []

        print(
            f"\n{'=' * 76}\n"
            f"Running {agent_type.upper()} | seed={seed} | sabotage={sabotage_type}\n"
            f"{'=' * 76}"
        )

        for episode in range(cfg.num_episodes):
            state, _ = env.reset()
            done = truncated = False
            total_reward = 0
            supervisor_calls_this_episode = 0
            td_errors_this_episode: List[float] = []

            if episode == cfg.sabotage_episode:
                env.trigger_sabotage(sabotage_type=active_context)
                worker.memory.clear()
                worker.td_error_history.clear()
                if agent_type == "hpd":
                    worker.expert_memory.clear()

                worker.epsilon = 1.0

            if episode == cfg.eval_episode:
                if agent_type == "hpd":
                    worker.vault_warm_up(context=active_context, steps=30)
                worker.epsilon = 0.0
                worker.epsilon_min = 0.0
                stability_threshold = 3.0

            lock_counter = 0
            last_expert_action = -1
            episode_llm_called = False

            while not (done or truncated):

                if lock_counter > 0 and last_expert_action >= 0:
                    action = last_expert_action
                    lock_counter -= 1
                else:
                    action = worker.select_action(state)

                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                selected_action = action

                td_error = worker.calculate_td_error(state, selected_action, reward, next_state, done)
                td_errors_this_episode.append(td_error)

                ood_state_drift = len(next_state) == 4 and abs(next_state[2]) > 0.10

                if (
                    (td_error > stability_threshold or ood_state_drift)
                    and info.get("sabotage_active")
                    and agent_type == "hpd"
                    and supervisor is not None
                    and episode < cfg.eval_episode
                ):
                    if not episode_llm_called:
                        # LLM called once per episode — provides BC training label
                        expert_action = supervisor.get_expert_action(next_state, active_context)
                        if getattr(supervisor, "last_call_used_llm", False):
                            supervisor_calls_this_episode += 1
                        episode_llm_called = True
                    else:
                        # Heuristic re-evaluated every subsequent OOD step — fast, correct
                        expert_action = supervisor._heuristic_fallback(next_state, active_context)

                    for _ in range(4):
                        worker.distill_policy(next_state, expert_action, context=active_context)

                    last_expert_action = expert_action
                    lock_counter = 1

                worker.memory.push(state, selected_action, reward, next_state, done)

                if episode < cfg.eval_episode:
                    worker.train_step(active_context=active_context)

                state = next_state

            if episode % 10 == 0:
                worker.update_target_network()

            mean_td_error = (
                float(np.mean(td_errors_this_episode)) if td_errors_this_episode else 0.0
            )
            episode_rewards.append(total_reward)

            episode_rows.append(
                {
                    "seed": seed,
                    "agent": agent_type,
                    "episode": episode,
                    "reward": float(total_reward),
                    "epsilon": float(worker.epsilon),
                    "llm_calls": int(supervisor_calls_this_episode),
                    "mean_td_error": mean_td_error,
                    "sabotage_active": int(episode >= cfg.sabotage_episode),
                    "sabotage_type": sabotage_type,
                    "evaluation_mode": int(episode >= cfg.eval_episode),
                }
            )

            if episode % 10 == 0 or supervisor_calls_this_episode > 0:
                print(
                    f"Ep {episode:3d} | Reward: {total_reward:5.1f} | "
                    f"Eps: {worker.epsilon:.2f} | Calls: {supervisor_calls_this_episode}"
                )

        env.close()

        ttr = compute_ttr(
            episode_rewards,
            cfg.sabotage_episode,
            cfg.recovery_threshold,
            cfg.sustain_episodes,
        )

        summary_row = {
            "seed": seed,
            "agent": agent_type,
            "env": cfg.env_name,
            "num_episodes": cfg.num_episodes,
            "sabotage_episode": cfg.sabotage_episode,
            "eval_episode": cfg.eval_episode,
            "sabotage_type": sabotage_type,
            "total_reward_sum": float(np.sum(episode_rewards)),
            "mean_reward": float(np.mean(episode_rewards)),
            "mean_llm_calls": float(np.mean([r["llm_calls"] for r in episode_rows])),
            "ttr_episodes": ttr if ttr is not None else "not_recovered",
            "recovery_threshold": cfg.recovery_threshold,
            "sustain_episodes": cfg.sustain_episodes,
        }

        all_episode_rows.extend(episode_rows)
        all_summary_rows.append(summary_row)

    return {"episodes": all_episode_rows, "summary": all_summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HPD vs Vanilla paired-seed benchmarks and export CSV logs."
    )
    parser.add_argument("--num-episodes", type=int, default=150)
    parser.add_argument("--sabotage-episode", type=int, default=75)
    parser.add_argument("--eval-episode", type=int, default=125)
    parser.add_argument(
        "--sabotage-types",
        nargs="+",
        default=["inverted_controls"],
        choices=["inverted_controls", "high_gravity", "actuator_delay", "sensor_noise", "sensor_inversion"],
    )
    parser.add_argument("--model-name", type=str, default="qwen2.5:7b")
    parser.add_argument(
        "--env",
        type=str,
        default="cartpole",
        choices=["cartpole", "lunarlander"],
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["vanilla", "hpd"],
        choices=["vanilla", "hpd", "pure_llm"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument("--output-dir", type=str, default="benchmark_outputs")
    parser.add_argument("--recovery-threshold", type=float, default=150.0)
    parser.add_argument("--sustain-episodes", type=int, default=4)
    parser.add_argument(
        "--plot-after-run",
        action="store_true",
        help="Generate thesis plot immediately after CSV export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = RunConfig(
        num_episodes=args.num_episodes,
        sabotage_episode=args.sabotage_episode,
        eval_episode=args.eval_episode,
        sabotage_types=args.sabotage_types,
        model_name=args.model_name,
        env_name=args.env,
        recovery_threshold=args.recovery_threshold,
        sustain_episodes=args.sustain_episodes,
    )

    all_episode_rows: List[Dict] = []
    all_summary_rows: List[Dict] = []

    for seed in args.seeds:
        seed_episode_rows: List[Dict] = []
        seed_summary_rows: List[Dict] = []

        for agent in args.agents:
            result = run_agent(seed=seed, agent_type=agent, cfg=cfg)
            seed_episode_rows.extend(result["episodes"])
            seed_summary_rows.extend(result["summary"])
            all_episode_rows.extend(result["episodes"])
            all_summary_rows.extend(result["summary"])

        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        write_csv(os.path.join(seed_dir, "benchmark_episode_metrics.csv"), seed_episode_rows)
        write_csv(os.path.join(seed_dir, "benchmark_run_summary.csv"), seed_summary_rows)

        print(f"\n>>> Seed {seed} results saved in: {seed_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_csv = os.path.join(args.output_dir, "ALL_SEEDS_combined_metrics.csv")
    summary_csv = os.path.join(args.output_dir, "ALL_SEEDS_combined_summary.csv")

    write_csv(metrics_csv, all_episode_rows)
    write_csv(summary_csv, all_summary_rows)

    print("\nCSV export complete:")
    print(f"- Episode metrics: {metrics_csv}")
    print(f"- Run summary:    {summary_csv}")

    if args.plot_after_run:
        try:
            from plot_benchmarks import plot_from_csv

            plot_from_csv(
                metrics_csv=metrics_csv,
                output_path=os.path.join(args.output_dir, "benchmark_hpd_vs_vanilla"),
                sabotage_episode=cfg.sabotage_episode,
                eval_episode=cfg.eval_episode,
            )
        except Exception as ex:
            print(f"Plot generation skipped due to error: {ex}")


if __name__ == "__main__":
    main()
