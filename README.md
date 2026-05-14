# Generative AI for Resilient Decision Support: A Hierarchical Policy Distillation Framework

This repository contains the implementation accompanying the MSc thesis *"Generative AI for Resilient Decision Support: A Hierarchical Policy Distillation Framework"* (MTU Cork, 2025).

The framework introduces a hierarchical **Supervisor–Worker** architecture that combines a lightweight DQN agent with a locally-hosted LLM to recover from out-of-distribution (OOD) perturbations without retraining from scratch.

---

## How It Works

| Component | Role |
|-----------|------|
| **Worker (DQN)** | Interacts with the environment; learns control policies via Q-learning |
| **Supervisor (Qwen2.5:7b via Ollama)** | Dormant during nominal operation; activated on OOD detection to provide expert actions |
| **Failure Detection** | TD-error proxy + state-drift threshold triggers the Supervisor |
| **Policy Distillation** | Supervisor guidance is distilled into the Worker via Behavioural Cloning loss blended with Q-learning |

Recovery is achieved through three mechanisms: **Amnesia Protocol** (replay buffer flush at perturbation onset), **Context-Routed Expert Vaults** (isolated memory per sabotage type), and **Gradient Blending** (λ=5.0 BC loss weight).

---

## Repository Structure

```
HPD/
├── main.py                    # Single-run experiment entry point
├── requirements.txt
│
├── src/
│   ├── environment.py         # SabotagedCartPole / SabotagedLunarLander wrappers
│   ├── worker.py              # DQNWorker: Q-network, distill_policy, gradient blending
│   └── supervisor.py          # LocalLLaMASupervisor: vector-to-text bridge, Ollama client
│
├── scripts/
│   ├── benchmark_runner.py    # Multi-seed HPD vs Vanilla benchmarking with CSV export
│   ├── plot_benchmarks.py     # Reward curve plots from CSV
│   ├── generate_final_plots.py       # Thesis Figures 5.1–5.3, 5.6
│   ├── generate_thesis_figures.py    # Combined reward + supervisor call figures
│   ├── generate_thesis_plots.py      # Coloured per-phase thesis plots
│   └── generate_ttr_chart.py         # Time-to-Recovery bar chart
│
├── results/
│   ├── benchmark_op_ic/       # Inverted Controls: 10-seed CSVs + combined figure
│   └── benchmark_op_si/       # Sensor Inversion: 10-seed CSVs + combined figure
│
└── figures/
    ├── recovery_rate_summary.png         # Figure 5.1
    ├── ttr_comparison_both_conditions.png # Figure 5.2
    ├── phase_reward_comparison.png        # Figure 5.3
    ├── ttr_comparison.png                 # Figure 5.4 (IC)
    └── supervisor_activation_decay.png    # Figure 5.6
```

---

## Prerequisites

Requires a locally hosted LLM via [Ollama](https://ollama.com/).

```bash
# Install Ollama, then pull the model
ollama pull qwen2.5:7b

# Keep the Ollama service running before executing any scripts
```

---

## Installation

```bash
git clone https://github.com/PumaTechhh/HPD.git
cd HPD

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## Experimental Protocol

Experiments run over **150 episodes** across three phases:

| Phase | Episodes | Description |
|-------|----------|-------------|
| **1 — Nominal** | 0–74 | Agent learns the standard task; Supervisor dormant |
| **2 — Recovery** | 75–124 | OOD perturbation injected; Supervisor activated on failure detection |
| **3 — Evaluation** | 125–149 | Exploration off; agent evaluated under perturbed conditions |

Two sabotage conditions are tested:

- `inverted_controls` — action polarity reversed
- `sensor_inversion`  — observation sign flipped

---

## Running

**Single run (quick experiment):**

```bash
python main.py
```

**Multi-seed benchmark (HPD vs Vanilla DQN):**

```bash
# Run from the project root
python scripts/benchmark_runner.py \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --agents vanilla hpd \
    --sabotage-types inverted_controls sensor_inversion \
    --num-episodes 150 \
    --output-dir benchmark_outputs
```

Outputs per seed: `benchmark_episode_metrics.csv`, `benchmark_run_summary.csv`  
Combined: `ALL_SEEDS_combined_metrics.csv`, `ALL_SEEDS_combined_summary.csv`

**Regenerate thesis figures from existing CSVs:**

```bash
python scripts/generate_final_plots.py
python scripts/generate_thesis_figures.py
python scripts/generate_ttr_chart.py \
    --summary-csvs results/benchmark_op_ic/ALL_SEEDS_combined_summary.csv \
                   results/benchmark_op_si/ALL_SEEDS_combined_summary.csv \
    --output figures/ttr_comparison
```

---

## Results

All benchmark data (10 seeds × 2 conditions) is in [results/](results/). Key metrics:

- **Time-to-Recovery (TTR)**: episodes from perturbation onset until reward exceeds threshold for 4 consecutive episodes
- **Recovery Rate**: fraction of seeds that achieved sustained recovery within the benchmark window
- **Mean Phase Reward**: average reward across Phase 3 evaluation episodes

---

## Notes

- All `scripts/` must be run from the **project root** (not from inside `scripts/`)
- The Supervisor is invoked at most once per episode during Phase 2; subsequent OOD steps use a fast heuristic fallback
- The framework is environment-agnostic; extend by subclassing the wrappers in `src/environment.py`
