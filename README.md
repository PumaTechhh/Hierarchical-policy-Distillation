
# Generative AI for Resilient Decision Support: A Hierarchical Policy Distillation Framework

This repository contains the implementation accompanying the research work titled *“Generative AI for Resilient Decision Support: A Hierarchical Policy Distillation Framework.”*

The proposed framework introduces a hierarchical Supervisor–Worker architecture that integrates a reinforcement learning agent with a generative model to improve robustness under dynamic and previously unseen environmental conditions. The primary objective is to address performance degradation caused by out-of-distribution (OOD) shifts and to reduce the effects of catastrophic forgetting.

---

## Overview

The system consists of two main components:

* **Worker Agent** : A Deep Q-Network (DQN)-based reinforcement learning agent responsible for interacting with the environment and learning control policies.
* **Supervisor Module** : A generative model interface that provides high-level guidance when the agent exhibits signs of failure or instability.

A failure detection mechanism based on temporal-difference (TD) error is used to identify brittle behaviour. Upon detection, the Supervisor module is invoked to generate corrective guidance, enabling recovery without retraining from scratch.

---

## Repository Structure

<pre class="overflow-visible! px-0!" data-start="1585" data-end="1945"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>HPD-Framework/</span><br/><span>│── main.py                # Entry point for running experiments</span><br/><span>│── requirements.txt       # List of dependencies</span><br/><span>│</span><br/><span>└── src/</span><br/><span>    ├── environment.py     # Custom environment with perturbation logic</span><br/><span>    ├── worker.py          # Reinforcement learning agent implementation</span><br/><span>    └── supervisor.py      # Supervisor module and model interface</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

## Prerequisites

This framework requires access to a locally hosted generative model.

1. Install Ollama from: [https://ollama.com/](https://ollama.com/)
2. Download the required model:

`ollama run llama3.2`

Ensure that the Ollama service is running in the background before executing the code.

---

## Installation

Clone the repository:

`git clone https://github.com/YourUsername/HPD-Framework.git cd HPD-Framework`

Create and activate a virtual environment (recommended):

`python -m venv venv `

`source venv/bin/activate        # On Windows: venv\Scripts\activate`

Install dependencies:
`pip install -r requirements.txt`

---

## Running the Experiment

To execute the primary experiment:

`python main.py`

---

## Experimental Protocol

The evaluation is conducted over three phases:

**Phase 1 (Episodes 0–74): Baseline Learning**

The agent learns the standard control task in a stable environment. The Supervisor module remains inactive.

**Phase 2 (Episodes 75–124): Environmental Perturbation**

The environment is modified without prior notification (for example, inverted control dynamics). This results in a decline in agent performance. The failure detection mechanism activates the Supervisor module, which provides corrective guidance.

**Phase 3 (Episodes 125–150): Evaluation**

Exploration is disabled and learning stabilises. The agent’s performance is evaluated under the modified conditions.

---

## Output

Upon completion, the framework generates a result file:

`hpd_results.png`

The output visualises:

* Performance degradation following environmental change
* Recovery behaviour over time
* Supervisor activation patterns

---

## Methodological Contribution

The framework demonstrates an alternative to conventional retraining approaches by incorporating a generative reasoning layer. Instead of relying solely on accumulated experience, the system adapts by integrating externally generated guidance, improving recovery speed and maintaining previously learned behaviours.

---

## Authors

Atharva Katurde
Dr. Ruairi O'Reilly

---

## Notes

* The implementation is intended for research and experimental purposes.
* The framework can be extended to other environments and domains with minimal modification.
* Alternative generative models may be integrated by modifying the Supervisor module.
