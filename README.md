# Robot Learning — Exercise 1

**Politecnico di Torino · A.Y. 2026–2027**

This repository explores **CartPole-v1** control using a **Linear Quadratic Regulator (LQR)** and a **policy-gradient reinforcement learning agent**. It includes experiments on controller weights, learning rates, repeatability, and reward design.

## Repository contents

| File / folder | Purpose |
|---|---|
| `cartpole_lqr.py` | LQR control and state/force comparison experiments. |
| `cartpole_rl.py` | RL training, evaluation, and custom reward functions. |
| `agent.py` | PyTorch policy network and learning algorithm. |
| `multiple_cartpoles_rl.py` | Repeated training runs to measure variability. |
| `utils.py` | Shared functions for rewards, space dimensions, and seeds. |
| `RL_Ex1_Colab_Workspace.ipynb` | Google Colab workspace for running experiments. |
| `report_lab1_2026.ipynb` | Exercise report with analysis and plots. |
| `experiments/reward_*` | Reward-design trials (center, target point, sweep), models, and plots. |
| `experiments/task_*` | Results and artifacts for individual assignment tasks. |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Files excluded from Git tracking. |

## Setup

```bash
pip install -r requirements.txt
```
