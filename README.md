
# Project 2‑1 – ML‑Agents Runner 

## Overview
Predicting Unity ML-Agents training performance from hyperparameters using supervised learning.
<!-- This repository supports our on predicting Unity ML-Agents training performance  from hyperparameters using supervised learning.  -->

## Description
This study investigates the extent to which an agent’s final performance, measured by mean reward, can be predicted using only training parameters available before training.
Using training logs collected from Unity ML-Agents benchmarks, we trained supervised machine learning models to map initial hyperparameter configurations to final agent performance. Both a linear regression model  and a non-linear Gradient Boosting Model were evaluated using 10-fold cross-validation, with performance assessed via R², Mean Absolure Error (MAE) and Root Mean Squared Error (RMSE). 


## Getting Started

**Prerequisites**
1) Install Unity Hub and a supported Unity Editor version
2) Create virtual environment
3) Install Python 3.10.11 

**Installation**
1) Clone the repository 
```bash
git clone https://github.com/jjbro9/Project2-1_Group4.git
```

2) Install packages
```bash
pip install -r requirements.txt
pip install mlagents
```

3) Build a Unity environment (e.g., `3DBall`) and note the **Behavior Name** on the Agent.
4) Test: Run an experiment

```bash
python run_experiment.py \
--algorithm ppo \
--lr 3e-4 \
--batch-size 1024 \
--env /path/to/3DBall/Build/3DBall \
--behavior-name 3DBall \
--max-steps 500000 \
--seed 1 \
--run-tag name \
--no-graphics
```

- A generated config goes to `experiments/_generated/`.
- ML‑Agents results go to `results/`.
- A **single row** of metadata is appended to `data/experiments.csv`.



## Usage 
**random_runs.py** \
Samples random hyperparameter configurations within a specified percentage range and runs multiple ML-Agents training sessions.
```bash
# choose own values
python random_runs.py \
--num-runs=num_runs \
--range-percent=0.95 \ # recommended value
--behavior-name=name \
--env-path "/path/to/Project2-1/ml-agents/Project/Build/EnvironmentBuild.app" \
--no-graphics # ensures running without animation
```

**grid_search.py** \
Performs a grid search over specified hyperparameter values.

```bash
# choose own values
python grid_search.py \
--batch_size 1 \ 
--buffer_size 1 \
--learning_rate 1\
--beta 1 \
--epsilon 1 \
--lambd 1 \
--num_epoch 1 \
--hidden_units 1 \
--num_layers 1 \
--gamma 1 \
--reward_strength 1 \
--time_horizon 1 \
--max_steps 1 \
--env "/path/to/Project2-1/ml-agents/Project/Build/EnvironmentBuild.app" \
--no-graphics # ensures running without animation
```


**sweep.py** \
Sweeps a single hyperparameter across a defined range while holding others constant.

```bash
# choose own values
python sweep.py \
--param param_name \
--start start_value \
--end end_value \
--steps step_count \
--repeats repeat_count \
--behavior-name behaviour_name \
--algorithm algorithm \
--batch-size batch_size \
--env-path "/path/to/Project2-1/ml-agents/Project/Build/EnvironmentBuild.app" \
--no-graphics # ensures running without animation
```

**optimizer.py**
```bash
```


## Data schema (data/experiments.csv)
Data will be exported to  `data/experiments.csv` with the following header: 

`run_id, timestamp, behavior_name, algorithm, batch_size, buffer_size, learning_rate, beta, epsilon, lambd, num_epoch, learning_rate_schedule, normalize, hidden_units, num_layers, gamma, reward_strength, max_steps, time_horizon, summary_freq, seed, mean_reward, cpu_count, ram_gb, gpu_name, gpu_mem_gb, platform, user, env_path`


