
# Project 2‑1 – ML‑Agents Runner (Starter Pack)

This starter pack standardizes how your group runs ML‑Agents experiments **and logs data**.

## Quick start

1) **Clone ML‑Agents (recommended fork/branch)** and create a Python 3.10.11/3.10.12 virtual env.
2) Install packages:

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
# Then install ML-Agents inside the same venv (either from pip or editable from the cloned repo)
pip install mlagents
```

3) Build a Unity environment (e.g., `3DBall`) and note the **Behavior Name** on the Agent.
4) Run an experiment (example):

```bash
python run_experiment.py   --algorithm ppo   --lr 3e-4   --batch-size 1024   --env /path/to/3DBall/Build/3DBall   --behavior-name 3DBall   --max-steps 500000   --seed 1   --run-tag jon1   --no-graphics
```

- A generated config goes to `experiments/_generated/`.
- ML‑Agents results go to `results/`.
- A **single row** of metadata is appended to `data/experiments.csv`.

## Who runs what (example mapping)

- **Jon** – switch algorithms (e.g., `ppo` vs `sac`)
- **Irene** – sweep **learning rate**
- **Mariam** – sweep **batch size**
- **Ronan** – learning rate (replication / other behaviors/seeds)
- **Lucie** – different **Unity environment** (e.g., `Walker`, `Crawler`, etc.)

Everyone uses the **same script** (`run_experiment.py`) so results are comparable.

## Tips

- Behavior name placeholder in `experiments/base_config.yaml` will be replaced the first time you pass `--behavior-name`.
- Use `--run-tag` to add a short personal tag to the `run_id`.
- Launch TensorBoard with `tensorboard --logdir results` to see training curves.

## Data schema (data/experiments.csv)

`run_id, timestamp, algorithm, learning_rate, batch_size, behavior_name, env_path, max_steps, seed, wall_time_sec, cpu_count, ram_gb, gpu_name, gpu_mem_gb, results_dir, git_commit, platform, user`

Keep `data/experiments.csv` under version control so the whole group can merge results.
