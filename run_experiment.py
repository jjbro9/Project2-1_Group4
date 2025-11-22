#!/usr/bin/env python3
import argparse
import os
import sys
import time
import json
import csv
import subprocess
import platform
import shutil
from datetime import datetime
from pathlib import Path
import re


import psutil
import yaml


def detect_gpu():
    system = platform.system().lower()

    # NVDIDIA chips
    if system in ["windows", "linux"]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"],
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                timeout=5
            ).strip()
            if out:
                line = out.splitlines()[0]
                name, mem = [part.strip() for part in line.split(",")]
                mem_gb = None
                for token in mem.split():
                    try:
                        mem_gb = float(token) / \
                            1024.0 if "MiB" in mem else float(token)
                    except:
                        pass
                return name, mem_gb
        except Exception:
            pass
        return None, None

    if system == "darwin":
        try:
            # Runs an external command and returns whatever it prints to stdout
            output = subprocess.check_output(
                # macOS version of obtaining GPU information
                ["system_profiler", "SPDisplaysDataType"], text=True
            )
            name = None
            mem_gb = None
            for line in output.splitlines():
                name_match = re.search(r"Chipset Model:\s*(.*)", line)
                vram_match = re.search(r"VRAM.*:\s*(.*)", line)

                if name_match:
                    name = name_match.group(1).strip()
                if vram_match:
                    mem_gb = vram_match.group(1).strip()
                    break

            # new mac devices dont have a separte GPU memory and instead "share" it with the CPU. So therefor we will take the general memory
            if not mem_gb and "Apple" in name:
                hw_output = subprocess.check_output(
                    ["system_profiler", "SPHardwareDataType"], text=True
                )
                mem_match = re.search(r"Memory:\s*(.*)", hw_output)
                if mem_match:
                    mem_gb = mem_match.group(1).strip()
            return name, mem_gb

        except Exception:
            pass
        return None, None


def ensure_header(csv_path, fieldnames, retries=8, delay=1.5):
    import time
    for i in range(retries):
        try:
            exists = os.path.exists(csv_path)
            f = open(csv_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            return f, writer
        except PermissionError:
            if i == retries - 1:
                raise
            print(f"[WARN] CSV locked; retrying in {delay}s...")
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(
        description="Run an ML-Agents experiment and log metadata.")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")

    # Hyperparameters
    parser.add_argument("--batch-size", type=int,
                        default=1024, help="Batch size")
    parser.add_argument("--buffer-size", type=int,
                        default=10240, help="Buffer size")
    parser.add_argument("--learning-rate", type=float,
                        default=3.0e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=5.0e-4,
                        help="Beta (entropy regularization)")
    parser.add_argument("--epsilon", type=float,
                        default=0.2, help="Epsilon (PPO clip)")
    parser.add_argument("--lambd", type=float,
                        default=0.95, help="Lambda (GAE)")
    parser.add_argument("--num-epoch", type=int,
                        default=3, help="Number of epochs")
    parser.add_argument("--learning-rate-schedule",
                        choices=["linear", "constant"], default="linear", help="LR schedule")

    # Network settings
    parser.add_argument("--normalize", type=bool,
                        default=False, help="Normalize observations")
    parser.add_argument("--hidden-units", type=int,
                        default=128, help="Hidden units per layer")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of hidden layers")

    # Reward signals
    parser.add_argument("--gamma", type=float,
                        default=0.99, help="Discount factor")
    parser.add_argument("--reward-strength", type=float,
                        default=1.0, help="Extrinsic reward strength")

    # Training settings
    parser.add_argument("--max-steps", type=int,
                        default=50000, help="Maximum training steps")
    parser.add_argument("--time-horizon", type=int,
                        default=64, help="Time horizon")
    parser.add_argument("--summary-freq", type=int,
                        default=10000, help="Summary frequency")

    # Other settings
    parser.add_argument("--env", default=None,
                        help="Path to built Unity env (omit for Editor mode)")
    parser.add_argument("--behavior-name", required=True,
                        help="Behavior name as shown on the Agent in Unity")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--base-config", default="experiments/base_config.yaml")
    parser.add_argument("--run-tag", default="",
                        help="Free text tag (e.g., jon1, ireneA)")
    parser.add_argument("--no-graphics", action="store_true")

    parser.add_argument("--set", nargs=2, action="append",
                        metavar=('KEY', 'VALUE'), help="Set an arbitrary hyperparameter")

    args = parser.parse_args()

    # Load base config and patch it
    with open(args.base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Replace placeholder behavior name if present
    behaviors = cfg.get("behaviors", {})
    if "__BEHAVIOR_NAME__" in behaviors and args.behavior_name not in behaviors:
        behaviors[args.behavior_name] = behaviors.pop("__BEHAVIOR_NAME__")

    if args.behavior_name not in behaviors:
        print(
            f"[ERROR] Behavior '{args.behavior_name}' not found in config. Available: {list(behaviors.keys())}", file=sys.stderr)
        sys.exit(2)

    # # Patch all fields
    b = behaviors[args.behavior_name]
    b["trainer_type"] = args.algorithm

    # Hyperparameters
    hp = b.setdefault("hyperparameters", {})
    hp["batch_size"] = args.batch_size
    hp["buffer_size"] = args.buffer_size
    hp["learning_rate"] = args.learning_rate
    hp["beta"] = args.beta
    hp["epsilon"] = args.epsilon
    hp["lambd"] = args.lambd
    hp["num_epoch"] = args.num_epoch
    hp["learning_rate_schedule"] = args.learning_rate_schedule

    # Setting the arguments
    if args.set:
        for key, value in args.set:
            try:
                value = float(value)
            except ValueError:
                pass  # leave as string if not a float
            hp[key] = value

    # Network settings
    ns = b.setdefault("network_settings", {})
    ns["normalize"] = args.normalize
    ns["hidden_units"] = args.hidden_units
    ns["num_layers"] = args.num_layers

    # Reward signals
    rs = b.setdefault("reward_signals", {})
    extrinsic = rs.setdefault("extrinsic", {})
    extrinsic["gamma"] = args.gamma
    extrinsic["strength"] = args.reward_strength

    # Training settings
    b["max_steps"] = args.max_steps
    b["time_horizon"] = args.time_horizon
    b["summary_freq"] = args.summary_freq

    # Write generated config
    gen_dir = Path("experiments/_generated")
    gen_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    gen_cfg_path = gen_dir / \
        f"{timestamp}_{args.behavior_name}_{args.algorithm}.yaml"
    with open(gen_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Compose run id
    tag_part = f"-{args.run_tag}" if args.run_tag else ""
    run_id = f"{timestamp}-{args.behavior_name}-{args.algorithm}-lr{args.learning_rate}-bs{args.batch_size}{tag_part}"

    # Detect hardware
    cpu_count = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    gpu_name, gpu_mem_gb = detect_gpu()

    # Git commit (if in repo)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
    except Exception:
        git_commit = None

    # Build command
    cmd = [
        "mlagents-learn",
        str(gen_cfg_path),
        f"--run-id={run_id}",
        "--train",
        f"--results-dir={args.results_dir}",
        f"--seed={args.seed}",
    ]

    if args.env and args.env.lower() not in {"editor", "none", "dummy"}:
        cmd.append(f"--env={args.env}")
        cmd.append("--no-graphics")  # Headless mode

    if args.no_graphics:
        cmd.append("--no-graphics")

    print("[INFO] Launching:", " ".join(cmd))
    start = time.time()
    try:
        # subprocess.Popen(cmd, ...) starts an external process without waiting for it to finish.
        # The Pipe makes sure that a buffer is created in memory
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        mean_rewards = []
        metrics = {
            "buffer_size": None,
            "beta": None,
            "epsilon": None,
            "lambd": None,
            "num_epoch": None,
            "time_horizon": None,
            "summary_freq": None,
            "num_layers": None,
        }

        # s*([0-9.+-e]+) Match one or more characters that are digits, a decimal point, a plus or minus sign, or the letter “e”.
        # also indents at the beginning of the hyperparameter
        patterns = {key: re.compile(
            rf"^\s*{key}:\s*([0-9.+-e]+)", re.IGNORECASE) for key in metrics}
        mean_pattern = re.compile(r"Mean Reward:\s*([0-9.+-e]+)")

        # reads every line in search of Mean Reward to pass that along to the results folder?
        for line in proc.stdout:
            print(line, end="")
            match_mean = mean_pattern.search(line)
            if match_mean:
                try:
                    value = float(match_mean.group(1).rstrip("."))
                    mean_rewards.append(value)
                except ValueError:
                    pass

            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    try:
                        metrics[key] = float(match.group(1))
                    except ValueError:
                        pass

        # proc.wait() blocks script until that process exits
        proc.wait()

        if mean_rewards:
            mean_reward = mean_rewards[-1]
        else:
            mean_reward = 0.0

    except FileNotFoundError:
        print("[ERROR] 'mlagents-learn' not found. Activate the ML-Agents virtualenv or install ML-Agents.", file=sys.stderr)
        sys.exit(127)
    end = time.time()
    wall_time_sec = round(end - start, 2)

    # Prepare log row with all hyperparameters
    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "behavior_name": args.behavior_name,
        "algorithm": args.algorithm,
        "batch_size": args.batch_size,
        "buffer_size": metrics["buffer_size"],
        "learning_rate": args.learning_rate,
        "beta": metrics["beta"],
        "epsilon": metrics["epsilon"],
        "lambd": metrics["lambd"],
        "num_epoch": metrics["num_epoch"],
        "learning_rate_schedule": args.learning_rate_schedule,
        "normalize": args.normalize,
        "hidden_units": args.hidden_units,
        "num_layers": metrics["num_layers"],
        "gamma": args.gamma,
        "reward_strength": args.reward_strength,
        "max_steps": b.get("max_steps", None),
        "time_horizon": metrics["time_horizon"],
        "summary_freq": metrics["summary_freq"],
        "seed": args.seed,
        "mean_reward": mean_reward,
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
        "gpu_name": gpu_name,
        "gpu_mem_gb": gpu_mem_gb,
        "platform": platform.platform(),
        "user": os.environ.get("USERNAME") or os.environ.get("USER"),
        "env_path": args.env
        # "wall_time_sec": wall_time_sec,
        # "results_dir": os.path.abspath(args.results_dir),
        # "git_commit": git_commit,

    }

    # Write to CSV
    fieldnames = list(row.keys())
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True, parents=True)
    csv_path = data_dir / "experiments.csv"
    f, writer = ensure_header(csv_path, fieldnames)
    with f:
        writer.writerow(row)

    print(f"[DONE] Logged run to {csv_path}")
    print(
        f"[HINT] You can visualize training with: tensorboard --logdir {args.results_dir}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs=2, action='append',
                        help="Set hyperparameter and value")
    args = parser.parse_args()

    # Convert list of [param, value] pairs to dict
    hyperparams = {param: float(value) for param, value in (args.set or [])}


if __name__ == "__main__":
    main()
