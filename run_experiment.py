#!/usr/bin/env python3
import argparse
import os
import sys
import time
import json
import csv
import subprocess
import platform
import re
import shutil
from datetime import datetime
from pathlib import Path

import psutil
import yaml
# import torch


def detect_gpu():
    system = platform.system().lower()

    # NVDIDIA chips
    if system in ["windows", "linux"]:
        try:
            # Runs an external command and returns whatever it prints to stdout
            out = subprocess.check_output(
                # Program: nvidia-smi
                # Args:
                # --query-gpu=name,memory.total → ask for each GPU’s name and total memory.
                # --format=csv,noheader → return it as CSV with no header row.
                ["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"],
                # Merge stderr into stdout, so any error messages from nvidia-smi end up in the captured output
                stderr=subprocess.STDOUT,
                # You get a string back (not bytes), with newlines normalized for your platform/locale.
                universal_newlines=True,
                timeout=5
            ).strip()
            # Just checks whether the command actually printed anything.
            if out:
                line = out.splitlines()[0]
                name, mem = [part.strip() for part in line.split(",")]
                mem_gb = None
                mem.split()
                # splits the string "10019 MiB" into a list of tokens: ["10019", "MiB"]
                for token in mem.split():
                    # Try to turn each token into a number (float(token)).
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


# csv_path: the file path to your CSV log.
# fieldnames: list of column names (keys of the dicts you’ll write).
# retries: how many times to retry if the file is locked (default 8).
# delay: seconds to wait between retries (default 1.5).


def ensure_header(csv_path, fieldnames, retries=8, delay=1.5):
    import time
    for i in range(retries):
        try:
            # Open the file in append mode ("a"):
            # If the file exists → it appends new rows at the end.
            # If it doesn’t exist → it creates the file automatically.
            exists = os.path.exists(csv_path)
            f = open(csv_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # If the file didn’t exist before (i.e., you just created it):
            # Write a header row once at the top — the column names.
            # If it did exist, skip this so you don’t add duplicate headers.
            if not exists:
                writer.writeheader()
            return f, writer
        except PermissionError:
            # If we’ve already tried retries times and it’s still locked, re-raise the error.
            if i == retries - 1:
                raise
            print(f"[WARN] CSV locked; retrying in {delay}s...")
            # Otherwise, print a warning and wait delay seconds before trying again.
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(
        description="Run an ML-Agents experiment and log metadata.")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--lr", type=float, required=True,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int,
                        required=True, help="Batch size")
    parser.add_argument("--env", default=None,
                        help="Path to built Unity env (omit for Editor mode)")
    parser.add_argument("--behavior-name", required=True,
                        help="Behavior name as shown on the Agent in Unity")
    parser.add_argument("--max-steps", type=int,
                        default=None, help="Override max_steps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--base-config", default="experiments/base_config.yaml")
    parser.add_argument("--run-tag", default="",
                        help="Free text tag (e.g., jon1, ireneA)")
    parser.add_argument("--no-graphics", action="store_true")
    args = parser.parse_args()

    # Opens your YAML file (e.g., experiments/base_config.yaml) and parses it into a normal Python dictionary called cfg.
    # YAML → dict means you can now read/modify it like cfg["behaviors"]["3DBall"]["hyperparameters"]["learning_rate"].
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

    # Patch fields
    b = behaviors[args.behavior_name]
    b["trainer_type"] = args.algorithm
    hp = b.setdefault("hyperparameters", {})
    hp["learning_rate"] = float(args.lr)
    hp["batch_size"] = int(args.batch_size)
    if args.max_steps is not None:
        b["max_steps"] = int(args.max_steps)

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
    run_id = f"{timestamp}-{args.behavior_name}-{args.algorithm}-lr{args.lr}-bs{args.batch_size}{tag_part}"

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

    # args.env is the argument you pass via --env on the command line.
    # In Unity ML-Agents, there are two ways to connect your environment:
    # Editor mode: you press Play in Unity and ML-Agents connects to the running Editor.
    # Built player: you export a standalone executable of the environment and run it headlessly.
    # So:
    # If you did supply an environment path (args.env is not None),
    # and it’s not a placeholder like "editor", "none", or "dummy",
    # then the script adds the flag --env=<path> to the ML-Agents command.
    if args.env and args.env.lower() not in {"editor", "none", "dummy"}:
        cmd.append(f"--env={args.env}")

    if args.no_graphics:
        cmd.append("--no-graphics")

    # This joins the list cmd into a single readable string for logging.
    print("[INFO] Launching:", " ".join(cmd))
    start = time.time()
    mean_rewards = []

    try:
        # subprocess.Popen(cmd, ...) starts an external process without waiting for it to finish. The Pipe makes sure that a buffer is created in memory
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # reads every line in search of Mean Reward to pass that along to the results folder?
        for line in proc.stdout:
            print(line, end="")
            match = re.search(r"Mean Reward:\s*([0-9.+-e]+)", line)
            if match:
                try:
                    value = float(match.group(1).rstrip("."))
                    mean_rewards.append(value)
                except ValueError:
                    pass

        # proc.wait() blocks your script until that process exits.
        proc.wait()

        if mean_rewards:
            mean_reward = sum(mean_rewards) / len(mean_rewards)
        else:
            mean_reward = 0.0

    except FileNotFoundError:
        print("[ERROR] 'mlagents-learn' not found. Activate the ML-Agents virtualenv or install ML-Agents.", file=sys.stderr)
        sys.exit(127)
    end = time.time()
    wall_time_sec = round(end - start, 2)

    # Prepare log row
    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "algorithm": args.algorithm,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "behavior_name": args.behavior_name,
        "env_path": args.env,
        "max_steps": b.get("max_steps", None),
        "seed": args.seed,
        "wall_time_sec": wall_time_sec,
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
        "gpu_name": gpu_name,
        "gpu_mem_gb": gpu_mem_gb,
        "mean_reward": mean_reward,
        "results_dir": os.path.abspath(args.results_dir),
        "git_commit": git_commit,
        "platform": platform.platform(),
        "user": os.environ.get("USERNAME") or os.environ.get("USER"),
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


if __name__ == "__main__":
    main()
