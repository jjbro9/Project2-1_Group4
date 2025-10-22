#!/usr/bin/env python3
import argparse, os, sys, time, json, csv, subprocess, platform, shutil
from datetime import datetime
from pathlib import Path

import psutil
import yaml

def detect_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
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
                    mem_gb = float(token) / 1024.0 if "MiB" in mem else float(token)
                except:
                    pass
            return name, mem_gb
    except Exception:
        pass
    return None, None

"""def ensure_header(csv_path, fieldnames):
    exists = os.path.exists(csv_path)
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
    return f, writer"""

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
    parser = argparse.ArgumentParser(description="Run an ML-Agents experiment and log metadata.")
    parser.add_argument("--algorithm", choices=["ppo","sac"], default="ppo")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--env", default=None, help="Path to built Unity env (omit for Editor mode)")
    parser.add_argument("--behavior-name", required=True, help="Behavior name as shown on the Agent in Unity")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--base-config", default="experiments/base_config.yaml")
    parser.add_argument("--run-tag", default="", help="Free text tag (e.g., jon1, ireneA)")
    parser.add_argument("--no-graphics", action="store_true")
    args = parser.parse_args()

    # Load base config and patch it
    with open(args.base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Replace placeholder behavior name if present
    behaviors = cfg.get("behaviors", {})
    if "__BEHAVIOR_NAME__" in behaviors and args.behavior_name not in behaviors:
        behaviors[args.behavior_name] = behaviors.pop("__BEHAVIOR_NAME__")

    if args.behavior_name not in behaviors:
        print(f"[ERROR] Behavior '{args.behavior_name}' not found in config. Available: {list(behaviors.keys())}", file=sys.stderr)
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
    gen_cfg_path = gen_dir / f"{timestamp}_{args.behavior_name}_{args.algorithm}.yaml"
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
        git_commit = subprocess.check_output(["git","rev-parse","HEAD"], universal_newlines=True).strip()
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

    # Only pass --env when you're using a built player (not Editor mode)
    if args.env and args.env.lower() not in {"editor", "none", "dummy"}:
        cmd.append(f"--env={args.env}")

    if args.no_graphics:
        cmd.append("--no-graphics")

    print("[INFO] Launching:", " ".join(cmd))
    start = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        ret = proc.wait()
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
    print(f"[HINT] You can visualize training with: tensorboard --logdir {args.results_dir}")

if __name__ == "__main__":
    main()
