# sweep.py
import argparse
import subprocess
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Sweep a single hyperparameter over multiple runs.")
    parser.add_argument("--param", required=True,
                        help="Hyperparameter to sweep (e.g., beta, epsilon, learning_rate, batch_size)")
    parser.add_argument("--start", type=float, required=True,
                        help="Start value of the hyperparameter")
    parser.add_argument("--end", type=float, required=True,
                        help="End value of the hyperparameter")
    parser.add_argument("--steps", type=int, required=True,
                        help="Number of discrete values between start and end")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeats per value")
    parser.add_argument("--behavior-name", default="3DBall",
                        help="Behavior name in Unity")
    parser.add_argument("--algorithm", default="ppo",
                        help="Trainer algorithm (ppo or sac)")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Batch size (kept fixed during sweep)")
    parser.add_argument("--env-path", default="none",
                        help="Path to built Unity env or 'none' for Editor mode")
    args = parser.parse_args()

    # generate the sweep values
    values = np.linspace(args.start, args.end, args.steps)

    for v in values:
        for repeat_idx in range(args.repeats):
            print(
                f"[INFO] Running {args.param}={v:.6g}, repeat {repeat_idx+1}/{args.repeats}")

            cmd = [
                sys.executable,
                "run_experiment.py",
                f"--behavior-name={args.behavior_name}",
                f"--algorithm={args.algorithm}",
                f"--batch-size={args.batch_size}",
                f"--lr={args.lr if hasattr(args,'lr') else 3e-4}",
            ]

            # include environment only if not 'none'
            if args.env_path.lower() != "none":
                cmd.extend(["--set", args.param, str(v)])

            # insert the hyperparameter being swept
            cmd.extend(["--set", args.param, str(v)])

            # run the experiment
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Run failed with return code {e.returncode}")
                continue

    print("[DONE] Sweep complete!")


if __name__ == "__main__":
    main()
