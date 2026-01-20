# random_runs.py
import argparse
import subprocess
import sys
import random
import numpy as np
import itertools


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments with random hyperparameters.")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="num of batch_size runs")
    parser.add_argument("--buffer_size", type=int, required=True,
                        help="num of buffer_size runs")
    parser.add_argument("--learning_rate", type=int, required=True,
                        help="num of batch_size runs")
    parser.add_argument("--beta", type=int, required=True,
                        help="num of beta runs")
    parser.add_argument("--epsilon", type=int, required=True,
                        help="num of epsilon runs")
    parser.add_argument("--lambd", type=int, required=True,
                        help="num of lambda runs")
    parser.add_argument("--num_epoch", type=int, required=True,
                        help="num of num_epoch runs")
    parser.add_argument("--gamma", type=int, required=True,
                        help="num of gamma runs")
    parser.add_argument("--reward_strength", type=int, required=True,
                        help="num of reward_strength runs")
    parser.add_argument("--time_horizon", type=int, required=True,
                        help="num of time_horizon runs")
    parser.add_argument("--max_steps", type=int, required=True,
                        help="num of max_steps runs")
    parser.add_argument("--hidden_units", type=int, required=True,
                        help="num of hidden_units runs")
    parser.add_argument("--num_layers", type=int, required=True,
                        help="num of num_layers runs")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="the number of reruns with difrent seeds")

    # Define hyperparameter ranges: (min, max, default)
    param_ranges = {
        "batch_size": (512, 4096, 1024),
        "buffer_size": (2048, 20480, 10240),
        "learning_rate": (1e-5, 1e-3, 3.0e-4),
        "beta": (1e-5, 1e-2, 5.0e-4),
        "epsilon": (0.01, 1.0, 0.2),
        "lambd": (0.8, 0.99, 0.95),
        "num_epoch": (1, 10, 3),
        "hidden_units": (64, 512, 128),
        "num_layers": (1, 5, 2),
        "gamma": (0.9, 0.999, 0.99),
        "reward_strength": (0.5, 2.0, 1.0),
        "time_horizon": (32, 256, 64),
        "max_steps": (50000, 1000000, 100000)

    }

    fixed_params = {
        "summary_freq": 10000,
        "learning_rate_schedule": "linear",
        "normalize": False,
    }
    args = parser.parse_args()
    param_grids = {}
    for param_name in param_ranges.keys():
        arg_name = f"{param_name}"
        num_values = getattr(args, arg_name)
        min_val, max_val, default_val = param_ranges[param_name]

        if num_values == 1:
            param_grids[param_name] = [default_val]
        else:
            # creates a grid centered around the default value and avoiding the edges
            t = np.linspace(-1, 1, num_values + 2)[1:-1]
            quad = 1 - t**2
            half_range = (max_val - min_val) / 2
            samples = default_val + (t * half_range * np.sqrt(quad))
            samples = np.clip(samples, min_val, max_val)
            if param_name in ['batch_size', 'buffer_size', 'num_epoch', 'hidden_units',
                              'num_layers', 'time_horizon', 'max_steps']:
                samples = np.round(samples).astype(int)
            param_grids[param_name] = samples.tolist()

    param_names = list(param_grids.keys())
    param_values = list(param_grids.values())

    # Grid search loop
    run_idx = 0

    for combination in itertools.product(*param_values):
        run_idx = run_idx+1
        params = dict(zip(param_names, combination))
        cmd = [
            sys.executable,
            "run_experiment.py",
            f"--behavior-name=3DBall",
            f"--algorithm=ppo",
            f"--seed={run_idx}",
        ]
        cmd.append(f"--batch-size={params['batch_size']}")
        cmd.append(f"--buffer-size={params['buffer_size']}")
        cmd.append(f"--learning-rate={params['learning_rate']}")
        cmd.append(f"--beta={params['beta']}")
        cmd.append(f"--epsilon={params['epsilon']}")
        cmd.append(f"--lambd={params['lambd']}")
        cmd.append(f"--num-epoch={params['num_epoch']}")
        cmd.append(f"--gamma={params['gamma']}")
        cmd.append(f"--reward-strength={params['reward_strength']}")
        cmd.append(f"--time-horizon={params['time_horizon']}")
        cmd.append(f"--max-steps={params['max_steps']}")
        cmd.append("--no-graphics")
        cmd.append(f"--run-tag=grid{run_idx}")
        cmd.append(f"--env=ml-agents/Project/Build/3DBallBuild.app")
        cmd.append(
            f"--learning-rate-schedule={fixed_params['learning_rate_schedule']}")
        cmd.append(f"--normalize={fixed_params['normalize']}")

        try:
            subprocess.check_call(cmd)
            print(
                f"[SUCCESS] Run {run_idx+1}/{args.num_runs} completed successfully")
        except subprocess.CalledProcessError as e:
            print(
                f"[ERROR] Run {run_idx+1}/{args.num_runs} failed with return code {e.returncode}")
            print(f"[WARNING] Continuing with next run...")
        except KeyboardInterrupt:
            print(
                f"\n[ABORT] User interrupted. Stopping after {run_idx} runs.")
            sys.exit(1)

        print(f"{'='*70}")
        print(f"[DONE] complete!")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
