# random_runs.py
import argparse
import subprocess
import sys
import random
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Run multiple experiments with random hyperparameters.")
    parser.add_argument("--num-runs", type=int, required=True, help="Number of random runs to execute")
    parser.add_argument("--range-percent", type=float, required=True, help="Percentage of the full range to sample from (e.g., 0.95 for center 95%)")
    parser.add_argument("--behavior-name", required=True, help="Behavior name in Unity")
    parser.add_argument("--algorithm", default="ppo", help="Trainer algorithm (ppo or sac)")
    parser.add_argument("--env-path", default="none", help="Path to built Unity env or 'none' for Editor mode")
    parser.add_argument("--no-graphics", action="store_true", help="Run without graphics")
    parser.add_argument("--seed-base", type=int, default=1, help="Base seed (will increment for each run)")
    args = parser.parse_args()

    # Define hyperparameter ranges: (min, max, default)
    # You can fill in your own min/max values here
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
    }

    fixed_params = {
        "max_steps": 50000,
        "summary_freq": 10000,
        "learning_rate_schedule": "linear",
        "normalize": False,
    }

    # Parameters to randomize
    params_to_randomize = list(param_ranges.keys())

    # Integer parameters (will be rounded)
    integer_params = {"batch_size", "buffer_size", "num_epoch", "hidden_units", "num_layers", "time_horizon"}

    # Log-scale parameters (better distribution for rates)
    log_scale_params = {"learning_rate", "beta"}

    print(f"[INFO] Starting {args.num_runs} random runs")
    print(f"[INFO] Sampling from center {args.range_percent*100}% of defined ranges")
    print(f"[INFO] Parameter ranges:")
    for param, (min_val, max_val, default_val) in param_ranges.items():
        print(f"  {param:20s}: [{min_val:12.6g}, {max_val:12.6g}]  (default: {default_val:12.6g})")
    print()

    for run_idx in range(args.num_runs):
        print(f"{'='*70}")
        print(f"[INFO] Random Run {run_idx+1}/{args.num_runs}")
        print(f"{'='*70}")

        # Generate random values for each hyperparameter
        random_params = {}
        for param in params_to_randomize:
            min_val, max_val, default_val = param_ranges[param]

            # Calculate the full range
            full_range = max_val - min_val

            # Calculate the center of the range
            center = (min_val + max_val) / 2.0

            # Calculate the reduced range based on percentage
            reduced_range = full_range * args.range_percent

            # Calculate new min and max (centered)
            sampling_min = center - (reduced_range / 2.0)
            sampling_max = center + (reduced_range / 2.0)

            # Ensure we don't go outside original bounds (edge case)
            sampling_min = max(sampling_min, min_val)
            sampling_max = min(sampling_max, max_val)

            # Generate random value in the sampling range
            if param in log_scale_params:
                # Log scale for learning rates and beta
                log_min = np.log10(sampling_min)
                log_max = np.log10(sampling_max)
                random_val = 10 ** random.uniform(log_min, log_max)
            elif param in integer_params:
                # Integer parameters
                random_val = random.randint(int(sampling_min), int(sampling_max))
            else:
                # Linear scale for other parameters
                random_val = random.uniform(sampling_min, sampling_max)

            random_params[param] = random_val

        # Build command
        cmd = [
            sys.executable,
            "run_experiment.py",
            f"--behavior-name={args.behavior_name}",
            f"--algorithm={args.algorithm}",
            f"--seed={args.seed_base + run_idx}",
        ]

        # Add all randomized hyperparameters
        cmd.append(f"--batch-size={random_params['batch_size']}")
        cmd.append(f"--buffer-size={random_params['buffer_size']}")
        cmd.append(f"--learning-rate={random_params['learning_rate']}")
        cmd.append(f"--beta={random_params['beta']}")
        cmd.append(f"--epsilon={random_params['epsilon']}")
        cmd.append(f"--lambd={random_params['lambd']}")
        cmd.append(f"--num-epoch={random_params['num_epoch']}")
        cmd.append(f"--hidden-units={random_params['hidden_units']}")
        cmd.append(f"--num-layers={random_params['num_layers']}")
        cmd.append(f"--gamma={random_params['gamma']}")
        cmd.append(f"--reward-strength={random_params['reward_strength']}")
        cmd.append(f"--time-horizon={random_params['time_horizon']}")

        # Add fixed parameters
        cmd.append(f"--max-steps={fixed_params['max_steps']}")
        cmd.append(f"--summary-freq={fixed_params['summary_freq']}")
        cmd.append(f"--learning-rate-schedule={fixed_params['learning_rate_schedule']}")
        cmd.append(f"--normalize={fixed_params['normalize']}")

        # Add environment path if specified
        if args.env_path.lower() != "none":
            cmd.append(f"--env={args.env_path}")

        # Add no-graphics flag if specified
        if args.no_graphics:
            cmd.append("--no-graphics")

        # Add run tag to identify this as a random run
        cmd.append(f"--run-tag=random{run_idx+1}")

        # Print the parameters being used
        print(f"[INFO] Random hyperparameters for this run:")
        for param in sorted(params_to_randomize):
            value = random_params[param]
            min_val, max_val, default_val = param_ranges[param]

            # Calculate sampling range for display
            full_range = max_val - min_val
            center = (min_val + max_val) / 2.0
            reduced_range = full_range * args.range_percent
            sampling_min = max(center - (reduced_range / 2.0), min_val)
            sampling_max = min(center + (reduced_range / 2.0), max_val)

            print(f"  {param:20s} = {value:12.6g}  "
                  f"(range: [{sampling_min:10.6g}, {sampling_max:10.6g}], "
                  f"full: [{min_val:10.6g}, {max_val:10.6g}])")
        print()

        # Run the experiment
        print(f"[INFO] Executing: {' '.join(cmd)}")
        print()

        try:
            subprocess.check_call(cmd)
            print(f"[SUCCESS] Run {run_idx+1}/{args.num_runs} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Run {run_idx+1}/{args.num_runs} failed with return code {e.returncode}")
            print(f"[WARNING] Continuing with next run...")
        except KeyboardInterrupt:
            print(f"\n[ABORT] User interrupted. Stopping after {run_idx} runs.")
            sys.exit(1)

        print()

    print(f"{'='*70}")
    print(f"[DONE] All {args.num_runs} random runs complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()