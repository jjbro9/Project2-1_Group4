# random_runs.py
import argparse
import subprocess
import sys
import random
import numpy as np
from itertools import product


def generate_log_scale_grid(min_val, max_val, num_points):
    """Generate grid points on a log scale."""
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)
    log_points = np.linspace(log_min, log_max, num_points)
    return [10 ** x for x in log_points]


def generate_linear_grid(min_val, max_val, num_points, is_integer=False):
    """Generate grid points on a linear scale."""
    points = np.linspace(min_val, max_val, num_points)
    if is_integer:
        # Round to nearest integer and remove duplicates
        points = [int(round(p)) for p in points]
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in points if not (x in seen or seen.add(x))]
    return points.tolist()


def generate_grid_combinations(param_grids):
    """Generate all combinations from parameter grids using itertools.product."""
    param_names = list(param_grids.keys())
    param_values = [param_grids[name] for name in param_names]
    
    combinations = []
    for combo in product(*param_values):
        combination_dict = {name: val for name, val in zip(param_names, combo)}
        combinations.append(combination_dict)
    
    return combinations


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments with grid search hyperparameter optimization.")
    
    # Grid search arguments
    parser.add_argument("--grid-size", type=int, default=3,
                        help="Number of grid points per parameter")
    parser.add_argument("--max-combinations", type=int, default=None,
                        help="Maximum number of combinations to test (None = all combinations)")
    parser.add_argument("--randomize-grid-order", action="store_true",
                        help="Randomize the order of grid combinations (useful for early stopping)")
    
    # Common arguments
    parser.add_argument("--behavior-name", required=True,
                        help="Behavior name in Unity")
    parser.add_argument("--algorithm", default="ppo",
                        help="Trainer algorithm (ppo or sac)")
    parser.add_argument("--env-path", default="none",
                        help="Path to built Unity env or 'none' for Editor mode")
    parser.add_argument("--no-graphics", action="store_true",
                        help="Run without graphics")
    parser.add_argument("--seed-base", type=int, default=1,
                        help="Base seed (will increment for each run)")
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

    # Parameters to search over
    params_to_search = list(param_ranges.keys())

    # Integer parameters (will be rounded)
    integer_params = {"batch_size", "buffer_size", "num_epoch",
                      "hidden_units", "num_layers", "time_horizon"}

    # Log-scale parameters (better distribution for rates)
    log_scale_params = {"learning_rate", "beta"}
    
    # Generate grid points for each parameter
    param_grids = {}
    for param in params_to_search:
        min_val, max_val, default_val = param_ranges[param]
        
        if param in log_scale_params:
            # Log scale for learning rates and beta
            grid_points = generate_log_scale_grid(min_val, max_val, args.grid_size)
        elif param in integer_params:
            # Integer parameters
            grid_points = generate_linear_grid(min_val, max_val, args.grid_size, is_integer=True)
        else:
            # Linear scale for other parameters
            grid_points = generate_linear_grid(min_val, max_val, args.grid_size, is_integer=False)
        
        param_grids[param] = grid_points
    
    # Generate all combinations
    all_combinations = generate_grid_combinations(param_grids)
    
    # Limit combinations if specified
    if args.max_combinations is not None and len(all_combinations) > args.max_combinations:
        if args.randomize_grid_order:
            random.shuffle(all_combinations)
        all_combinations = all_combinations[:args.max_combinations]
    elif args.randomize_grid_order:
        random.shuffle(all_combinations)
    
    num_runs = len(all_combinations)
    
    # Print configuration
    print(f"[INFO] Grid Search Mode: {num_runs} combinations to test")
    print(f"[INFO] Grid size per parameter: {args.grid_size}")
    total_combinations = np.prod([len(param_grids[p]) for p in params_to_search])
    if total_combinations != num_runs:
        print(f"[INFO] Total possible combinations: {total_combinations} (limited to {num_runs})")
    print()
    
    print(f"[INFO] Parameter ranges:")
    for param, (min_val, max_val, default_val) in param_ranges.items():
        print(f"  {param:20s}: [{min_val:12.6g}, {max_val:12.6g}]  (default: {default_val:12.6g})")
    print()

    for run_idx in range(num_runs):
        print(f"{'='*70}")
        print(f"[INFO] Grid Search Run {run_idx+1}/{num_runs}")
        print(f"{'='*70}")

        # Use pre-generated combination
        run_params = all_combinations[run_idx]

        # Build command
        cmd = [
            sys.executable,
            "run_experiment.py",
            f"--behavior-name={args.behavior_name}",
            f"--algorithm={args.algorithm}",
            f"--seed={args.seed_base + run_idx}",
        ]

        # Add all hyperparameters
        cmd.append(f"--batch-size={run_params['batch_size']}")
        cmd.append(f"--buffer-size={run_params['buffer_size']}")
        cmd.append(f"--learning-rate={run_params['learning_rate']}")
        cmd.append(f"--beta={run_params['beta']}")
        cmd.append(f"--epsilon={run_params['epsilon']}")
        cmd.append(f"--lambd={run_params['lambd']}")
        cmd.append(f"--num-epoch={run_params['num_epoch']}")
        cmd.append(f"--hidden-units={run_params['hidden_units']}")
        cmd.append(f"--num-layers={run_params['num_layers']}")
        cmd.append(f"--gamma={run_params['gamma']}")
        cmd.append(f"--reward-strength={run_params['reward_strength']}")
        cmd.append(f"--time-horizon={run_params['time_horizon']}")

        # Add fixed parameters
        cmd.append(f"--max-steps={fixed_params['max_steps']}")
        cmd.append(f"--summary-freq={fixed_params['summary_freq']}")
        cmd.append(
            f"--learning-rate-schedule={fixed_params['learning_rate_schedule']}")
        cmd.append(f"--normalize={fixed_params['normalize']}")

        # Add environment path if specified
        if args.env_path.lower() != "none":
            cmd.append(f"--env={args.env_path}")

        # Add no-graphics flag if specified
        if args.no_graphics:
            cmd.append("--no-graphics")

        # Add run tag to identify this run
        cmd.append(f"--run-tag=grid{run_idx+1}")

        # Print the parameters being used
        print(f"[INFO] Grid search hyperparameters for this run:")
        for param in sorted(params_to_search):
            value = run_params[param]
            min_val, max_val, default_val = param_ranges[param]
            print(f"  {param:20s} = {value:12.6g}  "
                  f"(range: [{min_val:10.6g}, {max_val:10.6g}], "
                  f"default: {default_val:10.6g})")
        print()

        # Run the experiment
        print(f"[INFO] Executing: {' '.join(cmd)}")
        print()

        try:
            subprocess.check_call(cmd)
            print(f"[SUCCESS] Run {run_idx+1}/{num_runs} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Run {run_idx+1}/{num_runs} failed with return code {e.returncode}")
            print(f"[WARNING] Continuing with next run...")
        except KeyboardInterrupt:
            print(f"\n[ABORT] User interrupted. Stopping after {run_idx} runs.")
            sys.exit(1)

        print()

    print(f"{'='*70}")
    print(f"[DONE] All {num_runs} grid search combinations complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
