import argparse
import subprocess
import sys
import time
from pathlib import Path


def get_latest_checkpoint(experiment_name: str) -> str | None:
    """Find the latest checkpoint file for the given experiment."""
    log_dir = Path("logs") / "rsl_rl" / experiment_name
    if not log_dir.exists():
        return None

    # Get all timestamped directories
    runs = sorted([d for d in log_dir.iterdir() if d.is_dir()], reverse=True)
    if not runs:
        return None

    # Check the latest run directory
    latest_run = runs[0]

    # Find all model_*.pt files and get the one with highest iteration
    checkpoints = list(latest_run.glob("model_*.pt"))
    if not checkpoints:
        return None

    # Sort by the number in model_<number>.pt
    def get_iteration(p: Path):
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    latest_checkpoint = sorted(checkpoints, key=get_iteration, reverse=True)[0]
    return str(latest_checkpoint)


def main():
    parser = argparse.ArgumentParser(description="Train Booster T1 using mjlab")
    parser.add_argument("--test", action="store_true", help="Play trained policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint file",
    )
    parser.add_argument(
        "--task", type=str, default="T1-Reach-v0", help="Task ID to train or play"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of parallel environments for training (ignored in play mode)",
    )
    args, unknown = parser.parse_known_args()

    task_name = args.task
    experiment_name = (
        task_name.replace("-v0", "") + f"{time.strftime('-%Y%m%d-%H%M%S')}"
    )

    if args.test:
        checkpoint = args.checkpoint
        if not checkpoint:
            checkpoint = get_latest_checkpoint(experiment_name)
            if checkpoint:
                print(f"Automatically selected latest checkpoint: {checkpoint}")
            else:
                print(
                    f"Error: No checkpoints found for experiment '{experiment_name}' in logs/rsl_rl/"
                )
                sys.exit(1)

        print(f"Playing task {task_name} using checkpoint {checkpoint}...")
        cmd = [
            "uv",
            "run",
            "play",
            task_name,
            "--checkpoint-file",
            checkpoint,
            "--viewer=viser",
        ]
    else:
        print(f"Training task {task_name}...")
        # Assuming 'train' script is available in the environment via mjlab
        cmd = ["uv", "run", "train", task_name, "--video=True", f"--env.scene.num-envs={args.num_envs}"]

    # Pass unknown args to mjlab if needed, but for now kept simple
    if unknown:
        print(f"Warning: Ignoring unknown arguments: {unknown}")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: 'uv' command not found. Please install uv and run 'uv sync'.")
        print(
            "Alternatively, ensure 'train' and 'play' scripts from mjlab are in your PATH."
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")


if __name__ == "__main__":
    main()
