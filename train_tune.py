import argparse
import json
from pathlib import Path

from train import train

def parse_args():
    """
    Parse command-line arguments for hyperparameter tuning.
    """
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning for learning rate"
    )
    parser.add_argument(
        "--lrs", nargs='+', type=float,
        default=[1e-5, 3e-5, 5e-5],
        help="List of learning rates to try"
    )
    parser.add_argument(
        "--output", type=str,
        default="tuning_results.json",
        help="Path to save tuning results as JSON"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = []

    for lr in args.lrs:
        print(f"🚀 Starting training with learning rate = {lr}")
        # train() returns training metrics or None; adjust train() accordingly
        metrics = train(lr=lr)
        print(f"✅ Finished lr={lr} -> metrics: {metrics}\n")
        results.append({"learning_rate": lr, "metrics": metrics})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"🎯 Tuning complete. Results saved to {out_path}")


if __name__ == "__main__":
    main()
