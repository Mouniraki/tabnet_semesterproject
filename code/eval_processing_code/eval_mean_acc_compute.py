import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="", type=str)
    parser.add_argument(
        "--keep_features",
        nargs="+",
        default=["n_steps", "n_shared", "n_ind", "accuracy"],
        choices=["n_steps", "n_shared", "n_ind", "eps_value", "n_attack_iters", "accuracy", "average-attack-cost", "success_rate", "rob_accuracy"],
        type=str
    )
    return parser.parse_args()

def main():
    # Get the command-line arguments
    args = get_args()

    # Read the file
    df = pd.read_csv(args.data)

    # Compute & print the average accuracy
    print(df['accuracy'].mean())

if __name__ == "__main__":
    main()