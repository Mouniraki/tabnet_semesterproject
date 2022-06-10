import argparse
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="", type=str)
    parser.add_argument(
        "--plot_for",
        nargs="+",
        default=["accuracy", "average-attack-cost", "rob_accuracy"],
        choices=["accuracy", "average-attack-cost", "success_rate", "rob_accuracy"],
        type=str
    )
    parser.add_argument(
        "--avg_for_n_steps",
        action="store_true"
    )
    parser.add_argument(
        "--avg_for_n_shared",
        action="store_true"
    )
    parser.add_argument(
        "--avg_for_n_ind",
        action="store_true"
    )
    parser.add_argument(
        "--plot_type",
        default="bar",
        choices=["bar", "heat"],
        type=str
    )
    return parser.parse_args()

def main():
    # Get the command-line arguments
    args = get_args()

    # Read the file
    df = pd.read_csv(args.data)

    # Retrieving the eps and attack_iters values for graph titles
    eps_val, attack_iters = df[['eps_val', 'n_attack_iters']].iloc[0]

    # Selecting the column on which to group and aggregate data
    if args.avg_for_n_shared:
        xlabel_col = 'n_shared'
    elif args.avg_for_n_ind:
        xlabel_col = 'n_ind'
    else:
        xlabel_col = 'n_steps'

    # Perform aggregation of data
    df = df[np.append([xlabel_col], args.plot_for)].groupby([xlabel_col], as_index=False).mean()

    print(df)

    # Generate type of plot (either bar plot or heatmap)
    if args.plot_type == 'heat':
        mod_df = df[args.plot_for]
        s = sns.heatmap(mod_df, yticklabels=df[xlabel_col])

        base_title = f"Heatmap of {args.plot_for} for some TabNet config., with eps={eps_val}, n_attack_iters={attack_iters} on {xlabel_col} parameter"
        s.set_title("\n".join(wrap(base_title, 80)))
        s.set_ylabel(f"Aggregate parameter ({xlabel_col})")
        plt.show()

    else:
        # Plotting the graph (bar-kind for the moment)
        df.plot(x = xlabel_col, y = args.plot_for, kind = "bar", color=['green'])

        base_title = f"Bar plot of {args.plot_for} for some TabNet config., with eps={eps_val}, n_attack_iters={attack_iters} on {xlabel_col} parameter"
        plt.title("\n".join(wrap(base_title, 80)))

        plt.xlabel(f"Aggregate parameter ({xlabel_col})")
        plt.ylim([0.3, 0.4])
        plt.show()

if __name__ == "__main__":
    main()