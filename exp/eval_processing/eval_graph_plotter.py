import argparse
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
        default=["accuracy", "cost-restricted", "average-attack-cost", "success_rate", "rob_accuracy"],
        choices=["accuracy", "cost-restricted", "average-attack-cost", "success_rate", "rob_accuracy"],
        type=str
    )
    parser.add_argument(
        "--fix_n_steps",
        default=None,
        type=int
    )
    parser.add_argument(
        "--fix_n_shared",
        default=None,
        type=int
    )
    parser.add_argument(
        "--fix_n_ind",
        default=None,
        type=int
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
    data = pd.read_csv(args.data)

    # Setting the selection condition on the n_steps value
    if args.fix_n_steps != None:
        n_steps_cond = data["n_steps"] == args.fix_n_steps
        n_steps_str = f"n_steps = {args.fix_n_steps}"
    else:
        n_steps_cond = True
        n_steps_str = ""

    # Setting the selection condition on the n_shared value
    if args.fix_n_shared != None:
        n_shared_cond = data["n_shared"] == args.fix_n_shared
        n_shared_str = f"n_shared = {args.fix_n_shared}"
    else:
        n_shared_cond = True
        n_shared_str = ""

    # Setting the selection condition on the n_ind value
    if args.fix_n_ind != None:
        n_ind_cond = data["n_ind"] == args.fix_n_ind
        n_ind_str = f"n_ind = {args.fix_n_ind}"
    else:
        n_ind_cond = True
        n_ind_str = ""

    # Performing the selection on the data
    if args.fix_n_steps == None and args.fix_n_shared == None and args.fix_n_ind == None:
        df = data
    else:
        df = data[n_steps_cond & n_shared_cond & n_ind_cond]

    source_col_loc = df.columns.get_loc('n_steps')
    df['model_params'] = df.iloc[:,source_col_loc:source_col_loc+3].apply(
        lambda x: ",".join(x.astype(str)), axis=1
    )

    if args.plot_type == 'heat':
        mod_df = df[["accuracy","cost-restricted","average-attack-cost","success_rate","rob_accuracy"]]
        s = sns.heatmap(mod_df, yticklabels=df['model_params'])

        base_title = f"Heatmap of eval criterions for some TabNet config., with eps={df['eps_val'].iloc[0]}, n_attack_iters={df['n_attack_iters'].iloc[0]}"
        text_arr = np.array([n_steps_str, n_shared_str, n_ind_str])
        text_arr = np.array2string(text_arr[text_arr != ''], separator=', ').replace('[', '').replace(']', '')
        if len(text_arr) > 0:
            s.set_title("\n".join(wrap(f"{base_title} and conditioned on {text_arr}", 80)))
        else:
            s.set_title("\n".join(wrap(base_title, 80)))

        s.set_ylabel("Model config. [n_steps, n_shared, n_ind]")
        plt.show()

    else:
        # Perform projection on requested columns
        proj_df = df[np.append(['model_params'], args.plot_for)]

        # Plotting the graph (bar-kind for the moment)
        proj_df.plot(x = 'model_params', y = args.plot_for, kind = "bar")

        base_title = f"Bar plot of {args.plot_for} for some TabNet config., with eps={df['eps_val'].iloc[0]}, n_attack_iters={df['n_attack_iters'].iloc[0]}"
        text_arr = np.array([n_steps_str, n_shared_str, n_ind_str])
        text_arr = np.array2string(text_arr[text_arr != ''], separator=', ').replace('[', '').replace(']', '')
        if len(text_arr) > 0:
            plt.title("\n".join(wrap(f"{base_title}, keeping results only for models with {text_arr}", 80)))
        else:
            plt.title("\n".join(wrap(base_title, 80)))

        plt.xlabel("Model configuration [n_steps, n_shared, n_ind]")
        plt.show()

if __name__ == "__main__":
    main()