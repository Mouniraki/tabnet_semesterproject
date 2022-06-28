import argparse
import os
import pickle
import sys
import time
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

sys.path.append("..")
sys.path.append(".")

from src.utils.data import one_hot_encode
from src.utils.data import diff
from src.utils.hash import fast_hash
from src.utils.counter import ExpansionCounter, CounterLimitExceededError
from src.transformations import TransformationGenerator
from src.transformations import CategoricalFeature, NumFeature
from src.search import a_star_search as generalized_a_star_search

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import autonotebook as tqdm

from utils import *
from train import default_model_dict as model_dict

from loaders import shape_dict
from exp.framework import ExperimentSuite
from exp.utils import TorchWrapper
from exp import settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--results_dir", default="../out", type=str)
    parser.add_argument(
        "--dataset",
        default="ieeecis",
        choices=["ieeecis", "twitter_bot", "home_credit"],
        type=str,
    )
    parser.add_argument("--attack", default="greedy", type=str)
    parser.add_argument("--cost_bound", default=None, type=float)
    parser.add_argument("--model", default="ieeecis", type=str)

    # TabNet specific parameters
    parser.add_argument("--n_d", default=16, type=int)
    parser.add_argument("--n_a", default=16, type=int)
    parser.add_argument("--n_shared", default=2, type=int)
    parser.add_argument("--n_ind", default=2, type=int)
    parser.add_argument("--n_steps", default=4, type=int)
    parser.add_argument("--relax", default=1.2, type=float)
    parser.add_argument("--vbs", default=512, type=int)

    parser.add_argument("--model_path", default="../models/default.pt", type=str)
    parser.add_argument(
        "--utility_type",
        default="maximum",
        choices=["maximum", "satisficing", "cost-restrictred", "average-attack-cost", "success_rate"],
        type=str,
    )
    parser.add_argument(
        "--satisfaction-value", default=-1, type=float, help="Value for satisfaction"
    )
    parser.add_argument(
        "--max-cost-value", default=-1, type=float, help="Max-cost value"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()


def get_utility(results, cost_orig, X_test, y, mode="maximum", cost=-1, t_value=-1):
    total_ut = 0
    for i, r in results.iterrows():
        #print(i, r.cost, cost_orig.iloc[int(r.orig_index)], r.orig_index)
        #if y.iloc[int(r.orig_index)] == 0: No need for it, only y = target is evaluated here
        #    continue
        if np.isnan(r.cost):
            if mode == "success_rate":
                total_ut += 1
            continue
        else:
            if mode == "maximum":
                total_ut += max(cost_orig.iloc[int(r.orig_index)] - r.cost, 0)
            elif mode == "satisficing":
                if cost_orig.iloc[int(r.orig_index)] - r.cost > t_value:
                    total_ut += max(cost_orig.iloc[int(r.orig_index)] - r.cost, 0)
            elif mode == "cost-restrictred":
                if r.cost < cost:
                    total_ut += max(cost_orig.iloc[int(r.orig_index)] - r.cost, 0)
            elif mode == "average-attack-cost":
                if r.cost > 0.001:
                    total_ut += r.cost
    
    if mode == "success_rate":
        return total_ut / len(results)
    
    return total_ut / len(X_test)


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # print("Cuda Device Available")
        # print("Name of the Cuda Device: ", torch.cuda.get_device_name())
        # print("GPU Computational Capablity: ", torch.cuda.get_device_capability())
    else:
        device = torch.device("cpu")

    eval_settings = settings.setup_dataset_eval(args.dataset, args.data_dir, seed=0)
    experiment_path = settings.get_experiment_path(
        args.results_dir, args.model_path, args.attack, args.cost_bound
    )
    if os.path.isfile(experiment_path) and not args.force:
        print(f"{experiment_path} already exists. Skipping attack...")
    else:
        if args.model == "tabnet_ieeecis":
            net = model_dict[args.model](inp_dim=shape_dict[args.dataset], 
                n_d=args.n_d, 
                n_a=args.n_a, 
                n_shared=args.n_shared, 
                n_ind=args.n_ind, 
                n_steps=args.n_steps, 
                relax=args.relax, 
                vbs=args.vbs).to(device)
        else:
            net = model_dict[args.model](inp_dim=shape_dict[args.dataset]).to(device)
        net.load_state_dict(torch.load(args.model_path))
        net.eval()
        clf = TorchWrapper(net, device)

        exp_suite = ExperimentSuite(
            clf,
            eval_settings.working_datasets.X_test,
            eval_settings.working_datasets.y_test,
            target_class=eval_settings.target_class,
            cost_bound=args.cost_bound,
            spec=eval_settings.spec,
            gain_col=eval_settings.gain_col,
            dataset=eval_settings.working_datasets.dataset,
            iter_lim=1000
        )
        preds = clf.predict(eval_settings.working_datasets.X_test)
        #print(preds[0], eval_settings.working_datasets.X_test.head())
        print("Acc: ", sum(preds == eval_settings.working_datasets.y_test) / len(eval_settings.working_datasets.X_test))

        attack_config = {a.name: a for a in eval_settings.experiments}[args.attack]

        results = exp_suite.run(attack_config)
        results.to_csv(experiment_path)

    result = pd.read_csv(experiment_path)
    ut = get_utility(
            result,
            eval_settings.working_datasets.orig_cost,
            eval_settings.working_datasets.X_test,
            eval_settings.working_datasets.orig_y,
            mode=args.utility_type,
            cost=args.max_cost_value,
            t_value=args.satisfaction_value,
        )
    print(ut)
    if (args.utility_type == "success_rate"):
        print("Rob Acc: ", ut)

if __name__ == "__main__":
    main()
