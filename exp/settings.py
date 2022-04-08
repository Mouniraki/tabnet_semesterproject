from dataclasses import dataclass
from pathlib import Path

from dotmap import DotMap

from exp import loaders
from exp.framework import AttackConfig
from src.transformations import CategoricalFeature
from src.transformations import NumFeature
from src.transformations import BinaryFeature


def get_experiment_path(results_dir, model_path, experiment_name, cost_bound, label=""):
    model_name = Path(model_path.replace(".", "_")).stem
    return Path(results_dir) / f"{model_name}_{experiment_name}_{cost_bound}{label}"


generic_experiments = [
    AttackConfig(name="random", scoring="greedy", heuristic="random", beam_size=1),
    AttackConfig(name="pgd", algo="pgd", kwargs=dict(steps=100)),
    AttackConfig(
        name="greedy", scoring="hc_ratio", heuristic="confidence", beam_size=1,
    ),
    AttackConfig(
        name="greedy_delta",
        scoring="delta_hc_ratio",
        heuristic="confidence",
        beam_size=1,
    ),
    # AttackConfig(
    #     name="greedy_beam10", scoring="hc_ratio", heuristic="confidence", beam_size=10,
    # ),
    # AttackConfig(
    #     name="greedy_beam100",
    #     scoring="hc_ratio",
    #     heuristic="confidence",
    #     beam_size=100,
    # ),
    # AttackConfig(
    #     name="greedy_full", scoring="hc_ratio", heuristic="confidence", beam_size=None,
    # ),
    # AttackConfig(
    #     name="submod_greedy_es_s_at_rf",
    #     algo="greedy",
    #     kwargs=dict(
    #         early_stop=True,
    #         criterion="score",
    #         change_feature_once=False,
    #         all_transformations=True,
    #     ),
    # ),
    # AttackConfig(
    #     name="submod_greedy_es_s_bt_rf",
    #     algo="greedy",
    #     kwargs=dict(
    #         early_stop=True,
    #         criterion="score",
    #         change_feature_once=False,
    #         all_transformations=False,
    #     ),
    # ),
    # AttackConfig(
    #     name="submod_greedy_es_ds_at_of",
    #     algo="greedy",
    #     kwargs=dict(
    #         early_stop=True,
    #         criterion="delta_score",
    #         change_feature_once=True,
    #         all_transformations=True,
    #     ),
    # ),
    # AttackConfig(
    #     name="submod_greedy_es_ds_bt_rf",
    #     algo="greedy",
    #     kwargs=dict(
    #         early_stop=True,
    #         criterion="delta_score",
    #         change_feature_once=False,
    #         all_transformations=False,
    #     ),
    # ),
    # AttackConfig(
    #     name="submod_greedy_es_ds_at_rf",
    #     algo="greedy",
    #     kwargs=dict(
    #         early_stop=True,
    #         criterion="delta_score",
    #         change_feature_once=False,
    #         all_transformations=True,
    #     ),
    # ),
]

common_a_star_family_experiments = [
    AttackConfig(
        name="astar_subopt_beam1",
        scoring="a_star",
        heuristic="confidence",
        beam_size=1,
    ),
    AttackConfig(
        name="astar_subopt_beam10",
        scoring="a_star",
        heuristic="confidence",
        beam_size=10,
    ),
    AttackConfig(
        name="astar_subopt_beam100",
        scoring="a_star",
        heuristic="confidence",
        beam_size=100,
    ),
    # AttackConfig(name="astar_subopt", scoring="a_star", heuristic="confidence"),
    AttackConfig(name="ucs", scoring="a_star", heuristic="zero"),
]

common_ps_family_experiments = [
    AttackConfig(
        name="ps_subopt_beam1", scoring="ps", heuristic="confidence", beam_size=1,
    ),
    AttackConfig(
        name="ps_subopt_beam10", scoring="ps", heuristic="confidence", beam_size=10,
    ),
    AttackConfig(
        name="ps_subopt_beam100", scoring="ps", heuristic="confidence", beam_size=100
    ),
    # AttackConfig(name="ps_subopt", scoring="ps", heuristic="confidence",),
]


def _get_working_datasets(data_test, target_col):
    X_test = data_test.X_test
    y_test = data_test.y_test
    orig_df = data_test.orig_df
    return DotMap(
        X_test=X_test,
        y_test=y_test,
        orig_y=orig_df[target_col],
        orig_cost=data_test.cost_orig_df,
        orig_df=orig_df,
    )


def get_dataset(dataset, data_dir, mode, seed=0, same_cost=False, cat_map=False):
    if dataset in ["ieeecis", "tabnet_noshared", "tabnet_noind", "tabnet_norelax", "tabnet_highrelax", "tabnet_low_na_nd", "tabnet_lowsteps", "tabnet_lowbatch"]:
        data = loaders.IEEECISDataset(
            data_dir,
            mode=mode,
            balanced=True,
            discrete_one_hot=True,
            seed=seed,
            same_cost=same_cost,
            cat_map=cat_map,
        )

    elif dataset == "twitter_bot":
        data = loaders.TwitterBotDataset(
            data_dir,
            seed=seed,
            mode=mode,
            balanced=True,
            discrete_one_hot=True,
            same_cost=same_cost,
            cat_map=cat_map,
        )

    elif dataset == "home_credit":
        data = loaders.HomeCreditDataset(
            data_dir,
            mode=mode,
            balanced=True,
            discrete_one_hot=True,
            seed=seed,
            same_cost=same_cost,
            cat_map=cat_map,
        )
    elif dataset == "syn":
        data = loaders.Synthetic(
            data_dir, mode=mode, seed=seed, same_cost=same_cost, cat_map=cat_map
        )
        data = loaders.Synthetic(data_dir, mode=mode, seed=seed, same_cost=same_cost,)
    return data


def _get_working_datasets(data_test, target_col):
    X_test = data_test.X_test
    y_test = data_test.y_test
    orig_df = data_test.orig_df
    return DotMap(
        X_test=X_test,
        y_test=y_test,
        orig_y=orig_df[target_col],
        orig_cost=data_test.cost_orig_df,
        orig_df=orig_df,
        dataset=data_test,
    )


@dataclass
class EvalSettings:
    target_col: str
    gain_col: str
    spec: list
    target_class: int
    experiments: list
    working_datasets: DotMap


def setup_dataset_eval(dataset, data_dir, seed=0):
    if dataset in ["ieeecis", "tabnet_noshared", "tabnet_noind", "tabnet_norelax", "tabnet_highrelax", "tabnet_low_na_nd", "tabnet_lowsteps", "tabnet_lowbatch"]:
        target_col = "isFraud"
        gain_col = "TransactionAmt"
        target_class = 0
        data_test = get_dataset(dataset, data_dir, mode="test", seed=seed, cat_map=True)
        working_datasets = _get_working_datasets(data_test, target_col)
        spec = [
            # Product type
            # CategoricalFeature(name="ProductCD", cost=1),
            # Card brand and type
            CategoricalFeature(name="card_type", cost=1),
            # Receiver email domain
            # CategoricalFeature(name="R_emaildomain", cost=0.2),
            # Payee email domain
            CategoricalFeature(name="P_emaildomain", cost=1),
            # Payment device
            CategoricalFeature(name="DeviceType", cost=1),
        ]
        experiments = (
            generic_experiments
            + common_ps_family_experiments
            + common_a_star_family_experiments
        ) + [
            AttackConfig(
                name="greedy_delta_beam10",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=10,
            ),
            AttackConfig(
                name="greedy_delta_beam100",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=100,
            ),
            AttackConfig(name="pgd_1k", algo="pgd", kwargs=dict(steps=1000)),
        ]

    if dataset == "twitter_bot":
        target_col = "is_bot"
        gain_col = "value"
        target_class = 0
        data_test = get_dataset(dataset, data_dir, mode="test", seed=seed, cat_map=True)
        working_datasets = _get_working_datasets(data_test, target_col)
        spec = [
            NumFeature(name="user_tweeted", inc_cost=2, integer=True).infer_range(
                working_datasets.orig_df, bins=10,
            ),
            NumFeature(name="user_replied", inc_cost=2, integer=True).infer_range(
                working_datasets.orig_df, bins=10,
            ),
            NumFeature(name="likes_per_tweet", inc_cost=0.025).infer_range(
                working_datasets.orig_df, bins=10,
            ),
            NumFeature(name="retweets_per_tweet", inc_cost=0.025).infer_range(
                working_datasets.orig_df, bins=10
            ),
        ]
        ps_family_experiments = common_ps_family_experiments + [
            AttackConfig(
                name="astar_opt",
                scoring="a_star",
                heuristic="linear",
                kwargs=dict(cost_coef=0.5, cost_min_step_value=0.01),
            ),
        ]
        a_star_family_experiments = common_a_star_family_experiments + [
            AttackConfig(
                name="ps_opt",
                scoring="ps",
                heuristic="linear",
                kwargs=dict(cost_coef=0.5, cost_min_step_value=0.025),
            )
        ]
        experiments = (
            generic_experiments + ps_family_experiments + a_star_family_experiments
        ) + [
            AttackConfig(
                name="greedy_delta_beam10",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=10,
            ),
            AttackConfig(
                name="greedy_delta_beam100",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=100,
            ),
            AttackConfig(name="pgd_1k", algo="pgd", kwargs=dict(steps=1000)),
        ]

    if dataset == "home_credit":
        target_col = "TARGET"
        gain_col = "AMT_CREDIT"
        data_test = get_dataset(dataset, data_dir, mode="test", seed=seed, cat_map=True)
        target_class = 1
        working_datasets = _get_working_datasets(data_test, target_col)
        df = working_datasets.orig_df
        spec = [
            CategoricalFeature("NAME_CONTRACT_TYPE", 0.1),
            CategoricalFeature("NAME_TYPE_SUITE", 0.1),
            BinaryFeature("FLAG_EMAIL", cost=0.1),
            CategoricalFeature("WEEKDAY_APPR_PROCESS_START", 0.1),
            CategoricalFeature("HOUR_APPR_PROCESS_START", 0.1),
            BinaryFeature("FLAG_MOBIL", cost=10),
            BinaryFeature("FLAG_EMP_PHONE", cost=10),
            BinaryFeature("FLAG_WORK_PHONE", cost=10),
            BinaryFeature("FLAG_CONT_MOBILE", cost=10),
            BinaryFeature("FLAG_OWN_CAR", cost=100),
            BinaryFeature("FLAG_OWN_REALTY", cost=100),

            BinaryFeature("REG_REGION_NOT_LIVE_REGION", cost=100),
            BinaryFeature("REG_REGION_NOT_WORK_REGION", cost=100),
            BinaryFeature("LIVE_REGION_NOT_WORK_REGION", cost=100),
            BinaryFeature("REG_CITY_NOT_LIVE_CITY", cost=100),
            BinaryFeature("REG_CITY_NOT_WORK_CITY", cost=100),
            BinaryFeature("LIVE_CITY_NOT_WORK_CITY", cost=100),

            CategoricalFeature("NAME_INCOME_TYPE", 100),
            CategoricalFeature("cluster_days_employed", 100),
            CategoricalFeature("NAME_HOUSING_TYPE", 100),
            CategoricalFeature("OCCUPATION_TYPE", 100),
            CategoricalFeature("ORGANIZATION_TYPE", 100),
            CategoricalFeature("NAME_FAMILY_STATUS", 1000),
            CategoricalFeature("NAME_EDUCATION_TYPE", 1000),
            BinaryFeature("has_children", cost=1000),
            NumFeature("AMT_INCOME_TOTAL", inc_cost=1).infer_range(df, bins=30),
            NumFeature("EXT_SOURCE_1", inc_cost=1000).infer_range(df, bins=30),
            NumFeature("EXT_SOURCE_2", inc_cost=1000).infer_range(df, bins=30),
            NumFeature("EXT_SOURCE_3", inc_cost=1000).infer_range(df, bins=30),
        ]
        experiments = [
            AttackConfig(
                name="random", scoring="greedy", heuristic="random", beam_size=1
            ),
            AttackConfig(
                name="greedy_delta",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=1,
            ),
        ]

    return EvalSettings(
        target_col=target_col,
        gain_col=gain_col,
        spec=spec,
        target_class=target_class,
        working_datasets=_get_working_datasets(data_test, target_col),
        experiments=experiments,
    )
