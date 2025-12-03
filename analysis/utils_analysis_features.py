import pandas as pd
import torch
from utils import open_folder


def load_features(task_name: str, model_name: str):
    base_path = "/home/mila/f/floresl/active-learning-assumptions/outputs"
    unlabeled_features = pd.read_csv(
        f"{base_path}/{task_name}/{model_name}_baseline_metrics.csv"
    )
    unlabeled_hidden_states = torch.load(
        f"{base_path}/{task_name}/{model_name}_baseline_hidden_states.pt",
        map_location=(
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda")
        ),
    )
    return {"features": unlabeled_features, "hidden_states": unlabeled_hidden_states}


def compute_avg_l2_dist_from_center(tensor: torch.Tensor):
    assert tensor.dim() == 2
    avg = tensor.mean(dim=0)
    return float(((tensor - avg) ** 2).sum(dim=-1).mean())


def add_numeric_features_of_samples(
    df_sampling: pd.DataFrame, df_features: pd.DataFrame
):
    df_sampling["avg_log_token_prob_mean"] = df_sampling["indices"].apply(  # type: ignore
        lambda lst: df_features.loc[lst]["avg_token_log_prob"].mean()  # type: ignore
    )
    df_sampling["avg_token_entropy_mean"] = df_sampling["indices"].apply(  # type: ignore
        lambda lst: df_features.loc[lst]["avg_token_entropy"].mean()  # type: ignore
    )
    df_sampling["beam_search_ratio_mean"] = df_sampling["indices"].apply(  # type: ignore
        lambda lst: df_features.loc[lst]["beam_score_ratios_2"].mean()  # type: ignore
    )
    df_sampling["beam_search_impt_mean"] = df_sampling["indices"].apply(  # type: ignore
        lambda lst: df_features.loc[lst]["beam_score_importance_weighted_sum_2"].mean()  # type: ignore
    )
    df_sampling["dropout_meteor_var_mean"] = df_sampling["indices"].apply(  # type: ignore
        lambda lst: df_features.loc[lst]["dropout_meteor_score"].mean()  # type: ignore
    )
    df_sampling["dropout_disagreement_mean"] = df_sampling["indices"].apply(  # type: ignore
        lambda lst: df_features.loc[lst]["dropout_disagreement"].mean()  # type: ignore
    )
    df_sampling["dropout_bald_mean"] = df_sampling["indices"].apply(  # type: ignore
        lambda lst: df_features.loc[lst]["dropout_bald"].mean()  # type: ignore
    )
    return df_sampling


def add_hidden_state_features_of_samples(
    df_sampling: pd.DataFrame, hidden_states: torch.Tensor
):
    df_sampling["baseline_avg_l2_from_center"] = df_sampling["indices"].apply(  # type: ignore
        lambda lst: compute_avg_l2_dist_from_center(hidden_states[lst])  # type: ignore
    )
    return df_sampling


def load_all_features(
    task_name: str,
    model_name: str,
    num_samples: int,
    num_shuffles: int,
    num_datapoints: int,
):
    sampled_runs = open_folder(
        task_name=task_name,
        model_name=model_name,
        num_samples=num_samples,
        num_shuffles=num_shuffles,
        num_datapoints=num_datapoints,
    )
    df_sampled_runs = pd.DataFrame(sampled_runs)
    df_sampled_runs = pd.concat(
        [
            df_sampled_runs,
            pd.DataFrame.from_records(
                [result[0] for result in df_sampled_runs["eval_metrics"]]
            ),
        ],
        axis=1,
    )
    features_dict = load_features(task_name=task_name, model_name=model_name)
    df_features = features_dict["features"]
    hidden_states = features_dict["hidden_states"]

    df_sampling = add_numeric_features_of_samples(
        df_sampling=df_sampled_runs,
        df_features=df_features,
    )
    df_sampling = add_hidden_state_features_of_samples(
        df_sampling=df_sampling,
        hidden_states=hidden_states,
    )
    return df_sampling
