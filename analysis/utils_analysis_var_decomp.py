import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import f_oneway  # type: ignore


def flatten(xss: list[list[object]]):
    return [x for xs in xss for x in xs]


def load_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def between_within_var_breakdown(results_by_trial: list[list[float]], G: int, N: int):

    ss_total = float(np.var(flatten(results_by_trial))) * G * N
    ss_within = float(
        sum([(N * np.var(trial_results)) for trial_results in results_by_trial])
    )

    ss_between = float(
        G * N * np.var([np.mean(trial_results) for trial_results in results_by_trial])
    )
    output = dict[str, float]()
    output["ss_total"] = ss_total
    output["ss_within"] = ss_within
    output["ss_between"] = ss_between
    output["ss_total_plus_between"] = ss_within + ss_between
    return output


def compute_percent_from_within(chrfs_by_sample: list[list[float]]):
    ss_total, ss_within, _, _ = between_within_var_breakdown(
        results_by_trial=chrfs_by_sample,
        G=len(chrfs_by_sample),
        N=len(chrfs_by_sample[0]),
    )
    return float(ss_within) / float(ss_total)


def compute_percents(folder: str, path: str):
    percents = list[list[float]]()
    for lr in (
        "1e-05",
        "5e-05",
        "0.0001",
    ):
        percents_by_bs = list[float]()
        for bs in ("8", "16", "32"):
            path = f"sampling_results_20reps_5eps_bs_{bs}_lr_{lr}_test_dataset.json"
            print(lr, bs)
            percent_attr_to_ordering = get_decomp_results(folder, path)
            print("\n\n")
            percents_by_bs.append(float(percent_attr_to_ordering))
        percents.append(percents_by_bs)
    return percents


def get_decomp_results(folder: str, path: str):
    sampling_reps = load_json(f"{folder}/{path}")

    sampled_chrfs = [
        [trial["eval_chrf"] for trial in rep["eval_metrics"]] for rep in sampling_reps
    ]
    percent_within = compute_percent_from_within(sampled_chrfs)
    print(f_oneway(*sampled_chrfs))
    return percent_within


def get_decomp_results_by_k(folder: str):
    path = "sampling_resultsvar_decomp_20reps_5eps_bs_8_lr_5e-05_test_dataset.json"

    sampling_reps = load_json(f"{folder}/{path}")
    sampled_chrfs = [
        [trial["eval_chrf"] for trial in rep["eval_metrics"]] for rep in sampling_reps
    ]
    return [compute_percent_from_within(sampled_chrfs[:k]) for k in range(2, 201)]


def plot_matrix(percents: list[list[float]], dataset: str):
    percents = np.array(percents)  # type: ignore

    _, ax = plt.subplots()  # type: ignore
    _ = ax.imshow(percents, cmap="Blues", vmin=0, vmax=1)  # type: ignore

    ax.set_xticks(range(3), labels=["8", "16", "32"], fontsize=24)  # type: ignore
    ax.set_yticks(range(3), labels=["1e-5", "5e-5", "1e-4"], fontsize=24)  # type: ignore
    ax.set_xlabel("Batch Size", fontsize=24)  # type: ignore
    ax.set_ylabel("Learning Rate", fontsize=24)  # type: ignore
    # Loop over data dimensions and create text annotations.
    for i in range(3):
        for j in range(3):
            ax.text(  # type: ignore
                j,
                i,
                round(100 * percents[i, j], 1),  # type: ignore
                ha="center",
                va="center",
                color="w",
                size="xx-large",
            )
    ax.set_title(f"{' '.join(dataset.split('_')).title()}", fontsize=24)  # type: ignore


def plot_lines_by_num_groups(percents_by_k: list[list[float]], labels: list[str]):
    assert len(percents_by_k) == len(labels)
    for lst, label in zip(percents_by_k, labels):
        lst = lst[:200]
        plt.plot(list(range(2, len(lst) + 2)), lst, label=label)  # type: ignore
    plt.legend(fontsize=16)  # type: ignore
    plt.xticks(fontsize=20, rotation=45)  # type: ignore
    plt.yticks(fontsize=20, rotation=45)  # type: ignore
    plt.xlabel("Number of Groups", fontsize=20)  # type: ignore
    plt.ylabel("% Variance from Ordering", fontsize=20)  # type: ignore
