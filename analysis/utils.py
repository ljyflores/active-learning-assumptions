import json


def open_folder(
    task_name: str,
    model_name: str,
    num_samples: int,
    num_shuffles: int,
    num_datapoints: int,
):
    base_path = "/home/mila/f/floresl/active-learning-assumptions/outputs"
    return json.load(
        open(
            f"{base_path}/{task_name}/{model_name}_{num_samples}_samples_{num_datapoints}_datapoints_{num_shuffles}_shuffles_200_eps_bs_8_lr_5e-05.json",
            "r",
        )
    )
