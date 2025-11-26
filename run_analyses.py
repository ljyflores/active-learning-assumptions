import argparse
import json
import os

from dataclasses import asdict
from src.loading import load_data
from utils import run_training_on_multiple_samples

model_mapping: dict[str, str] = {
    "bart": "facebook/bart-base",
    "t5": "google/flan-t5-base",
    "mbart": "facebook/mbart-large-50",
    "llama": "unsloth/Meta-Llama-3.1-8B",
    "gemma": "unsloth/gemma-2-2b-it-bnb-4bit",
}
MODEL_OUTPUT_PATH = f"./results"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--num_datapoints_per_sample", type=int, required=True)
    parser.add_argument("--num_shuffles_per_sample", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--suffix", type=str, required=False, default="")
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--filename", type=str, required=False, default=None)
    args_dict = vars(parser.parse_args())

    os.environ["WANDB_PROJECT"] = f"al-assumptions-{args_dict['num_samples']}-{args_dict['num_datapoints_per_sample']}-{args_dict['num_shuffles_per_sample']}-{str(args_dict['dataset']).split('/')[-1]}"  # type: ignore

    dataset = args_dict.pop("dataset")
    filename = args_dict.pop("filename")

    eps_suffix = f"_{args_dict['epochs']}_eps"
    hyperparam_suffix = f"_bs_{args_dict['batch_size']}_lr_{args_dict['learning_rate']}"
    output_folder = f"outputs/{dataset.split('/')[-1]}"
    output_path = f"{output_folder}/{args_dict['num_samples']}_samples_{args_dict['num_datapoints_per_sample']}_datapoints_{args_dict['num_shuffles_per_sample']}_shuffles_{eps_suffix}{hyperparam_suffix}{args_dict.pop("suffix")}.json"
    args_dict["output_path"] = output_path
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    data = load_data(
        dataset=dataset,
        model_path=model_mapping[args_dict["model_name"]],
    )

    results = run_training_on_multiple_samples(
        model_name=args_dict["model_name"],
        train_df=data["train"],
        val_df=data["val"],
        test_df=data["test"],
        num_samples=args_dict["num_samples"],
        num_datapoints_per_sample=args_dict["num_datapoints_per_sample"],
        num_shuffles_per_sample=args_dict["num_shuffles_per_sample"],
        epochs=args_dict["epochs"],
        batch_size=args_dict["batch_size"],
        learning_rate=args_dict["learning_rate"],
    )
    with open(output_path, "w") as f:
        json.dump([asdict(item) for item in results], f, indent=4)
