import argparse
import json
import numpy as np
import os
import pandas as pd
import torch
import wandb

from copy import deepcopy
from datasets import Dataset, DatasetDict  # type: ignore
from string import punctuation
from transformers import EvalPrediction, Seq2SeqTrainingArguments, TrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, PreTrainedTokenizer, PreTrainedModel  # type: ignore
from transformers.trainer_utils import EvalLoopOutput
from torch import Tensor
from trl import SFTTrainer  # type: ignore
from typing import cast

from src.loading import load_model
from src.preprocess import encode_dataset
from src.eval import compute_metrics as eval

remove_chars = punctuation + "\n"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

CLASSIFICATION_DATASETS = ["data/qnli", "data/trec6", "data/dbpedia"]
QA_DATASETS = ["data/squad", "data/hotpotqa"]
MATH_DATASETS = ["data/gsm8k", "data/math"]

model_mapping: dict[str, str] = {
    "bart": "facebook/bart-base",
    "t5": "google/flan-t5-base",
    "mbart": "facebook/mbart-large-50",
    "llama": "unsloth/Meta-Llama-3.1-8B",
    "gemma": "unsloth/gemma-2-2b-it-bnb-4bit",
}


def compute_confidence_scores(logits: Tensor):
    probs = torch.tensor(logits).softmax(dim=-1)
    log_probs = torch.log(probs)
    average_token_log_prob = log_probs.max(dim=-1).values.mean(dim=-1)
    average_token_entropy = (-1.0 * probs * log_probs).sum(dim=-1).mean(dim=-1)
    scores = dict[str, float | list[float]]()
    scores["avg_token_log_prob"] = [float(x) for x in average_token_log_prob.cpu()]
    scores["avg_token_entropy"] = [float(x) for x in average_token_entropy.cpu()]
    return scores


def formatting_func_llama(example: dict[str, str]):
    system_prompt = "You are a helpful assistant."
    user_prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}"""
    return user_prompt.format(system_prompt, example["source"], example["target"])


def formatting_func_gemma(example: dict[str, str]):
    prompt = """<bos><start_of_turn>user
{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>"""
    return prompt.format(example["source"], example["target"])


class SFTTrainerWithLogits(SFTTrainer):
    def prediction_step(
        self,
        model: PreTrainedModel,
        inputs: dict[str, object],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        # Get loss and logits from parent
        loss, preds, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )

        # Forward pass for logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        return loss, (preds, logits), labels

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool = None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        self.model.eval()
        metrics_list = list[dict[str, float | list[float]]]()
        losses = list[float]()

        for inputs in dataloader:
            loss, (preds, logits), labels = self.prediction_step(
                self.model, inputs, prediction_loss_only, ignore_keys
            )

            if loss is not None:
                losses.append(loss.item())

            if self.compute_metrics is not None:
                eval_pred = EvalPrediction(
                    predictions=(preds.cpu().numpy(), logits.cpu().numpy()),
                    label_ids=labels.cpu().numpy(),
                    inputs=(
                        inputs["input_ids"].cpu().numpy()
                        if "input_ids" in inputs
                        else None
                    ),
                )
                batch_metrics = self.compute_metrics(eval_pred)
                metrics_list.append(batch_metrics)

        # Aggregate metrics
        final_metrics = {}
        if metrics_list:
            for metric in metrics_list[0]:
                if isinstance(metrics_list[0][metric], (int, float)):
                    final_metrics[metric] = float(
                        np.mean([m[metric] for m in metrics_list])
                    )
                elif isinstance(metrics_list[0][metric], list):
                    all_values = [val for m in metrics_list for val in m[metric]]
                    final_metrics[metric] = all_values

        if losses:
            final_metrics[f"{metric_key_prefix}_loss"] = float(np.mean(losses))

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=final_metrics,
            num_samples=len(dataloader.dataset),
        )


class Seq2SeqTrainerWithLogits(Seq2SeqTrainer):
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        # Get loss, generated tokens, labels from parent
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )

        # Forward pass for logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        return loss, (generated_tokens, logits), labels

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool = None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        self.model.eval()
        metrics_list = []
        losses = []

        for inputs in dataloader:
            loss, (generated_tokens, logits), labels = self.prediction_step(
                self.model, inputs, prediction_loss_only, ignore_keys
            )

            if loss is not None:
                losses.append(loss.item())

            # Call compute_metrics per batch
            if self.compute_metrics is not None:
                eval_pred = EvalPrediction(
                    predictions=(
                        generated_tokens.cpu().numpy(),
                        logits.cpu().numpy(),
                    ),
                    label_ids=labels.cpu().numpy(),
                    inputs=(
                        inputs["input_ids"].cpu().numpy()
                        if "input_ids" in inputs
                        else None
                    ),
                )
                batch_metrics = self.compute_metrics(eval_pred)
                metrics_list.append(batch_metrics)

        # Aggregate metrics across batches
        final_metrics = {}
        if metrics_list:
            for metric in metrics_list[0].keys():
                if isinstance(metrics_list[0][metric], (int, float)):
                    final_metrics[metric] = float(
                        np.mean([m[metric] for m in metrics_list])
                    )
                elif isinstance(metrics_list[0][metric], list):
                    # Flatten list of lists
                    all_values = [val for m in metrics_list for val in m[metric]]
                    final_metrics[metric] = all_values

        if losses:
            final_metrics[f"{metric_key_prefix}_loss"] = float(np.mean(losses))

        return EvalLoopOutput(
            predictions=None,  # don’t save all predictions
            label_ids=None,
            metrics=final_metrics,
            num_samples=len(dataloader.dataset),
        )


def main(
    dataset: str,
    model_name: str,
    epochs: int = 10,
    train_batch_size: int = 2,
    train_grad_acc: int = 4,
    seed: int = 16,
    finetune_size: int = 2000,
    degenerate_case: bool = False,
    random_case: bool = False,
    suffix: str = "",
    ablate_by_confidence: bool = False,
    prefinetune_size: int | None = None,
    wandb_off: bool = False,
    get_preds_only: bool = False,
):
    # Declare metrics
    if dataset in QA_DATASETS:
        eval_metrics = ["f1", "f1_indiv"]
        indiv_metric = "f1_indiv"
    elif dataset in MATH_DATASETS:
        if dataset.endswith("gsm8k"):
            eval_metrics = [
                "exact_match_gsm8k",
                "exact_match_gsm8k_indiv",
                "rougel_indiv",
            ]
            # indiv_metric = "exact_match_gsm8k_indiv"
            indiv_metric = "rougel_indiv"
        elif dataset.endswith("math"):
            eval_metrics = [
                "exact_match_math",
                "exact_match_math_indiv",
                "rougel_indiv",
            ]
            # indiv_metric = "exact_match_math_indiv"
            indiv_metric = "rougel_indiv"
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
    elif dataset in CLASSIFICATION_DATASETS:
        eval_metrics = ["accuracy", "accuracy_indiv"]
        indiv_metric = "accuracy_indiv"
    else:
        eval_metrics = ["chrf", "chrf_indiv"]
        indiv_metric = "chrf_indiv"

    # Declare paths
    model_path = model_mapping[model_name]
    if degenerate_case:
        suffix += "_degenerate"
    if random_case:
        suffix += "_random"
    if ablate_by_confidence:
        suffix += "_confidence"
    if prefinetune_size is not None:
        suffix += f"_pft_{prefinetune_size}"
    if get_preds_only:
        suffix += "_preds"
    MODEL_OUTPUT_PATH = f"{dataset.split('/')[-1]}_eps_{epochs}_{model_path.split('/')[-1]}_seed_{seed}{suffix}"

    # Read in data
    training_df = pd.read_csv(f"{dataset}/train_labeled.csv")  # type: ignore
    unlabeled_df = pd.read_csv(f"{dataset}/train_candidate.csv")  # type: ignore
    test_df = pd.read_csv(f"{dataset}/test_sample.csv") if "eng_" in dataset else pd.read_csv(f"{dataset}/test.csv")  # type: ignore
    test_df = test_df.sort_values(
        by="source", key=lambda s: s.str.len(), ascending=False
    ).reset_index(drop=True)

    if model_name == "llama":
        response_prefix = "### Response:\n"
        formatting_func = formatting_func_llama
    elif model_name == "gemma":
        response_prefix = "model\n"
        formatting_func = formatting_func_gemma
    else:
        response_prefix = None
        formatting_func = None

    # Add to labeled data if needed
    if (prefinetune_size is not None) and (len(training_df) < prefinetune_size):
        num_samples_to_add = prefinetune_size - len(training_df)
        sampled = unlabeled_df.sample(n=num_samples_to_add, random_state=seed)
        unlabeled_df = unlabeled_df.drop(sampled.index).reset_index(drop=True)
        training_df = pd.concat([training_df, sampled]).reset_index(drop=True)

    if degenerate_case:
        unlabeled_df["target"] = unlabeled_df["source"]

    training_pool = Dataset.from_pandas(training_df)
    unlabeled_pool = Dataset.from_pandas(unlabeled_df)
    test_pool = Dataset.from_pandas(test_df)

    # Load model and tokenizer
    instantiate_model_func, tokenizer = load_model(  # type: ignore
        model_path=model_path
    )
    tokenizer = cast(PreTrainedTokenizer, tokenizer)
    model = instantiate_model_func(seed=seed)  # type: ignore
    model = cast(PreTrainedModel, model)

    def prepare_dataset(dataset: Dataset | DatasetDict):
        dataset = encode_dataset(
            dataset,
            tokenizer,
            "source",
            "target",
            id_column_name=None,
            model_name=model_name,
        )
        assert isinstance(dataset, Dataset)
        dataset.set_format(  # pyright: ignore[reportUnknownMemberType]
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return dataset

    # Declare training arguments and evaluation functions
    def clean_output_tensor(tensor: Tensor):
        tensor[tensor == -100] = 0
        return tokenizer.batch_decode(  # type: ignore
            tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def custom_preprocess_logits(logits: Tensor, labels: Tensor):
        # Example: Convert logits to predicted class IDs
        if isinstance(logits, tuple):
            logits = logits[0]  # Handle cases where logits might be a tuple
        return logits.argmax(dim=-1)

    def compute_metrics(eval_pred: EvalPrediction):
        label_raw = eval_pred.label_ids
        source_raw = eval_pred.inputs
        if isinstance(eval_pred.predictions, tuple):
            pred_raw = eval_pred.predictions[0]
            logits = eval_pred.predictions[1]
            if pred_raw.shape == logits.shape:
                pred_raw = custom_preprocess_logits(
                    torch.Tensor(logits), torch.Tensor(label_raw)
                )
        else:
            pred_raw = eval_pred.predictions
            logits = None

        pred_raw = cast(Tensor, pred_raw)
        label_raw = cast(Tensor, label_raw)
        source_raw = cast(Tensor, source_raw)

        predictions = clean_output_tensor(pred_raw)
        sources = clean_output_tensor(source_raw)
        labels = clean_output_tensor(label_raw)

        if response_prefix is not None:
            predictions = [
                p.split(response_prefix)[-1].strip(remove_chars) for p in predictions
            ]
            labels = [l.split(response_prefix)[-1].strip(remove_chars) for l in labels]

        labels = [[s] for s in labels]  # labels must be a list of LISTS
        for item in list(zip(predictions, labels, sources))[:2]:
            print(f"Source: {item[2]}\n\nLabel: {item[1]}\n\nPrediction: {item[0]}")

        all_metrics: dict[str, float | list[float] | list[str]] = eval(
            sources, predictions, labels, eval_metrics
        )  # type: ignore
        confidence_metrics = (
            compute_confidence_scores(torch.tensor(logits))
            if logits is not None
            else {}
        )
        all_metrics.update(confidence_metrics)
        if get_preds_only:
            all_metrics["predictions"] = predictions
            all_metrics["labels"] = [lst[0] for lst in labels]
            all_metrics["sources"] = sources
        return all_metrics

    training_pool_tokenized = None
    unlabeled_pool_tokenized = None
    test_pool_tokenized = None

    if model_name in ["t5", "mbart", "bart"]:
        # Tokenize datasets
        training_pool_tokenized = prepare_dataset(training_pool)
        unlabeled_pool_tokenized = prepare_dataset(unlabeled_pool)
        test_pool_tokenized = prepare_dataset(test_pool)

        training_args = Seq2SeqTrainingArguments(
            f"outputs/{MODEL_OUTPUT_PATH}",
            # Training parameters
            num_train_epochs=epochs,
            learning_rate=5e-5,
            warmup_steps=0,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=train_grad_acc,
            lr_scheduler_type="constant",
            # Evaluation parameters
            eval_strategy="no",
            eval_accumulation_steps=1,
            per_device_eval_batch_size=32,
            predict_with_generate=True,
            include_inputs_for_metrics=True,
            generation_max_length=400,
            generation_num_beams=1,
            # Logging parameters
            logging_strategy="steps",
            logging_steps=1,
            report_to="none" if wandb_off else "wandb",
            run_name=MODEL_OUTPUT_PATH,
            # Saving parameters
            save_strategy="no",
            save_total_limit=0,
            # Randomization
            seed=seed,
        )
        trainer = Seq2SeqTrainerWithLogits(
            model=model,  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            args=training_args,
            train_dataset=training_pool_tokenized,  # type: ignore
            eval_dataset=unlabeled_pool_tokenized,  # type: ignore
            data_collator=DataCollatorForSeq2Seq(tokenizer),
            processing_class=tokenizer,
            compute_metrics=compute_metrics,  # type: ignore
        )
    elif model_name in ["llama", "gemma"]:
        training_args = TrainingArguments(
            # TRAIN ARGUMENTS
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            optim="adamw_8bit",
            learning_rate=5e-5,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=1.0,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            # EVAL ARGUMENTS
            do_eval=False,
            eval_strategy="no",
            per_device_eval_batch_size=2,
            include_inputs_for_metrics=True,
            # LOG ARGUMENTS
            logging_strategy="steps",
            logging_steps=1,
            report_to="none" if wandb_off else "wandb",
            run_name=MODEL_OUTPUT_PATH,
            # SAVE ARGUMENTS
            save_strategy="no",
            save_total_limit=0,
            # Randomization
            seed=seed,
        )
        trainer = SFTTrainerWithLogits(
            model=model,
            processing_class=tokenizer,
            train_dataset=training_pool,  # type: ignore
            eval_dataset=unlabeled_pool,  # type: ignore
            formatting_func=formatting_func,
            args=training_args,
            preprocess_logits_for_metrics=custom_preprocess_logits,
            compute_metrics=compute_metrics,  # type: ignore
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Train on subset of data
    trainer.train()  # type: ignore

    # Evaluate on candidates
    candidate_results = trainer.evaluate()  # type: ignore

    if ablate_by_confidence:
        unlabeled_df["scores"] = candidate_results["avg_token_log_prob"]
    elif f"eval_{indiv_metric}" in candidate_results.keys():
        unlabeled_df["scores"] = candidate_results[f"eval_{indiv_metric}"]
    else:
        unlabeled_df["scores"] = candidate_results[indiv_metric]

    if get_preds_only:
        unlabeled_df["sources"] = candidate_results["sources"]
        unlabeled_df["labels"] = candidate_results["labels"]
        unlabeled_df["predictions"] = candidate_results["predictions"]
        unlabeled_df.to_csv(f"analysis/{MODEL_OUTPUT_PATH}.csv")
        return

    unlabeled_df = unlabeled_df.sort_values(by="scores").reset_index(drop=True)  # type: ignore

    def prepare_sft_trainer(
        trainer: SFTTrainerWithLogits,
        training_dataset: Dataset,
        testing_dataset: Dataset,
        training_args: TrainingArguments,
        tokenizer: PreTrainedTokenizer,
    ):
        new_trainer = SFTTrainer(
            model=deepcopy(trainer.model),  # type: ignore
            processing_class=tokenizer,
            train_dataset=training_dataset,  # type: ignore
            eval_dataset=testing_dataset,  # type: ignore
            formatting_func=formatting_func,
            args=training_args,
            preprocess_logits_for_metrics=custom_preprocess_logits,
            compute_metrics=compute_metrics,  # type: ignore
        )
        return new_trainer

    def prepare_seq2seq_trainer(
        trainer: Seq2SeqTrainerWithLogits,
        tokenized_training_dataset: Dataset,
        tokenized_testing_dataset: Dataset,
        training_args: Seq2SeqTrainingArguments,
        tokenizer: PreTrainedTokenizer,
    ):
        new_trainer = Seq2SeqTrainer(
            model=deepcopy(trainer.model),  # type: ignore
            args=training_args,
            train_dataset=tokenized_training_dataset,  # type: ignore
            eval_dataset=tokenized_testing_dataset,  # type: ignore
            data_collator=DataCollatorForSeq2Seq(tokenizer),
            processing_class=tokenizer,
            compute_metrics=compute_metrics,  # type: ignore
        )
        return new_trainer

    # Select easy/hard samples
    outputs = {}
    if random_case:
        final_samples = unlabeled_df.sample(n=finetune_size, random_state=seed)  # type: ignore
        final_dataset = Dataset.from_pandas(final_samples)
        final_dataset_tokenized = prepare_dataset(final_dataset)

        # Finetune a copy of the model and evaluate on test set
        if not wandb_off:
            wandb.init(
                project=os.environ["WANDB_PROJECT"],
                name=f"{MODEL_OUTPUT_PATH}_random",
                reinit=True,
            )
            training_args.run_name = f"{MODEL_OUTPUT_PATH}_random"
        if model_name in ["llama", "gemma"]:
            assert isinstance(trainer, SFTTrainerWithLogits)
            new_trainer = prepare_sft_trainer(
                trainer,
                training_dataset=final_dataset,
                testing_dataset=test_pool,
                training_args=training_args,
                tokenizer=tokenizer,
            )
        else:
            assert isinstance(trainer, Seq2SeqTrainerWithLogits)
            assert isinstance(training_args, Seq2SeqTrainingArguments)
            assert test_pool_tokenized is not None
            new_trainer = prepare_seq2seq_trainer(
                trainer,
                tokenized_training_dataset=final_dataset_tokenized,
                tokenized_testing_dataset=test_pool_tokenized,
                training_args=training_args,
                tokenizer=tokenizer,
            )
        new_trainer.train()
        test_results = new_trainer.evaluate()
        outputs["random"] = test_results

    else:
        for percentile in np.arange(0, 1.1, 0.1):
            start_idx = round((len(unlabeled_df) - finetune_size) * percentile)
            end_idx = start_idx + finetune_size
            final_samples = unlabeled_df.iloc[start_idx:end_idx]

            # Train on selected samples
            final_dataset = Dataset.from_pandas(final_samples)
            final_dataset_tokenized = prepare_dataset(final_dataset)

            # Finetune a copy of the model and evaluate on test set
            if not wandb_off:
                wandb.init(
                    project=os.environ["WANDB_PROJECT"],
                    name=f"{MODEL_OUTPUT_PATH}_{percentile}",
                    reinit=True,
                )
                training_args.run_name = f"{MODEL_OUTPUT_PATH}_{percentile}"
            if model_name in ["llama", "gemma"]:
                assert isinstance(trainer, SFTTrainerWithLogits)
                new_trainer = prepare_sft_trainer(
                    trainer,
                    training_dataset=final_dataset,
                    testing_dataset=test_pool,
                    training_args=training_args,
                    tokenizer=tokenizer,
                )
            else:
                assert isinstance(trainer, Seq2SeqTrainerWithLogits)
                assert isinstance(training_args, Seq2SeqTrainingArguments)
                assert test_pool_tokenized is not None
                new_trainer = prepare_seq2seq_trainer(
                    trainer,
                    tokenized_training_dataset=final_dataset_tokenized,
                    tokenized_testing_dataset=test_pool_tokenized,
                    training_args=training_args,
                    tokenizer=tokenizer,
                )
            new_trainer.train()  # type: ignore
            test_results = new_trainer.evaluate()  # type: ignore

            print(percentile, test_results)
            outputs[percentile] = test_results

    json.dump(
        outputs, open(f"results/{MODEL_OUTPUT_PATH}_ablation.json", "w"), indent=4
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--finetune_size", type=int, default=2000, required=False)
    parser.add_argument("--degenerate_case", action="store_true", required=False)
    parser.add_argument("--random_case", action="store_true", required=False)
    parser.add_argument("--suffix", type=str, default="", required=False)
    parser.add_argument("--ablate_by_confidence", action="store_true", required=False)
    parser.add_argument("--prefinetune_size", type=int, default=None, required=False)
    parser.add_argument("--wandb_off", action="store_true", required=False)
    parser.add_argument("--get_preds_only", action="store_true", required=False)
    args_dict = vars(parser.parse_args())

    if args_dict["wandb_off"]:
        os.environ["WANDB_DISABLED"] = "true"
    if args_dict["ablate_by_confidence"]:
        os.environ["WANDB_DISABLED"] = "false"
        os.environ["WANDB_PROJECT"] = "sample_difficulty_ablate_by_confidence"
    else:
        os.environ["WANDB_DISABLED"] = "false"
        os.environ["WANDB_PROJECT"] = "sample_difficulty_ablate_by_difficulty"

    print(args_dict)
    main(**args_dict)
