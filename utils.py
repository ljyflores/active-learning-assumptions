import random

from dataclasses import dataclass
from datasets import Dataset, DatasetDict  # type: ignore
from pandas import DataFrame  # type: ignore
from transformers import (  # type: ignore
    DataCollatorForSeq2Seq,  # type: ignore
    PreTrainedTokenizer,  # type: ignore
    PreTrainedTokenizerFast,  # type: ignore
    Seq2SeqTrainer,  # type: ignore
    Seq2SeqTrainingArguments,  # type: ignore
    EvalPrediction,  # type: ignore
    EarlyStoppingCallback,  # type: ignore
    TrainingArguments,  # type: ignore
)
from trl import SFTTrainer  # type: ignore

from typing import Any, cast, Sequence, Literal
from torch import Tensor
from string import punctuation
from src.eval import compute_metrics as eval
from src.preprocess import encode_dataset
from src.loading import load_model

model_mapping: dict[str, str] = {
    "bart": "facebook/bart-base",
    "t5": "google/flan-t5-base",
    "mbart": "facebook/mbart-large-50",
    "llama": "unsloth/Meta-Llama-3.1-8B",
    "gemma": "unsloth/gemma-2-2b-it-bnb-4bit",
}

SHUFFLE_SEEDS = [
    42,
    21,
    38,
    24,
    6,
    72,
    90,
    63,
    11,
    85,
    51,
    96,
    2,
    37,
    68,
    29,
    81,
    15,
    76,
    49,
    57,
    52,
    64,
    9,
    10,
]
MODEL_OUTPUT_PATH = f"./results"
REMOVE_CHARS = punctuation + "\n"


@dataclass
class TrainResult:
    indices: list[int]
    eval_metrics: Sequence[dict[str, Any]] | dict[str, object]
    similarity_to_test: float
    coverage_of_test: float
    vocab_size: int


def compute_jaccard_similarity(s1: str, s2: str):
    s1_words = set(s1.lower().split())
    s2_words = set(s2.lower().split())
    return len(s1_words.intersection(s2_words)) / len(s1_words.union(s2_words))


def compute_percent_of_s2_covered_by_s1(s1: str, s2: str):
    s1_words = set(s1.lower().split())
    s2_words = set(s2.lower().split())
    return len(s1_words.intersection(s2_words)) / len(s2_words)


def remove_punctuation(s: str):
    s = s.translate(str.maketrans("", "", punctuation))
    s = s.replace("“", "")
    return s


def count_vocab(s: str):
    return len(set(s.lower().split()))


def custom_preprocess_logits(logits: Tensor, labels: Tensor):
    # Example: Convert logits to predicted class IDs
    if isinstance(logits, tuple):
        logits = logits[0]  # Handle cases where logits might be a tuple
    return logits.argmax(dim=-1)


def prepare_compute_metrics(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    eval_metrics: list[str],
    response_prefix: str | None = None,
):

    def decode_output_tensor(tensor: Tensor):  # type: ignore
        tensor[tensor == -100] = 0
        return tokenizer.batch_decode(  # type: ignore
            tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def compute_metrics(eval_pred: EvalPrediction):
        label_raw = eval_pred.label_ids  # type: ignore
        source_raw = eval_pred.inputs  # type: ignore
        if isinstance(eval_pred.predictions, tuple):  # type: ignore
            pred_raw = eval_pred.predictions[0]  # type: ignore
            _ = eval_pred.predictions[1]  # type: ignore
        else:
            pred_raw = eval_pred.predictions  # type: ignore
            _ = None
        pred_raw = cast(Tensor, pred_raw)
        label_raw = cast(Tensor, label_raw)
        source_raw = cast(Tensor, source_raw)

        predictions = decode_output_tensor(pred_raw)
        sources = decode_output_tensor(source_raw)
        labels = decode_output_tensor(label_raw)

        if response_prefix:
            predictions = [
                p.split(response_prefix)[-1].strip(REMOVE_CHARS) for p in predictions
            ]
            labels = [l.split(response_prefix)[-1].strip(REMOVE_CHARS) for l in labels]

        labels = [[s] for s in labels]  # labels must be a list of LISTS
        all_metrics = eval(sources, predictions, labels, eval_metrics)
        return all_metrics

    return compute_metrics


def formatting_func_llama(example: dict[str, list[object]]):
    system_prompt = "You are a helpful assistant."
    user_prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}"""
    return [
        user_prompt.format(system_prompt, s, t)
        for (s, t) in zip(example["source"], example["target"])
    ]


def formatting_func_gemma(example: dict[str, list[object]]):
    prompt = """<bos><start_of_turn>user
{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>"""
    return [prompt.format(s, t) for (s, t) in zip(example["source"], example["target"])]


def prepare_trainer(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_name: str,
    seed: int,
    response_prefix: str | None = None,
):
    assert batch_size % 2 == 0
    EVAL_METRICS = ["chrf", "chrf_indiv"]
    METRIC_FOR_BEST_MODEL = "eval_chrf"

    model_init, tokenizer = load_model(model_mapping[model_name])
    model = model_init(seed)  # type: ignore

    compute_metrics_fn = prepare_compute_metrics(
        tokenizer, EVAL_METRICS, response_prefix
    )

    if model_name in ["t5", "mbart", "bart"]:
        train_dataset = encode_dataset(train_dataset, tokenizer, "source", "target")  # type: ignore
        val_dataset = encode_dataset(val_dataset, tokenizer, "source", "target")  # type: ignore
        test_dataset = encode_dataset(test_dataset, tokenizer, "source", "target")  # type: ignore

        training_args = Seq2SeqTrainingArguments(
            f"outputs/{MODEL_OUTPUT_PATH}",
            # Training parameters
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            warmup_steps=0,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=batch_size // 2,
            lr_scheduler_type="constant",
            # Evaluation parameters
            do_eval=True,
            eval_strategy="epoch",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            include_inputs_for_metrics=True,
            generation_max_length=400,
            generation_num_beams=1,
            metric_for_best_model=METRIC_FOR_BEST_MODEL,
            # Logging parameters
            logging_strategy="steps",
            logging_steps=1,
            report_to="none",
            # Saving parameters
            save_strategy="no",
            save_total_limit=0,
            # Randomization
            seed=seed,
        )
        trainer = Seq2SeqTrainer(
            model=model,  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  # type: ignore
            data_collator=DataCollatorForSeq2Seq(tokenizer),
            processing_class=tokenizer,
            compute_metrics=compute_metrics_fn,  # type: ignore
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        return trainer, test_dataset
    elif model_name in ["llama", "gemma"]:
        if model_name == "llama":
            formatting_func_chosen = formatting_func_llama
        else:
            formatting_func_chosen = formatting_func_gemma
        training_args = TrainingArguments(
            f"results/calibration/{MODEL_OUTPUT_PATH}",
            # TRAIN ARGUMENTS
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=batch_size // 2,
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
            do_eval=True,
            eval_strategy="epoch",
            per_device_eval_batch_size=2,
            include_inputs_for_metrics=True,
            metric_for_best_model=METRIC_FOR_BEST_MODEL,
            # LOG ARGUMENTS
            logging_strategy="steps",
            logging_steps=1,
            report_to="wandb",
            run_name=MODEL_OUTPUT_PATH,
            # SAVE ARGUMENTS
            save_strategy="no",
            save_total_limit=0,
            # Randomization
            seed=seed,
        )
        trainer = SFTTrainer(
            model=model,  # type: ignore
            processing_class=tokenizer,  # type: ignore
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            formatting_func=formatting_func_chosen,  # type: ignore
            args=training_args,
            preprocess_logits_for_metrics=custom_preprocess_logits,
            compute_metrics=compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        test_dataset = cast(
            Dataset,
            trainer._prepare_dataset(  # type: ignore
                test_dataset,
                trainer.processing_class,  # type: ignore
                trainer.args,  # type: ignore
                trainer.args.packing,  # type: ignore
                formatting_func_chosen,  # type: ignore
                "test",
            ),
        )
        return trainer, test_dataset
    else:
        raise AssertionError("Wrong model name")


def run_training_on_multiple_samples(
    model_name: str,
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    num_samples: int,
    num_datapoints_per_sample: int | Literal["all"],
    num_shuffles_per_sample: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    test_all_vocab = remove_punctuation(" ".join(test_df["target"].tolist()))
    if num_datapoints_per_sample == "all":
        list_of_sampled_idxs = [list(range(len(train_df))) for _ in range(num_samples)]
    else:
        list_of_sampled_idxs = [
            random.sample(range(len(train_df)), k=num_datapoints_per_sample)
            for _ in range(num_samples)
        ]

    results = list[TrainResult]()
    for sampled_idxs in list_of_sampled_idxs:
        train_sample = train_df.loc[sampled_idxs]  # type: ignore
        train_all_vocab = remove_punctuation(" ".join(train_sample["target"].values))
        metrics = run_training(
            model_name=model_name,
            train_df=train_sample,
            val_df=val_df,
            test_df=test_df,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_shuffles_per_sample=num_shuffles_per_sample,
        )
        result = TrainResult(
            indices=sampled_idxs,
            eval_metrics=metrics,
            similarity_to_test=compute_jaccard_similarity(
                train_all_vocab, test_all_vocab
            ),
            coverage_of_test=compute_percent_of_s2_covered_by_s1(
                train_all_vocab, test_all_vocab
            ),
            vocab_size=count_vocab(train_all_vocab),
        )
        results.append(result)
    return results


def run_training(
    model_name: str,
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_shuffles_per_sample: int = 1,
):
    training_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    if model_name == "llama":
        response_prefix = "### Response:\n"
    elif model_name == "gemma":
        response_prefix = "model\n"
    else:
        response_prefix = None

    metrics_list = list[dict[str, object]]()
    for run_idx in range(num_shuffles_per_sample):
        trainer, test_dataset_processed = prepare_trainer(
            train_dataset=training_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_name=model_name,
            seed=SHUFFLE_SEEDS[run_idx],
            response_prefix=response_prefix,
        )
        trainer.train()  # type: ignore
        metrics = cast(dict[str, object], trainer.evaluate(test_dataset_processed))  # type: ignore
        metrics_list.append(metrics)
    return metrics_list
