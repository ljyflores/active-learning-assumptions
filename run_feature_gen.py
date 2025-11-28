import argparse
import os
import pandas as pd
import torch

from datasets import Dataset  # type: ignore
from transformers import GenerationConfig  # type: ignore
from transformers.generation.utils import GenerationMixin
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocess import encode_dataset
from src.loading import load_data, load_model
from src.calibration.postprocess import (
    reshape_token_logits_by_beam,
    decode_beam_search_sentences,
)
from src.calibration.beam_scores import compute_confidence_beam_scores
from src.calibration.dropout import get_dropout_scores
from src.calibration.cocoa import get_cocoa_scores


def patched_reorder_cache(self, past, beam_idx):  # type: ignore
    """
    Reorders past_key_values for beam search.
    Supports nested tuples returned by PEFT / LoRA models.
    """
    if isinstance(past, tuple):
        # recursively index each tensor
        return tuple(
            (
                patched_reorder_cache(self, p, beam_idx)  # type: ignore
                if isinstance(p, tuple)
                else p.index_select(0, beam_idx.to(p.device))  # type: ignore
            )
            for p in past  # type: ignore
        )  # type: ignore
    elif hasattr(past, "reorder_cache"):  # type: ignore
        return past.reorder_cache(beam_idx)  # type: ignore
    else:
        return past  # type: ignore


# Monkey-patch the GenerationMixin._reorder_cache method globally
GenerationMixin._reorder_cache = patched_reorder_cache  # type: ignore

model_mapping: dict[str, str] = {
    "bart": "facebook/bart-base",
    "t5": "google/flan-t5-base",
    "mbart": "facebook/mbart-large-50",
    "llama": "unsloth/Meta-Llama-3.1-8B",
    "gemma": "unsloth/gemma-2-2b-it-bnb-4bit",
}


def compute_confidence_prob_scores(logits: torch.Tensor):
    assert logits.dim() == 3
    stability_constant = 1e-20
    probs = logits.softmax(dim=-1)
    log_probs = torch.log(probs + stability_constant)
    average_token_log_prob = log_probs.max(dim=-1).values.mean(dim=-1)
    average_token_entropy = (-1.0 * probs * log_probs).sum(dim=-1).mean(dim=-1)
    scores = dict[str, float | list[float]]()
    scores["avg_token_log_prob"] = [float(x) for x in average_token_log_prob.cpu()]
    scores["avg_token_entropy"] = [float(x) for x in average_token_entropy.cpu()]
    return scores


def append_hidden_state(hidden_state_path: str, new_hidden_state: torch.Tensor):
    if os.path.exists(hidden_state_path):
        existing_hidden_states = torch.load(hidden_state_path)  # type: ignore
        combined_hidden_states = torch.vstack(
            [existing_hidden_states, new_hidden_state]
        )
        torch.save(combined_hidden_states, hidden_state_path)  # type: ignore
    else:
        torch.save(new_hidden_state, hidden_state_path)  # type: ignore


def main(
    model_name: str,
    dataset: str,
):

    output_folder = f"outputs/{dataset.split('/')[-1]}"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    batch_size: int = 4
    num_beams: int = 3

    device = torch.device("cuda")
    data = load_data(dataset, model_mapping[model_name])

    training_dataset = Dataset.from_pandas(data["train"])

    model_init, tokenizer = load_model(model_mapping[model_name])  # type: ignore
    model = model_init(42)
    model = model.to(device)  # type: ignore

    training_dataset = encode_dataset(training_dataset, tokenizer, input_column_name="source", output_column_name="target", model_name=model_name)  # type: ignore
    dataloader = DataLoader(training_dataset, batch_size=batch_size)  # type: ignore

    beam_search_config = GenerationConfig(
        max_new_tokens=400,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True,
    )

    metrics_list = dict[str, list[object]]()

    for batch in tqdm(dataloader):  # type: ignore
        input_ids = batch["input_ids"].to(device)
        if input_ids.shape[0] != batch_size:
            batch_size = input_ids.shape[0]
        output = model.generate(  # type: ignore
            input_ids=input_ids,
            generation_config=beam_search_config,
            pad_token_id=int(tokenizer.pad_token_id),  # type: ignore
            eos_token_id=int(tokenizer.eos_token_id),  # type: ignore
        )

        sequences, _ = decode_beam_search_sentences(
            output.sequences, tokenizer, batch_size, num_beams
        )
        logits = reshape_token_logits_by_beam(output.scores, batch_size, num_beams)

        if hasattr(output, "decoder_hidden_states"):
            last_decoder_hidden_state = torch.stack(
                [hs[-1] for hs in output.decoder_hidden_states], axis=1
            )
            last_decoder_hidden_state = last_decoder_hidden_state.mean(axis=1).mean(
                axis=1
            )
            last_decoder_hidden_state = last_decoder_hidden_state.reshape(
                batch_size, num_beams, -1
            )
            append_hidden_state(
                f"{output_folder}/{model_name}_baseline_hidden_states.pt",
                last_decoder_hidden_state[:, 0, :],
            )
            del last_decoder_hidden_state

        elif hasattr(output, "hidden_states"):
            hidden_state = output.hidden_states[0][-1]
            hidden_state = hidden_state.mean(axis=1).reshape(batch_size, num_beams, -1)
            append_hidden_state(
                f"{output_folder}/{model_name}_baseline_hidden_states.pt",
                hidden_state[:, 0, :],
            )
            del hidden_state

        batch_metrics = dict[str, object]()

        prob_scores = compute_confidence_prob_scores(logits[:, 0, :, :])
        batch_metrics.update(prob_scores)

        beam_scores = compute_confidence_beam_scores(
            output.sequences_scores, batch_size, num_beams
        )
        batch_metrics.update(beam_scores)

        dropout_score = get_dropout_scores(model, tokenizer, batch)
        batch_metrics.update(dropout_score)

        cocoa_scores = get_cocoa_scores(sequences, logits)
        batch_metrics.update(cocoa_scores)

        for key in batch_metrics:
            if isinstance(batch_metrics[key], list):
                if key not in metrics_list.keys():
                    metrics_list[key] = []
                metrics_list[key].extend(batch_metrics[key])
            elif isinstance(batch_metrics[key], dict):
                for subkey in batch_metrics[key].keys():
                    if f"{key}_{subkey}" not in metrics_list.keys():
                        metrics_list[f"{key}_{subkey}"] = []
                    metrics_list[f"{key}_{subkey}"].extend(batch_metrics[key][subkey])

    pd.DataFrame(metrics_list).to_csv(
        f"{output_folder}/{model_name}_baseline_metrics.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args_dict = vars(parser.parse_args())
    main(**args_dict)
