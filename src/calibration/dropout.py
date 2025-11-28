import torch
import numpy as np

from itertools import combinations
from torch import Tensor
from transformers import (  # pyright: ignore[reportMissingTypeStubs]
    PreTrainedModel,  # type: ignore
    GenerationConfig,  # type: ignore
    PreTrainedTokenizer,  # type: ignore
)
from typing import Tuple
from typing_extensions import Literal
from torch.nn.functional import kl_div

from src.calibration.eval import calculate_bleu, calculate_meteor
from src.calibration.postprocess import decode_monte_carlo_dropout_sentences


def get_dropout_predictions(
    model: PreTrainedModel,
    item: dict[str, Tensor],
    num_dropout_samples: int = 10,
    max_new_tokens: int = 200,
):
    # Set to train mode, so that dropout is activated during the forward pass
    model.train()

    with torch.no_grad():

        # Expand the batch dimension, which effectively performs `num_models`forward passes
        item.pop("labels")
        batch_size, _ = item["input_ids"].shape

        # Generate num_models copies of each entry
        # This turns the tensor from batch_size x seq_len
        # into (batch_size * num_models) x seq_len
        item["input_ids"] = torch.repeat_interleave(
            item["input_ids"].clone(), num_dropout_samples, dim=0
        ).to(model.device)
        item["attention_mask"] = torch.repeat_interleave(
            item["attention_mask"].clone(), num_dropout_samples, dim=0
        ).to(model.device)

        # Get the logits
        token_scores: Tuple[Tensor] = model.generate(  # type: ignore
            input_ids=item.get("input_ids"),
            generation_config=GenerationConfig(
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            ),
            num_beams=1,
            num_return_sequences=1,
        ).scores  # type: ignore
        logits = torch.stack(token_scores, dim=1)
        logits[logits == -torch.inf] = 1e-12
        probs = logits.softmax(dim=-1)
        _, seq_len, vocab_size = probs.shape
        probs = probs.reshape(batch_size, num_dropout_samples, seq_len, vocab_size)

    model.eval()  # Return the model to eval mode
    return probs


def compute_average_pairwise_similarity(
    texts: list[str], similarity_method: Literal["bleu", "meteor"]
):
    id_pairs = combinations(iterable=list(range(len(texts))), r=2)
    # BLEU Variance: https://arxiv.org/pdf/2006.08344
    if similarity_method == "bleu":
        bleu_scores = map(
            lambda pair: calculate_bleu([texts[pair[0]]], [[texts[pair[1]]]]),
            id_pairs,
        )
        score = float(np.mean((1 - np.array(list(bleu_scores))) ** 2))
    # Dropout Based Lexical Similarity: https://arxiv.org/pdf/2211.14880
    elif similarity_method == "meteor":
        meteor_scores = map(
            lambda pair: calculate_meteor([texts[pair[0]]], [[texts[pair[1]]]]),
            id_pairs,
        )
        score = float(np.mean(list(meteor_scores)))
    else:
        raise ValueError("Similarity method is not recognized")
    return score


def compute_mean_token_entropy(token_probs_tensor: Tensor):
    stability_constant = 1e-20
    # Input: batch_size x num_beams x vocab_size
    assert token_probs_tensor.dim() == 3
    entropy = (
        -1.0 * token_probs_tensor * (token_probs_tensor + stability_constant).log()
    )
    return [float(x) for x in entropy.sum(dim=2).mean(dim=1).detach()]


def compute_monte_carlo_mean_token_entropy(dropout_token_probs_tensor: Tensor):
    # Input: batch_size x num_dropout x seq_len x vocab_size
    return [
        float(np.mean(compute_mean_token_entropy(dropout_token_probs_tensor[i])))
        for i in range(dropout_token_probs_tensor.shape[0])
    ]


def compute_disagreement(dropout_token_probs: Tensor):
    assert dropout_token_probs.dim() == 4
    # actual_token_probs: batch_size x seq_len x vocab_size
    # dropout_token_probs: batch_size x num_dropout x seq_len x vocab_size
    _, num_dropout, _, _ = dropout_token_probs.shape
    mean_dropout_token_probs = dropout_token_probs.mean(dim=1)
    mean_dropout_token_probs = mean_dropout_token_probs.unsqueeze(dim=1).repeat(
        1, num_dropout, 1, 1
    )
    kl_div_tensor = kl_div(
        mean_dropout_token_probs.log(), dropout_token_probs, reduction="none"
    )
    disagreement_scores = kl_div_tensor.sum(dim=-1).mean(dim=-1).sum(dim=-1)
    return [float(x) for x in disagreement_scores]


def compute_bald(logits: torch.Tensor):
    stability_constant = 1e-20
    probs = torch.tensor(logits).softmax(dim=-1)
    log_probs = torch.log(probs + stability_constant)

    assert log_probs.dim() == 4

    # Get entropy (shape: batch_size x num_beams x seq_len x vocab_size)
    dropout_token_entropy = log_probs * log_probs.exp()
    # For each entry, sum the total entropy (shape: batch_size x num_beams)
    dropout_total_entropy_for_one_forward_pass = dropout_token_entropy.sum(dim=3).sum(
        dim=2
    )
    # Get the average total entropy per entry (shape: batch_size)
    dropout_average_entropy = dropout_total_entropy_for_one_forward_pass.mean(dim=1)

    # Average probs
    avg_probs = log_probs.exp().mean(dim=1)
    avg_token_entropy = avg_probs * avg_probs.log()
    avg_token_entropy[avg_token_entropy.isnan()] = 0.0
    total_entropy = avg_token_entropy.sum(dim=2).sum(dim=1)

    bald_score = (-1.0 * total_entropy) + dropout_average_entropy
    bald_score = [float(x) for x in bald_score.detach()]
    return bald_score


def get_dropout_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    item: dict[str, Tensor],
):
    model.eval()

    dropout_probs = get_dropout_predictions(
        model,
        item,
        num_dropout_samples=3,
    )

    # Post-process outputs
    dropout_sentences = decode_monte_carlo_dropout_sentences(dropout_probs, tokenizer)

    # Compute BLEU and Meteor scores
    scores_dropout_bleu_variance = [
        compute_average_pairwise_similarity(sents, "bleu")
        for sents in dropout_sentences
    ]

    scores_dropout_meteor_score = [
        compute_average_pairwise_similarity(sents, "meteor")
        for sents in dropout_sentences
    ]

    # Compute dropout disagreement
    scores_dropout_disagreement = compute_disagreement(dropout_probs)

    scores_dropout_entropy = compute_monte_carlo_mean_token_entropy(dropout_probs)

    # Compute BALD
    scores_bald = compute_bald(dropout_probs.log())

    result = dict[str, list[float]]()
    result["dropout_bleu_variance"] = scores_dropout_bleu_variance
    result["dropout_meteor_score"] = scores_dropout_meteor_score
    result["dropout_entropy"] = scores_dropout_entropy
    result["dropout_disagreement"] = scores_dropout_disagreement
    result["dropout_bald"] = scores_bald
    return result
