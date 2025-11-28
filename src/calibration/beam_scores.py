from torch import Tensor
from src.calibration.postprocess import reshape_sequence_probs_by_beam


def compute_beam_score_sum_top_k(scores_per_beam: Tensor):
    # Input: batch_size x num_beams
    assert scores_per_beam.dim() == 2
    scores_by_k_dict = {
        k: [float(x) for x in scores_per_beam[:, : k + 1].sum(dim=-1)]
        for k in range(scores_per_beam.shape[1])
    }
    return scores_by_k_dict


def compute_beam_score_ratios(scores_per_beam: Tensor):
    # Input: batch_size x num_beams
    assert scores_per_beam.dim() == 2
    best_beam_score = scores_per_beam[:, 0]
    scores_by_k_dict = {
        k: [float(x) for x in (best_beam_score - scores_per_beam[:, k]).exp()]
        for k in range(scores_per_beam.shape[1])
    }
    return scores_by_k_dict


def compute_importance_weighted_log_probs(scores_per_beam: Tensor):
    # Input: batch_size x num_beams
    assert scores_per_beam.dim() == 2
    scores_by_k_dict = dict[int, list[float]]()
    for i in range(scores_per_beam.shape[1]):
        top_k_scores = scores_per_beam[:, : i + 1]
        beam_probs = top_k_scores.exp()
        beam_importance_weights = beam_probs / beam_probs.sum(dim=-1).unsqueeze(-1)
        scores = (-1.0 * beam_importance_weights * top_k_scores).sum(dim=-1)
        scores = [float(x) for x in scores]
        scores_by_k_dict[i] = scores
    return scores_by_k_dict


def compute_confidence_beam_scores(
    sequence_scores: Tensor, batch_size: int, num_beams: int
):
    sequence_probs = reshape_sequence_probs_by_beam(
        sequence_scores, batch_size, num_beams  # type: ignore
    )

    # Compute beam score ratios
    scores_beam_score_ratios = compute_beam_score_ratios(sequence_probs)

    # Compute beam score sums
    scores_beam_score_sums = compute_beam_score_sum_top_k(sequence_probs)

    # Compute importance weighted probs
    scores_importance_weighted_log_probs = compute_importance_weighted_log_probs(
        sequence_probs
    )
    return {
        "beam_score_ratios": scores_beam_score_ratios,
        "beam_score_sums": scores_beam_score_sums,
        "beam_score_importance_weighted_sum": scores_importance_weighted_log_probs,
    }
