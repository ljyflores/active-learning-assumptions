from torch import Tensor
from src.calibration.similarity import CrossEncoderSimilarityMatrixCalculator
from src.calibration.cocoa_classes import CocoaMSP, CocoaMTE, CocoaPPL


cocoa_msp = CocoaMSP()
cocoa_mte = CocoaMTE()
cocoa_ppl = CocoaPPL()
similarity_calculator = CrossEncoderSimilarityMatrixCalculator()


def get_cocoa_scores(
    beam_sentences: list[list[str]],
    beam_logits: Tensor,
):
    # Compute similarity scores
    top_texts = [texts[0] for texts in beam_sentences]
    beam_texts = [texts[1:] for texts in beam_sentences]
    scores = Tensor(
        similarity_calculator(top_texts=top_texts, comparison_texts=beam_texts)
    )

    # Get logits, probs, logprobs
    probs = beam_logits[:, 0, :, :].softmax(dim=-1)
    log_probs = (probs + 1e-12).log()
    token_log_probs = log_probs.max(dim=-1).values
    token_entropies = (-1 * probs * log_probs).sum(dim=-1)

    # Compute CoCoA scores
    cocoa_msp_scores = cocoa_msp(
        token_log_likelihoods=token_log_probs, sentence_similarity=scores
    )
    cocoa_ppl_scores = cocoa_ppl(
        token_log_likelihoods=token_log_probs, sentence_similarity=scores
    )
    cocoa_mte_scores = cocoa_mte(
        token_entropies=token_entropies, sentence_similarity=scores
    )

    results = dict[str, list[list[float]] | list[float]]()
    results["cocoa_msp"] = cocoa_msp_scores
    results["cocoa_ppl"] = cocoa_ppl_scores
    results["cocoa_mte"] = cocoa_mte_scores
    results["similarity"] = [
        [float(x) for x in list(scores[i])] for i in range(scores.shape[0])
    ]
    return results
