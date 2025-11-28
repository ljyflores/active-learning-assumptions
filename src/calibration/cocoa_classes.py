from torch import Tensor


class CocoaMSP:
    def __init__(self):
        pass

    def __str__(self):
        return "CocoaMSP"

    def __call__(
        self,
        token_log_likelihoods: Tensor,  # batch_size x seq_len
        sentence_similarity: Tensor,  # batch_size x num_beams
    ):
        batch_size = token_log_likelihoods.shape[0]
        enriched_metrics = list[float]()  # To store enriched metrics for each sample
        for i in range(batch_size):
            # Compute probabilities (negative log-probs)
            prob = -1.0 * token_log_likelihoods[i].sum()

            # Compute row-wise average similarity, excluding self-similarity
            # Diagonal contains self-similarities
            avg_dissimilarity = float((1 - sentence_similarity[i]).mean())

            enriched_metric = float(prob * avg_dissimilarity)
            enriched_metrics.append(enriched_metric)

        return enriched_metrics


class CocoaPPL:
    def __init__(self):
        pass

    def __str__(self):
        return "CocoaPPL"

    def __call__(
        self,
        token_log_likelihoods: Tensor,  # batch_size x seq_len
        sentence_similarity: Tensor,  # batch_size x num_beams
    ):
        batch_size = token_log_likelihoods.shape[0]
        enriched_ppl = list[float]()  # To store enriched PPL for each sample

        for i in range(batch_size):
            # get PPL for each sample
            ppl = -1 * token_log_likelihoods[i].mean()

            # Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = float((1 - sentence_similarity[i]).mean())

            enriched_value = float(ppl * avg_dissimilarity)
            enriched_ppl.append(enriched_value)

        return enriched_ppl


class CocoaMTE:
    def __init__(
        self,
    ):
        pass

    def __str__(self):
        return "CocoaMTE"

    def __call__(
        self,
        token_entropies: Tensor,  # batch_size x seq_len
        sentence_similarity: Tensor,  # batch_size x num_beams
    ):
        batch_size = token_entropies.shape[0]
        enriched_entropy = list[float]()

        for i in range(batch_size):
            #  Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = float((1 - sentence_similarity[i]).mean())
            entropy = float(token_entropies[i].mean())
            enriched_value = entropy * avg_dissimilarity
            enriched_entropy.append(enriched_value)

        return enriched_entropy
