import numpy as np
import torch

import itertools
from typing import Dict, List, Tuple
from tqdm import tqdm

from sentence_transformers import CrossEncoder


class CrossEncoderSimilarityMatrixCalculator:
    """
    Calculates the cross-encoder similarity between greedy sequence and sampled sequences.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return (
            [
                "greedy_sentence_similarity_forward",
                "greedy_sentence_similarity_backward",
                "greedy_sentence_similarity",
            ],
            ["input_texts", "sample_texts", "greedy_texts"],
        )

    def __init__(
        self,
        batch_size: int = 10,
        cross_encoder_name: str = "cross-encoder/stsb-roberta-large",
    ):
        self.crossencoder_setup = False
        self.batch_size = batch_size
        self.cross_encoder_name = cross_encoder_name
        self.crossencoder = CrossEncoder(
            self.cross_encoder_name, device=torch.device("cuda")
        )

    def __call__(
        self,
        top_texts: List[str] | None,
        comparison_texts: List[List[str]],
    ) -> Dict[str, np.ndarray]:

        batch_pairs = list[list[tuple[str, str]]]()
        batch_invs = list[object]()

        for top_text, comparison_text in zip(
            top_texts or comparison_texts, comparison_texts
        ):
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            unique_texts, inv = np.unique(comparison_text, return_inverse=True)
            if isinstance(top_text, str):
                batch_pairs.append(list(itertools.product([top_text], unique_texts)))
            else:
                batch_pairs.append(list(itertools.product(unique_texts, unique_texts)))
            batch_invs.append(inv)

        sim_arrays_f = []
        sim_arrays_b = []
        sim_arrays = []
        for i, pairs in tqdm(enumerate(batch_pairs)):
            pairs_b = [(b, a) for a, b in pairs]
            sim_scores_f = self.crossencoder.predict(pairs, batch_size=self.batch_size)
            sim_scores_b = self.crossencoder.predict(
                pairs_b, batch_size=self.batch_size
            )

            inv = batch_invs[i]

            sim_arrays_f.append(sim_scores_f[inv])
            sim_arrays_b.append(sim_scores_b[inv])
            sim_arrays.append((sim_scores_f[inv] + sim_scores_b[inv]) / 2)

        sim_arrays_f = np.stack(sim_arrays_f)
        sim_arrays_b = np.stack(sim_arrays_b)
        sim_arrays = np.stack(sim_arrays)

        return sim_arrays
