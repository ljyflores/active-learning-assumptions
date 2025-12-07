import numpy as np
import pandas as pd
from collections import Counter
from string import punctuation
from typing import cast


# HELPERS
def flatten(xss: list[list[object]]):
    return [x for xs in xss for x in xs]


def remove_punctuation(s: str):
    return s.translate(str.maketrans("", "", punctuation))


def generate_delfy_scores(unlabeled_corpus: list[str], labeled_corpus: list[str]):
    lambda_1 = 1.0
    lambda_2 = 1.0

    # Compute C(w|L)
    labeled_vocabs = list(map(lambda s: remove_punctuation(s.lower()).split(), labeled_corpus))  # type: ignore
    labeled_vocabs_counts = dict(Counter(flatten(labeled_vocabs)))  # type: ignore

    # Compute G(w|U)
    corpus_vocabs = list(map(lambda s: remove_punctuation(s.lower()).split(), unlabeled_corpus))  # type: ignore
    corpus_vocabs_counts = Counter(flatten(corpus_vocabs)).most_common()  # type: ignore
    corpus_vocabs_log_counts = {
        vocab: float(np.log(count + 1)) for vocab, count in corpus_vocabs_counts
    }

    # Compute F(w|U)
    sum_of_log_count = sum(corpus_vocabs_log_counts.values())
    corpus_vocabs_log_count_normalized = {
        vocab: count / sum_of_log_count
        for vocab, count in corpus_vocabs_log_counts.items()
    }

    # Collect F(w|U) and decay_1 for each word per sentence
    word_log_counts_normalized_per_sentence = [
        [corpus_vocabs_log_count_normalized[w] for w in sentence]
        for sentence in corpus_vocabs
    ]
    word_decay_1_per_sentence = [
        [
            float(np.exp(-1.0 * lambda_1 * labeled_vocabs_counts.get(w, 0)))
            for w in sentence
        ]
        for sentence in corpus_vocabs
    ]

    # Compute lf(s)
    lf_scores = [
        float(np.mean([f * d for (f, d) in zip(log_counts, decay_1s)]))
        for log_counts, decay_1s in zip(
            word_log_counts_normalized_per_sentence, word_decay_1_per_sentence
        )
    ]
    lf_scores_np = np.array(lf_scores)

    # Compute delfy(s)
    delfy_scores = list[float]()
    for words, word_log_counts, word_decay_1s, lf_score in zip(
        corpus_vocabs,
        word_log_counts_normalized_per_sentence,
        word_decay_1_per_sentence,
        lf_scores,
    ):
        (idxs_with_higher_lf,) = np.where(lf_scores_np > lf_score)
        vocab_counts_of_sentences_with_higher_lf = Counter(
            flatten([corpus_vocabs[idx] for idx in idxs_with_higher_lf])
        )
        word_decay_2s = [
            float(
                np.exp(-1.0 * lambda_2 * vocab_counts_of_sentences_with_higher_lf[word])
            )
            for word in words
        ]
        delfy = float(
            np.mean(
                [
                    log_count * decay_1 * decay_2
                    for log_count, decay_1, decay_2 in zip(
                        word_log_counts, word_decay_1s, word_decay_2s
                    )
                ]
            )
        )
        delfy_scores.append(delfy)
    return delfy_scores


def compute_delfy_scores(task_name: str):
    base_path = "/home/mila/f/floresl/active-learning-assumptions"
    unlabeled_data = pd.read_csv(f"{base_path}/data/{task_name}/train_candidate.csv")  # type: ignore
    unlabeled_data["source_delfy"] = generate_delfy_scores(
        cast(list[str], unlabeled_data["source"]), []
    )
    unlabeled_data["target_delfy"] = generate_delfy_scores(
        cast(list[str], unlabeled_data["target"]), []
    )
    unlabeled_data[["source", "source_delfy", "target", "target_delfy"]].to_csv(
        f"{base_path}/analysis/delfy_{task_name}.csv"
    )
