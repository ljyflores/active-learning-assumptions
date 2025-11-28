import json
import os
import pandas as pd
import pickle

from collections import Counter
from dataclasses import dataclass
from googletrans import Translator  # type: ignore
from string import punctuation
from tqdm import tqdm
from typing import cast, Iterable

from utils_analysis_by_order import flatten

ROOT = "~/active-learning-for-nlg"


@dataclass
class Prediction:
    predicted_words: set[str]
    expected_words: set[str]
    expected_but_not_predicted: set[str]
    unexpected_but_predicted: set[str]
    detected_languages: list[tuple[str, str, float]]


def remove_punctuation(s: str):
    return s.translate(str.maketrans("", "", punctuation))


def load_json(path: str):
    file = open(path, "rb")
    result = json.load(file)
    file.close()
    return result


def save_pickle(obj: object, path: str):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    with open(path, "rb") as handle:
        b = pickle.load(handle)
        return b


async def generate_vocab_lists(
    train_targets: list[str],
    test_targets: list[str],
    folder: str,
    language_code: str,
):
    train_vocab_path = f"assets/{folder}_train_vocabs.pkl"
    test_vocab_path = f"assets/{folder}_test_vocabs.pkl"

    if os.path.exists(train_vocab_path):
        train_vocabs = load_pickle(train_vocab_path)
        train_vocabs = cast(set[str], train_vocabs)
    else:
        train_vocabs = get_vocabs(" ".join(train_targets))
        train_vocabs = await filter_to_lang(train_vocabs, language_code)
        save_pickle(train_vocabs, train_vocab_path)

    if os.path.exists(test_vocab_path):
        test_vocabs = load_pickle(test_vocab_path)
        test_vocabs = cast(set[str], test_vocabs)
    else:
        test_vocabs = get_vocabs(" ".join(test_targets))
        test_vocabs = await filter_to_lang(test_vocabs, language_code)
        save_pickle(test_vocabs, test_vocab_path)

    return train_vocabs, test_vocabs


def display_preds_for_idx(predictions: list[list[str]], idx: int):
    return list(map(lambda lst: lst[idx], predictions))


async def filter_to_lang(candidates: Iterable[str], lang: str):
    valid_candidates = set[str]()
    for candidate in tqdm(candidates):
        lang_pred, _ = await detect_languages(candidate)
        if lang_pred == lang:
            valid_candidates.add(candidate)
    return valid_candidates


async def detect_languages_list(
    candidates: Iterable[str], cache: dict[str, tuple[str, float]]
):
    languages_detected = list[tuple[str, str, float]]()
    for v in candidates:
        if v in cache:
            detected_lang, detected_conf = cache[v]
        # else:
        #     detected_lang, detected_conf = await detect_languages(v)
        #     cache[v] = (detected_lang, detected_conf)
        else:
            detected_lang, detected_conf = "tl", 0.00
        languages_detected.append((v, detected_lang, detected_conf))
    return languages_detected


def get_vocabs(s: str):
    return set(remove_punctuation(s).lower().split())


def compute_number_common_words(s1: str, s2: str):
    return len(set(s1.lower().split()).intersection(set(s2.lower().split())))


def load_prediction_results(
    folder: str,
    filename: str = "sampling_results_pred_by_step_bs_1_3eps.json",
    run: int = 0,
):
    df_train = pd.read_csv(f"{ROOT}/data/{folder}/train_candidate.csv")  # type: ignore
    df_test = pd.read_csv(f"{ROOT}/data/{folder}/test.csv")  # type: ignore
    results = load_json(f"../outputs/analyses/{folder}/100/{filename}")[run]
    df_train = df_train.loc[[int(i) for i in results["indices"]]]

    preds = [
        [str(s) for s in result["predictions"]] for result in results["eval_metrics"]
    ]

    test_sources = [str(s).split(":", 1)[1].strip() for s in df_test["source"]]  # type: ignore
    test_targets = [str(s) for s in df_test["target"]]  # type: ignore

    train_sources = [str(x).split(":", 1)[1].strip() for x in df_train["source"]]  # type: ignore
    train_targets = [str(x) for x in df_train["target"]]  # type: ignore

    output: dict[str, list[str] | list[list[str]]] = {
        "train_sources": train_sources,
        "train_targets": train_targets,
        "test_sources": test_sources,
        "test_targets": test_targets,
        "predictions": preds,
    }
    return output


async def detect_languages(text: str):
    async with Translator() as translator:
        result = await translator.detect(text)
        return result.lang, result.confidence


async def extract_predictions(
    predictions: dict[str, object],
    train_vocabs: list[str],
    test_vocabs: list[str],
    cache: dict[str, tuple[str, float]],
):
    outputs = list[Prediction]()
    for idx in range(1012):

        # Target vocabs
        target_vocabs = get_vocabs(predictions["test_targets"][idx])  # type: ignore
        target_filipino_vocabs = target_vocabs.intersection(test_vocabs)

        # Words that were predicted
        pred_vocabs = get_vocabs(
            " ".join(display_preds_for_idx(predictions["predictions"], idx))  # type: ignore
        )

        # Languages of predicted words
        pred_detected_languages = await detect_languages_list(pred_vocabs, cache)

        # Filipino words in the target label, that should be able to be predicted because it's in the training data
        id_test_vocabs = target_filipino_vocabs.intersection(train_vocabs)

        # Filipino words in the target label, that should not be able to be predicted because it's not in the training data
        ood_test_vocabs = target_filipino_vocabs.difference(train_vocabs)

        # Words that the model was expected to get, but did not
        id_failed_words = id_test_vocabs.difference(pred_vocabs)

        # Words that the model was not expected to get, but did
        ood_correct_words = pred_vocabs.intersection(ood_test_vocabs)

        foreign_languages = list(
            filter(lambda tup: tup[1] not in ["en", "tl"], pred_detected_languages)
        )
        output = Prediction(
            predicted_words=pred_vocabs,
            expected_words=id_test_vocabs,
            expected_but_not_predicted=id_failed_words,
            unexpected_but_predicted=ood_correct_words,
            detected_languages=foreign_languages,
        )
        outputs.append(output)
    return outputs


def generate_vocabs_any_step(predictions: list[list[str]]):
    predicted_vocabs = [
        list(
            set(
                remove_punctuation(" ".join(display_preds_for_idx(predictions, idx)))
                .lower()
                .split()
            )
        )
        for idx in range(1012)
    ]
    return Counter(flatten(predicted_vocabs)).most_common()  # type: ignore


def generate_vocabs_final_step(predictions: list[list[str]]):
    predicted_vocabs = [
        list(
            set(
                remove_punctuation(display_preds_for_idx(predictions, idx)[-1])
                .lower()
                .split()
            )
        )
        for idx in range(1012)
    ]
    return Counter(flatten(predicted_vocabs)).most_common()  # type: ignore


def find_idxs_where_present(sentences: list[str], word: str):
    return [idx for (idx, s) in enumerate(sentences) if word in s.lower()]


def count_num_sents_with_word(sentences: list[str], word: str):
    return sum([word in remove_punctuation(s) for s in sentences])
