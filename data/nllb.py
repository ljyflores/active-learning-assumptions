from datasets import Dataset, IterableDataset, load_dataset  # type: ignore
from functools import partial
from pandas import DataFrame
from typing import cast


def gen_from_iterable_dataset(iterable_ds: IterableDataset):  # type: ignore
    yield from iterable_ds


src_lang = ("eng_Latn", "eng")
tgt_lang = (
    "tgl_Latn",
    "fil",
)  # ("afr_Latn", "afr"), ("deu_Latn", "deu"), ("fra_Latn", "fra"), ("ilo_Latn", "ilo"), ("hat_Latn", "hat"), ("tgl_Latn", "fil")

try:
    dataset = load_dataset(
        "allenai/nllb", f"{src_lang[0]}-{tgt_lang[0]}", streaming=True, split="train"
    )
except ValueError:
    dataset = load_dataset(
        "allenai/nllb", f"{tgt_lang[0]}-{src_lang[0]}", streaming=True, split="train"
    )
dataset = cast(IterableDataset, dataset)

shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10000000)  # type: ignore
shuffled_dataset = shuffled_dataset.take(20000)  # Used to be 10100

ds = Dataset.from_generator(partial(gen_from_iterable_dataset, shuffled_dataset), features=shuffled_dataset.features)  # type: ignore
df = cast(DataFrame, ds.to_pandas())  # type: ignore

df["source"] = df["translation"].apply(lambda dictionary: f"Translate this sentence from {src_lang[1]} to {tgt_lang[1]}: {dictionary[src_lang[0]]}")  # type: ignore
df["target"] = df["translation"].apply(lambda dictionary: dictionary[tgt_lang[0]])  # type: ignore
df = df.drop(columns=["translation"])

df_labeled = df.iloc[:10000]  # Used to be 100
df_candidate = df.iloc[10000:]  # Used to be 100

df_labeled.to_csv(f"{src_lang[1]}_{tgt_lang[1]}_scale/train_labeled.csv", index=False)
df_candidate.to_csv(
    f"{src_lang[1]}_{tgt_lang[1]}_scale/train_candidate.csv", index=False
)
