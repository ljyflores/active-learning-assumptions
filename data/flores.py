import pandas as pd
from datasets import load_dataset # type: ignore
from typing import cast

src_lang = "eng"
tgt_lang = "ilo" # "afr", "deu", "fil", "fra", "hat", "ilo"

ds = load_dataset("openlanguagedata/flores_plus")
df = cast(pd.DataFrame, ds["devtest"].to_pandas()) # type: ignore

df = (
    df.loc[df["iso_639_3"].isin([src_lang, tgt_lang])] # type: ignore
    .pivot(columns="iso_639_3", index=["id", "topic", "domain", "url"], values="text")
    .reset_index()
)

df = df.rename(columns={src_lang: "source", tgt_lang: "target"})
df["source"] = df["source"].apply( # type: ignore
    lambda s: f"Translate this sentence from {src_lang} to {tgt_lang}: {s}" # type: ignore
)
df.to_csv(f"{src_lang}_{tgt_lang}/test.csv", index=False)
