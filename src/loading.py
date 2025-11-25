import json
import pandas as pd
import torch

from transformers import (  # type: ignore
    PreTrainedTokenizerFast,  # type: ignore
    MBart50TokenizerFast,  # type: ignore
    PreTrainedModel,  # type: ignore
    AutoModelForSeq2SeqLM,  # type: ignore
    AutoModelForSequenceClassification,  # type: ignore
    AutoTokenizer,  # type: ignore
    MBartConfig,  # type: ignore
    set_seed,  # type: ignore
)
from unsloth import FastLanguageModel  # type: ignore
from typing import cast


mbart_dict: dict[str, tuple[str, str, list[str] | None]] = {
    "data/eng_afr": ("en_XX", "af_ZA", None),
    "data/eng_afr_scale": ("en_XX", "af_ZA", None),
    "data/eng_deu": ("en_XX", "de_DE", None),
    "data/eng_deu_scale": ("en_XX", "de_DE", None),
    "data/eng_testing": ("en_XX", "af_ZA", None),
    "data/eng_fil": ("en_XX", "tl_XX", None),
    "data/eng_fil_scale": ("en_XX", "tl_XX", None),
    "data/eng_hat": ("en_XX", "ht_XX", ["ht_XX"]),
}


def load_model(model_path: str):
    using_llama = ("llama" in model_path.lower()) or ("gemma" in model_path.lower())
    if using_llama:
        llama_model, tokenizer = FastLanguageModel.from_pretrained(  # type: ignore
            model_name=model_path,
            # max_seq_length=400,
            load_in_4bit=True,
            dtype=torch.float16,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore
        config = None
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)
    tokenizer_vocab_size = len(tokenizer)  # type: ignore

    def model_init(seed: int = 42):  # type: ignore
        print(f"Using seed: {seed}")
        set_seed(seed)
        if using_llama and llama_model is not None:
            model = FastLanguageModel.get_peft_model(  # type: ignore
                llama_model,
                r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=16,
                lora_dropout=0,  # Supports any, but = 0 is optimized
                bias="none",  # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
                random_state=seed,
                use_rslora=False,  # We support rank stabilized LoRA
                loftq_config=None,  # And LoftQ
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config)  # type: ignore
            model = cast(PreTrainedModel, model)
            if model.config.vocab_size < tokenizer_vocab_size:  # type: ignore
                model.resize_token_embeddings(tokenizer_vocab_size)
        return model  # type: ignore

    return model_init, tokenizer  # type: ignore


def load_data(
    dataset: str,
    model_path: str,
):
    train_df = pd.read_csv(f"{dataset}/train_candidate.csv")  # type: ignore
    val_df = pd.read_csv(f"{dataset}/train_labeled.csv").head(50)  # type: ignore
    test_df = pd.read_csv(f"{dataset}/test_sample.csv")  # type: ignore
    test_df = test_df.sort_values(  # type: ignore
        by="source", key=lambda s: s.str.len(), ascending=False
    ).reset_index(drop=True)

    if model_path == "facebook/mbart-large-50":

        def remove_instruction(s: str):
            return s.split(":", 1)[1]

        train_df["source"] = train_df["source"].apply(remove_instruction)  # type: ignore
        val_df["source"] = val_df["source"].apply(remove_instruction)  # type: ignore
        test_df["source"] = (
            test_df["source"].astype(str).apply(remove_instruction)  # type: ignore
        )
    return {"train": train_df, "val": val_df, "test": test_df}


def load_json(path: str):
    file = open(path, "rb")
    result = json.load(file)
    file.close()
    return result
