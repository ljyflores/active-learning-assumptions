from datasets import Dataset, DatasetDict  # pyright: ignore[reportMissingTypeStubs]
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast  # type: ignore


def prepare_encode_function(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    input_column_name: str = "input",
    output_column_name: str = "label",
    id_column_name: str | None = None,
):

    def encode(examples: dict[str, list[str]]):
        """This function takes a batch of samples,
        and tokenizes them into IDs for the model."""
        # Tokenize the Findings (the input)
        model_inputs = tokenizer(
            [str(s) for s in examples[input_column_name]],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize the Impressions (the output)
        if hasattr(tokenizer, "as_target_tokenizer"):
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    [str(s) for s in examples[output_column_name]],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
        else:
            labels = tokenizer(
                [str(s) for s in examples[output_column_name]],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        # Set the label as the token ids (i.e. the vocab IDs) of the findings
        model_inputs["labels"] = labels["input_ids"]

        if id_column_name is not None:
            model_inputs["id"] = examples[id_column_name]

        return model_inputs

    return encode


def encode_dataset(
    dataset: Dataset | DatasetDict,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    input_column_name: str,
    output_column_name: str,
    id_column_name: str | None = None,
):
    encoding_fn = prepare_encode_function(
        tokenizer, input_column_name, output_column_name, id_column_name
    )
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    if id_column_name is not None:
        columns_to_keep.append("id")
    if isinstance(dataset, Dataset):
        columns_to_remove = list(
            set(dataset.column_names).difference(set(columns_to_keep))
        )
        prepared_dataset = dataset.map(  # pyright: ignore[reportUnknownMemberType]
            encoding_fn, batched=True, remove_columns=columns_to_remove
        )
        prepared_dataset.set_format(  # pyright: ignore[reportUnknownMemberType]
            type="torch", columns=columns_to_keep
        )
        return prepared_dataset
    else:
        for key in dataset.keys():
            columns_to_remove = list(
                set(dataset[key].column_names).difference(set(columns_to_keep))
            )
            dataset[key] = dataset[key].map(  # pyright: ignore[reportUnknownMemberType]
                encoding_fn, batched=True, remove_columns=columns_to_remove
            )
            dataset[key].set_format(  # pyright: ignore[reportUnknownMemberType]
                type="torch", columns=columns_to_keep
            )
        return dataset
