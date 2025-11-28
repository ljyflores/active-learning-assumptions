from datasets import Dataset, DatasetDict  # pyright: ignore[reportMissingTypeStubs]
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast  # type: ignore


def formatting_func_llama(example: dict[str, list[object]]):
    system_prompt = "You are a helpful assistant."
    user_prompt = """### Instruction:
{}

### Input:
{}

### Response:
"""
    return [user_prompt.format(system_prompt, s) for s in example["source"]]


def formatting_func_gemma(example: dict[str, list[object]]):
    prompt = """<bos><start_of_turn>user
{}<end_of_turn>
<start_of_turn>model
"""
    return [prompt.format(s) for s in example["source"]]


def prepare_encode_function(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    input_column_name: str = "input",
    output_column_name: str = "label",
    id_column_name: str | None = None,
    model_name: str | None = None,
):

    def encode(examples: dict[str, list[str]]):
        """This function takes a batch of samples,
        and tokenizes them into IDs for the model."""
        # Tokenize the Findings (the input)
        if model_name == "llama":
            input_strings = formatting_func_llama(examples)
        elif model_name == "gemma":
            input_strings = formatting_func_gemma(examples)
        else:
            input_strings = [str(s) for s in examples[input_column_name]]
        model_inputs = tokenizer(
            input_strings,
            padding=True,
            truncation=True,
            return_tensors="pt",
            padding_side="left" if model_name in ["llama", "gemma"] else "right",
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
    model_name: str | None = None,
):
    encoding_fn = prepare_encode_function(
        tokenizer, input_column_name, output_column_name, id_column_name, model_name
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
