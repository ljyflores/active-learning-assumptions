import asyncio
from analyses.utils_analysis_by_step_pred import (
    get_vocabs,
    load_prediction_results,
    display_preds_for_idx,
    generate_vocab_lists,
    detect_languages_list,
    save_pickle,
)
from tqdm import tqdm


async def main():
    folder = "eng_fil"
    language_code = "tl"

    data = load_prediction_results(folder)

    _ = await generate_vocab_lists(
        data["train_targets"], data["test_targets"], folder, language_code
    )

    cache = dict[str, tuple[str, float]]()

    for idx in tqdm(range(1012)):

        # Words that were predicted
        pred_vocabs = get_vocabs(
            " ".join(display_preds_for_idx(data["predictions"], idx))
        )

        # Languages of predicted words
        _ = await detect_languages_list(pred_vocabs, cache)

    save_pickle(cache, f"{folder}_words.pkl")


asyncio.run(main())
