import json
import pandas as pd

from analyses.utils_analysis_by_coverage import greedy_select_idxs_by_covering

folder = "eng_deu"
df_train = pd.read_csv(f"{folder}/train_candidate.csv") # type: ignore
df_test = pd.read_csv(f"{folder}/test.csv") # type: ignore

train_targets = [str(s) for s in df_train["target"].apply(lambda s: remove_punctuation(s.lower()))] # type: ignore
test_targets = [str(s) for s in df_test["target"].apply(lambda s: remove_punctuation(s.lower()))] # type: ignore

idxs_to_use = list[list[int]]()
for coverage_level in [
    0.000, 0.025, 0.025, 
    0.025, 0.050, 0.050, 
    0.050, 0.075, 0.075, 
    0.075, 0.100, 0.100, 
    0.200, 0.200, 0.300, 
    0.300, 0.400, 0.400, 
    0.700, 0.700, 1.000
    ]:
    idxs_and_sentences = greedy_select_idxs_by_covering(
        train_targets,
        test_targets,
        coverage_level, 
        1000
        )
    idxs = [item[0] for item in idxs_and_sentences]
    idxs_to_use.append(idxs)
    with open(f"{folder}/coverage_idxs.json", "w") as fout:
        json.dump(idxs_to_use, fout, indent=4)
