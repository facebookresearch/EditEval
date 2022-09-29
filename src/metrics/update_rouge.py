# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict
import re
import numpy as np
from matplotlib.pyplot import sca
from rouge_score import rouge_scorer, scoring


def extract_additions(source: str, target: str):
    """Simple heuristic for extracting added text from normalized inputs/outputs."""
    normalized_additions = []
    for match in re.finditer(r"[^.]+\.?", target):
        if match.group(0) not in source:
            normalized_additions.append(match.group(0).strip())
    return normalized_additions


def update_rouge(
    sources: List[str], targets: List[str], predictions: List[str], scale: float = 1.0, debug: bool = False
):
    """Measures a variety of different ROUGE scores."""
    # We do not measure ROUGE-L for updates since LCS is likely entirely contained in source.
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"])
    aggregator = scoring.BootstrapAggregator()
    stats: Dict[str, List] = {"_target_diff_len": [], "_prediction_diff_len": []}

    for prediction, source, target in zip(predictions, sources, targets):

        all_scores = {}

        target_additions = extract_additions(source=source, target=target)
        target_additions = " ".join(target_additions)
        prediction_additions = extract_additions(source=source, target=prediction)
        prediction_additions = " ".join(prediction_additions)

        stats["_target_diff_len"].append(len(target_additions))
        stats["_prediction_diff_len"].append(len(prediction_additions))

        addition_scores = scorer.score(
            target=target_additions,
            prediction=prediction_additions,
        )

        if debug:
            print("")
            print(source)
            print("-----")
            print(f"->> Target | {target}")
            print(f"->> Target Diff | {target_additions}")
            print("-----")
            print(f"->> Prediction | {prediction}")
            print(f"->> Prediction Diff | {prediction_additions}")
            print(addition_scores)
            input()

        if target_additions.strip() or prediction_additions.strip():
            all_scores.update({f"update_{k}": v for k, v in addition_scores.items()})
        else:
            all_scores.update({f"update_{k}": scoring.Score(1.0, 1.0, 1.0) for k, _ in addition_scores.items()})

        aggregator.add_scores(all_scores)

    result = aggregator.aggregate()
    return {
        **{key: value.mid.fmeasure * scale for key, value in result.items()},
        **{k: np.mean(v) for k, v in stats.items()},
    }


if __name__ == "__main__":
    print(update_rouge(["Foo Bar Baz"], ["Completely replaced"], ["Completely replaced"]))
