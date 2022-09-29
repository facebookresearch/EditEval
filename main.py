# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from src.dataset import EditDataset
import json
import src.utils
from typing import Dict, List
import os
import random
import pprint


def load_prediction_file(filename: str) -> List[Dict]:
    data = []
    print(filename)
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


def download_dataset(dataset_name: str, directory: str = None) -> EditDataset:
    edit_data = EditDataset(dataset_name, output_dir=directory)
    return edit_data


def output_inputs_to_file(dataset_name: str, output_dir: str, splits: List[str] = ["test"]) -> None:
    """Convert a dataset that uses no references into a format for prompting autoregressive LMs (GPT-3, T0, ...)."""

    dataset = EditDataset(dataset_name)
    inputs = dataset.get_inputs(splits=splits, transpose=True)

    print(f"There are {len(inputs)} examples")
    output_file = f"{output_dir}{dataset_name}_input.jsonl"
    print(f"Writing {len(inputs)} examples to the file {output_file}")

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w", encoding="utf8") as fh:
        for example in inputs:
            record = {
                "id": example["id"],
                "input": example["input"],
                "retrieved_documents": example["retrieved_documents"],
                "title": example["title"],
                "task_type": example["tasks"],
            }
            if record["task_type"] == "info_addition":
                record["task_type"] = "updating"

            if "meta" in example:
                record["meta"] = example["meta"]
            else:
                record["meta"] = []
            fh.write(json.dumps(record) + "\n")


def sample_dataset(
    dataset: str = None,
    edit_data: EditDataset = None,
    n: int = 5,
    index: int = None,
    features: List[str] = src.utils.CORE_FEATURES,
) -> None:
    if edit_data is None and dataset is None:
        raise ValueError("Needs a dataset name or a the EditDataset object.")
    if edit_data is None:
        edit_data = EditDataset(dataset)
    split_data = src.utils.transpose_dict(edit_data.get_features(splits=["dev"], features=features))
    if index is None:
        index = random.randint(0, len(split_data) - n - 1)
    sampled_data = split_data[index : index + n]
    pprint.pprint(sampled_data)


def get_dataset_dimensions(dataset: str, edit_data: EditDataset = None, splits=["train", "dev", "test"]) -> None:
    if edit_data is None:
        edit_data = EditDataset(dataset)
    lengths = [str(len(edit_data.dataset[split])) for split in splits]
    print(f"Dataset {dataset}: Splits {', '.join(splits)} have lengths {', '.join(lengths)}, respectively.")


def evaluate_predictions(
    metrics: List[str],
    dataset_name: str,
    predictions: List[Dict],
    edit_data: EditDataset = None,
    normalize: bool = False,
    report_type: str = "markdown",
) -> None:
    if edit_data is None:
        edit_data = EditDataset(dataset_name)
    ids = [e["id"] for e in predictions]
    preds = [e["output"] for e in predictions]
    edit_data.evaluate(
        pred_dict={"id": ids, "pred": preds},
        print_report=report_type,
        metrics=metrics,
        normalize=normalize,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help=f"Dataset name from the set {src.utils.ALL_DATASETS}. Use 'all' for all datasets.",
    )
    parser.add_argument(
        "--download_all",
        action="store_true",
        help="Download all datasets: jfleg, iterater_fluency, iterater_clarity, iterater_coherence, stsb_multi_mt, turk, asset, wnc, fruit, wafer_insert.",
    )
    parser.add_argument(
        "--write_to_jsonl", type=str, help="Writes the dataset and fields to a jsonl file at the specified directory."
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        help="Path to prediction file. This should be jsonl file where each line has fields named 'id' and 'output' for the id and predictions, respectively.",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="Path to where the datasets should be saved or are already saved. Defaults to output directories specified in ./configs/dataset_paths.json",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=["rouge", "update_rouge", "em", "em_diff", "sari", "gleu"],
        help=f"Metrics to report from the set {src.utils.ALL_METRICS}.",
        nargs="+",
    )
    parser.add_argument("--sample", type=int, default=0, help="Sample some input-output pairs")
    parser.add_argument(
        "--no_normalization", action="store_true", help="Do not perform normalization before evaluation"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=["test", "dev"],
        help=f"Splits to evaluate on. Default is just the test split",
        nargs="+",
    )
    args = parser.parse_args()

    if args.dataset_name == "all":
        datasets = src.utils.ALL_DATASETS
    else:
        datasets = [args.dataset_name]

    for dataset_name in datasets:
        edit_data = download_dataset(dataset_name, args.output_directory)
        get_dataset_dimensions(dataset_name, edit_data, splits=args.splits)

        if args.sample > 0:
            sample_dataset(dataset_name, n=args.sample)

        if args.write_to_jsonl:
            output_inputs_to_file(dataset_name, args.write_to_jsonl, args.splits)

    # load predictions
    if args.dataset_name != "all" and args.prediction_file:
        predictions = load_prediction_file(args.prediction_file)
        evaluate_predictions(
            metrics=args.metrics,
            dataset_name=args.dataset_name,
            predictions=predictions,
            edit_data=edit_data,
            normalize=(not args.no_normalization),
        )
