# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Dict, Any, Tuple
from src.metrics.custom_metrics import (
    ExactMatchMetric,
    ExactMatchDiffMetric,
    GLEUMetric,
    InsertDeleteMetric,
    EasseSariMetric,
    UpdateRougeMetric,
    iBLEUMetric,
)
import numpy as np
import pandas as pd
import urllib
import os

CORE_FEATURES: List[str] = ["id", "title", "input", "edits", "retrieved_documents", "tasks"]
SPLITS: List[str] = ["train", "dev", "test"]

DEFAULT_SPLITS = {
    "jfleg": ["test"],
    "wnc": ["train", "dev", "test"],
    "tmu_gfm_dataset": ["train"],
    "asset": ["test"],
    "wafer_insert": ["dev", "test"],
    "fruit": ["test"],
    "iterater_clarity": ["test"],
}

HUGGINGFACE_DATASETS: List[str] = [
    "jfleg",
    "asset",
    "turk",
    "iterater",
    "stsb_multi_mt",
]
CUSTOM_DATASETS: List[str] = ["wnc", "fruit", "wafer_insert"]

HUGGINGFACE_METRICS = ["bleu", "sacrebleu", "meteor", "bert_score", "rouge"]
CUSTOM_METRICS = {
    "em_diff": ExactMatchDiffMetric,
    "em": ExactMatchMetric,
    "id": InsertDeleteMetric,
    "ibleu": iBLEUMetric,
    "sari": EasseSariMetric,
    "gleu": GLEUMetric,
    "update_rouge": UpdateRougeMetric,
}

ALL_DATASETS = HUGGINGFACE_DATASETS + CUSTOM_DATASETS
ALL_METRICS = HUGGINGFACE_METRICS + list(CUSTOM_METRICS.keys())

categories_dict = {
    "clarity": ["iterater_clarity"],
    "coherence": ["iterater_coherence"],
    "fluency": ["jfleg", "iterater_fluency"],
    "neutralization": ["wnc"],
    "paraphrasing": ["paws", "stsb_multi_mt", "mrpc"],
    "simplification": ["turk", "asset"],
    "updating": ["fruit", "wafer_insert"],
}


def download_url(url, root, filename=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    try:
        print("Downloading " + url + " to " + fpath)
        urllib.request.urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead." " Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath)


def get_dataset_names_from_task(tasks: List[str]) -> List[str]:
    dataset_names = []
    for task in tasks:
        dataset_names.extend(categories_dict[task])
    return dataset_names


def get_dataset_categories(dataset_names: List[str]) -> Tuple[List[str], Dict[str, str]]:
    categories_list = []
    for dataset_name in dataset_names:
        for categories, datasets in categories_dict.items():
            if dataset_name in datasets:
                categories_list.append(categories)
                continue
    return categories_list


def get_dataset_category(dataset_name: str) -> Tuple[List[str], Dict[str, str]]:
    for category, datasets in categories_dict.items():
        if dataset_name in datasets:
            return category
    return None


def transpose_dict(dict_of_lists: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Converts a dict of lists into a list of dicts."""
    return [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]


def print_metric_report(
    scores_dict: Dict[str, Dict[str, float]], size: int = None, dataset_name: str = None, format: str = "markdown"
) -> None:
    report_df = []
    for metric, metric_scores in scores_dict.items():
        row = [metric, np.mean(metric_scores)]
        if size:
            row.append(size)
        if dataset_name:
            row.append(dataset_name)
        report_df.append(row)

    cols = ["Metric", "Average score"]
    if size:
        cols.append("Size")
    if dataset_name:
        cols.append("Dataset")

    report_df = pd.DataFrame(
        report_df,
        columns=cols,
    )
    if format == "markdown":
        print(report_df.to_markdown())
    elif format == "tsv":
        print(report_df.to_csv(sep="\t"))
    else:
        raise NotImplementedError
