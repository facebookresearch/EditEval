# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from src.base_processor import ExternalProcessor
import os
import pandas as pd
from datasets import DatasetDict, Dataset
from src.utils import SPLITS, download_url


class WNCProcessor(ExternalProcessor):
    def __init__(self, raw_path: str):
        super().__init__()
        self.dataset_name: str = "wnc"
        self.raw_path: str = raw_path
        self._needs_normalize: bool = False
        self.split_mapping: Dict[str, str] = {}
        self.features_mapping: Dict[str, str] = {}
        self.task_name: str = "neutralization"
        self.host_link: str = "http://bit.ly/bias-corpus"

    def download_and_process(self) -> None:
        if not os.path.exists(self.raw_path):
            download_url(self.host_link, self.raw_path, filename="bias-corpus.zip")
            os.system(f"unzip {os.path.join(self.raw_path, 'bias-corpus')} -d {self.raw_path}")
            os.system(f"rm {os.path.join(self.raw_path, 'bias-corpus')}")

        original_columns = [
            "id",
            "src_tok",
            "tgt_tok",
            "input",
            "edits",
            "src_POS_tags",
            "tgt_parse_tags",
        ]

        dfs = []

        for split in SPLITS:
            full_path = os.path.join(self.raw_path, "bias_data", "WNC", "biased.word." + split)
            df = pd.read_csv(full_path, sep="\t", names=original_columns)
            df["id"] = df["id"].apply(lambda x: "wnc-" + split + "-" + str(x))
            df["wiki_id"] = df["id"].apply(lambda x: x)
            dfs.append(df)

        dev_dataset = Dataset.from_pandas(dfs[1])
        test_dataset = Dataset.from_pandas(dfs[2])
        self.dataset = DatasetDict({"dev": dev_dataset, "test": test_dataset})


if __name__ == "__main__":
    e = WNCProcessor("/checkpoint/janeyu/side_eval_datasets/raw/wnc")
    e.download_and_process()
    print(e.dataset)
