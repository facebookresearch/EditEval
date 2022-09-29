# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from src.base_processor import BaseProcessor
from datasets import load_dataset
from typing import List, Dict, Union
from src.prompts import TASK_INSTRUCTIONS


class JFLEGProcessor(BaseProcessor):
    def __init__(self, raw_path):
        super().__init__()
        self.dataset_name: str = "jfleg"
        self.raw_path: str = raw_path
        self._needs_normalize: bool = True
        self.split_mapping: Dict[str, str] = {"validation": "dev"}
        self.features_mapping: Dict[str, str] = {
            "sentence": "input",
            "corrections": "edits",
        }
        self.task_name: str = "fluency"

    def download_and_process(self) -> None:
        self.dataset = load_dataset(self.dataset_name, cache_dir=self.raw_path)


class ITERProcessor(BaseProcessor):
    def __init__(self, raw_path, task_type=None):
        super().__init__()
        self.dataset_name: str = "iterater"
        self.raw_path: str = raw_path
        self._needs_normalize: bool = True
        self.split_mapping: Dict[str, str] = {"validation": "dev"}
        self.features_mapping: Dict[str, str] = {
            "before_sent": "input",
            "after_sent": "edits",
        }
        self.task_name: str = task_type

    def download_and_process(self) -> None:
        dataset_dict = load_dataset("wanyu/IteraTeR_human_sent", cache_dir=self.raw_path)
        for split in dataset_dict.column_names:
            if self.task_name is None:
                dataset_dict[split] = dataset_dict[split].filter(
                    lambda x: x["labels"] in ["fluency", "clarity", "coherence"]
                )
            else:
                dataset_dict[split] = dataset_dict[split].filter(lambda x: x["labels"] == self.task_name)
            dataset_dict[split] = dataset_dict[split].filter(lambda x: len(x["after_sent"]) > 1)
            dataset_dict[split] = dataset_dict[split].add_column(
                "instructions", [TASK_INSTRUCTIONS[x] for x in dataset_dict[split]["labels"]]
            )
        self.dataset = dataset_dict


class TMUGFMProcessor(BaseProcessor):
    def __init__(self, raw_path: str):
        super().__init__()
        self.dataset_name: str = "tmu_gfm_dataset"
        self.raw_path: str = raw_path
        self._needs_normalize: bool = False
        self.split_mapping: Dict[str, str] = {}
        self.features_mapping: Dict[str, str] = {"source": "input", "output": "edits"}
        self.task_name: str = "fluency"

    def _wrap_outputs(
        self, data_split: Dict[str, Union[List[str], List[List[str]]]]
    ) -> Dict[str, Union[List[str], List[List[str]]]]:
        new_data_split = {}
        for key, values in data_split.items():
            if key == "output":
                new_data_split[key] = [values]
            else:
                new_data_split[key] = values
        return new_data_split

    def download_and_process(self) -> None:
        self.dataset = load_dataset(self.dataset_name, cache_dir=self.raw_path)

        for split in self.dataset.column_names:
            self.dataset[split] = self.dataset[split].filter(
                lambda x: x["ave_g"] > 3 and x["ave_f"] > 3 and x["ave_m"] > 3
            )
            self.dataset[split] = self.dataset[split].map(lambda x: self._wrap_outputs(x))


class PAWSProcessor(BaseProcessor):
    def __init__(self, raw_path: str):
        super().__init__()
        self.dataset_name: str = "paws"
        self.raw_path: str = raw_path
        self._needs_normalize: bool = True
        self.split_mapping: Dict[str, str] = {"validation": "dev"}
        self.features_mapping: Dict[str, str] = {
            "sentence1": "input",
            "sentence2": "edits",
            "id": "original_id",
        }
        self.task_name = "paraphrasing"

    def download_and_process(self) -> None:
        self.dataset = load_dataset(self.dataset_name, "labeled_final", cache_dir=self.raw_path)

        for split in self.dataset.column_names:
            self.dataset[split] = self.dataset[split].filter(lambda x: x["label"] == 1)


class STSBProcessor(BaseProcessor):
    def __init__(self, raw_path: str):
        super().__init__()
        self.dataset_name: str = "stsb_multi_mt"
        self.raw_path: str = raw_path
        self._needs_normalize: bool = False
        self.split_mapping: Dict[str, str] = {}
        self.features_mapping: Dict[str, str] = {
            "sentence1": "input",
            "sentence2": "edits",
        }
        self.task_name = "paraphrasing"

    def download_and_process(self) -> None:
        self.dataset = load_dataset(self.dataset_name, "en", cache_dir=self.raw_path)

        for split in self.dataset.column_names:
            self.dataset[split] = self.dataset[split].filter(lambda x: x["similarity_score"] == 5)


class MRPCProcessor(BaseProcessor):
    def __init__(self, raw_path: str):
        super().__init__()
        self.dataset_name: str = "mrpc"
        self.raw_path: str = raw_path
        self._needs_normalize: bool = True
        self.split_mapping: Dict[str, str] = {"validation": "dev"}
        self.features_mapping: Dict[str, str] = {
            "sentence1": "input",
            "sentence2": "edits",
        }
        self.task_name = "paraphrasing"

    def download_and_process(self) -> None:
        self.dataset = load_dataset("glue", self.dataset_name, cache_dir=self.raw_path)

        for split in self.dataset.column_names:
            self.dataset[split] = self.dataset[split].filter(lambda x: x["label"] == 1)


class TURKProcessor(BaseProcessor):
    def __init__(self, raw_path):
        super().__init__()
        self.dataset_name: str = "turk"
        self.raw_path: str = raw_path
        self._needs_normalize: bool = True
        self.split_mapping: Dict[str, str] = {"validation": "dev"}
        self.features_mapping: Dict[str, str] = {
            "original": "input",
            "simplifications": "edits",
        }
        self.task_name: str = "simplification"

    def download_and_process(self) -> None:
        self.dataset = load_dataset(self.dataset_name, cache_dir=self.raw_path)


class ASSETProcessor(BaseProcessor):
    def __init__(self, raw_path):
        super().__init__()
        self.dataset_name = "asset"
        self.raw_path = raw_path
        self._needs_normalize = False
        self.split_mapping = {"validation": "dev"}
        self.features_mapping = {"original": "input", "simplifications": "edits"}
        self.task_name = "simplification"

    def download_and_process(self) -> None:
        self.dataset = load_dataset(self.dataset_name, "simplification", cache_dir=self.raw_path)


if __name__ == "__main__":
    x = ITERProcessor("/checkpoint/janeyu/side_eval_datasets/raw/iterator")
    x.download_and_process()
    # print(x.dataset)
