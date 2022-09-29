# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from datasets import DatasetDict
from typing import Callable, List, Dict, Union, Optional, Tuple, Any
from src.evaluator import Evaluator
import os
from src.processors import huggingface_processors, wnc, fruit, wafer_insert
from src.utils import SPLITS, transpose_dict, print_metric_report
from functools import partial

PROCESSORS = {
    "asset": huggingface_processors.ASSETProcessor,
    "fruit": fruit.FRUITProcessor,
    "jfleg": huggingface_processors.JFLEGProcessor,
    "iterater": huggingface_processors.ITERProcessor,
    "iterater_clarity": partial(huggingface_processors.ITERProcessor, task_type="clarity"),
    "iterater_coherence": partial(huggingface_processors.ITERProcessor, task_type="coherence"),
    "iterater_fluency": partial(huggingface_processors.ITERProcessor, task_type="fluency"),
    "iterater_style": partial(huggingface_processors.ITERProcessor, task_type="style"),
    "stsb_multi_mt": huggingface_processors.STSBProcessor,
    "turk": huggingface_processors.TURKProcessor,
    "wnc": wnc.WNCProcessor,
    "wafer_insert": wafer_insert.WAFERInsertProcessor,
}


def instantiate_processor(name, raw_path):
    if name not in PROCESSORS:
        raise ValueError(f"{name} not in available processors.")
    return PROCESSORS[name](raw_path)


class EditDataset:
    """Class for combining and evaluating datasets of a particular editing task.
    Resulting dataset should be a DatasetDict with keys ['train', 'dev', and 'test].
    """

    def __init__(
        self,
        dataset_name: str,
        output_dir: str = None,
        dataset_paths_config: str = "./configs/dataset_paths.json",
        verbose: bool = False,
    ):
        """
        :param dataset_name: Name of the dataset to load (as enumerated in self.categories_config_file).
        :param output_dir: Directory to hold all the raw and processed data for all datasets (where dataset_name is used as the subdir).
        :param dataset_paths_config: Path to the config with the paths to the specific raw data processed data directories if output_dir is None.
        """

        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.dataset_paths_config = dataset_paths_config

        raw_path = self._get_dataset_raw_path(dataset_name)
        processed_dataset_path = self._get_processed_dataset_dir(dataset_name)

        if verbose:
            print(f"Dataset name is {str(set(self.dataset_name))}.")

        dataset = self._download_and_process_files(
            dataset_name,
            raw_path,
            processed_dataset_path,
        )
        self.dataset = dataset

    def _download_and_process_files(
        self,
        dataset_name: str,
        raw_path: str,
        processed_dataset_path: str,
    ) -> DatasetDict:
        """Download and process the dataset if it has not been done before.
        :param dataset_name: The name of the dataset.
        :param raw_path: Path to the raw data.
        :param processed_dataset_path: Path to the processed data.
        """
        try:
            f = open(processed_dataset_path)
            dataset = DatasetDict.load_from_disk(processed_dataset_path)
            for col in ["train", "dev", "test"]:
                if col not in dataset or dataset[col].num_rows == 0:
                    print(f"WARNING: {col} is missing from {dataset_name}")
            print(f"Loaded from disk: {processed_dataset_path}")
        except:
            processor = instantiate_processor(dataset_name, raw_path)
            processor.download_and_process()
            processor.core_processing()
            processor.dataset.save_to_disk(dataset_dict_path=processed_dataset_path)
            print(f"Downloaded and processed to cache: {processed_dataset_path}")
            dataset = processor.dataset
        return dataset

    def _get_processed_dataset_dir(self, dataset_name: str) -> str:
        if self.output_dir is not None:
            print(f"Path to processed dir is {os.path.join(self.output_dir, 'processed', dataset_name)}")
            return os.path.join(self.output_dir, "processed", dataset_name)
        else:
            with open(self.dataset_paths_config, "r") as cf:
                config_dict = json.load(cf)
            return config_dict[dataset_name]["processed_dir"]

    def _get_dataset_raw_path(self, dataset_name: str) -> str:
        if self.output_dir is not None:
            return os.path.join(self.output_dir, "raw", dataset_name)
        else:
            with open(self.dataset_paths_config, "r") as cf:
                config_dict = json.load(cf)
            return config_dict[dataset_name]["raw_dir"]

    def get_features(
        self, features: List[str], ids: List[str] = None, splits: Optional[List[str]] = SPLITS, transpose: bool = False
    ) -> Dict[str, List]:
        all_data = {}

        if ids:
            ids = set(ids)

            for split in SPLITS:
                for i, idx in enumerate(self.dataset[split]["id"]):
                    if idx not in ids:
                        continue
                    example = self.dataset[split][i]  # row indexing first is faster
                    for feature in features:
                        if feature not in all_data:
                            all_data[feature] = []
                        if feature == "dataset_name":
                            all_data[feature].append(self.dataset_name)
                        else:
                            all_data[feature].append(example[feature])
        else:
            for split in splits:
                for feature in features:
                    if feature not in all_data:
                        all_data[feature] = []
                    if feature == "dataset_name":
                        all_data[feature].extend([self.dataset_name] * len(self.dataset[split]))
                    else:
                        all_data[feature].extend(self.dataset[split][feature])
        return transpose_dict(all_data) if transpose else all_data

    def get_edits(self, ids: Optional[List[str]] = None, splits: Optional[List[str]] = SPLITS) -> List[List[str]]:
        return self.get_features(["edits"], splits=splits, ids=ids)["edits"]

    def get_raw(self, ids: Optional[List[str]] = None, splits: Optional[List[str]] = SPLITS) -> List[Dict]:
        return self.get_features(["raw"], splits=splits, ids=ids)["raw"]

    def get_inputs(
        self,
        ids: Optional[List[str]] = None,
        splits: Optional[List[str]] = SPLITS,
        transpose: bool = False,
    ) -> Dict[str, List]:
        return self.get_features(
            ["dataset_name", "id", "title", "input", "edits", "tasks", "retrieved_documents"],
            splits=splits,
            ids=ids,
            transpose=transpose,
        )

    def evaluate(
        self,
        pred_dict: Dict[str, List],
        metrics: Optional[List[str]] = None,
        input_dict: Optional[Dict[str, List]] = None,
        print_report: str = None,
        normalize: Union[Callable, bool] = False,
    ) -> None:
        if input_dict is None:
            input_dict = self.get_inputs(ids=pred_dict["id"])
        ref_edits = self.get_edits(ids=pred_dict["id"])

        print(len(input_dict["id"]))
        scores_dict = Evaluator.evaluate(
            targets=ref_edits,
            input_dict=input_dict,
            pred_dict=pred_dict,
            metrics=metrics,
            normalize=True,
            dataset_name=self.dataset_name,
        )

        if print_report:
            print_metric_report(scores_dict, format=print_report, size=len(ref_edits), dataset_name=self.dataset_name)
