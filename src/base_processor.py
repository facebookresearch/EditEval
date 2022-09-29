# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Dict, Union, Optional
from datasets import DatasetDict, Dataset
import json
from abc import abstractmethod
from src.utils import SPLITS, CORE_FEATURES
import warnings
from src.prompts import TASK_INSTRUCTIONS
import src.preprocessing


class BaseProcessor:
    def __init__(self):
        self.dataset = DatasetDict()
        self.dataset_name: Optional[str] = None
        self._needs_normalize: bool = True
        self.split_mapping: Dict[str, str] = {}
        self.features_mapping: Dict[str, str] = {}
        self.task_name: Optional[str] = None
        self.raw_path: Optional[str] = None

    def _normalize_data(
        self, data_split: Dict[str, Union[List[str], List[List[str]]]]
    ) -> Dict[str, Union[List[str], List[List[str]]]]:
        """Normalizes the text. This functions does not need to be customized per datasets."""
        new_data_split = {}
        for key, values in data_split.items():
            if key == "input":
                new_data_split[key] = src.preprocessing.normalize_text(values)
            elif key == "edits":
                if isinstance(values, str):
                    values = [values]
                new_data_split[key] = [src.preprocessing.normalize_text(text) for text in values]
        return new_data_split

    def _check_references_are_list(
        self, data_split: Dict[str, Union[List[str], List[List[str]]]]
    ) -> Dict[str, Union[List[str], List[List[str]]]]:
        """Normalizes the text. This functions does not need to be customized per datasets."""
        new_data_split = {}
        for key, values in data_split.items():
            if key == "edits":
                if isinstance(values, List):
                    return data_split
                else:
                    values = [values]
            new_data_split[key] = values
        return new_data_split

    def core_processing(self) -> None:
        """Standardizes the dataset format:
        1. Maps the splits of the DatasetDict based on self.split_mapping.
        2. Maps the features based on self.features_mapping.
        3. Normalizes data if self._need_normalizing is True.
        4. Wraps the references in a List if its just a string.
        5. Adds the type of task as a feature.
        6. Adds an id if not already present in features.
        7. Adds placeholder for retrieved documents if the task is retrieval-dependent.
        8. Filter out empty examples.
        This functions does not need to be customized per datasets."""

        # Renames the splits according to self.split_mapping which should only have values train, dev, or test
        for old_split, new_split in self.split_mapping.items():
            self.dataset[new_split] = self.dataset[old_split]
            del self.dataset[old_split]

        for split in self.dataset.column_names:
            # Renames the features in each split based on self.features_mapping, which only have values that are also in CORE_FEATURES.
            self.dataset[split] = self.dataset[split].rename_columns(self.features_mapping)

            # Normalize the data, for example, if it appears tokenized.
            if self._needs_normalize:
                self.dataset[split] = self.dataset[split].map(self._normalize_data)

            # Wrap the references in a List if its just a string.
            self.dataset[split] = self.dataset[split].map(self._check_references_are_list)

            # Adds task_name as a feature
            self.dataset[split] = self.dataset[split].add_column(
                "tasks", [self.task_name for _ in range(len(self.dataset[split]))]
            )

            if "id" not in self.dataset[split].column_names:
                self.dataset[split] = self.dataset[split].add_column(
                    "id",
                    [self.dataset_name + "-" + split + "-" + str(i) for i in range(len(self.dataset[split]))],
                )
            if "retrieved_documents" not in self.dataset[split].column_names:
                self.dataset[split] = self.dataset[split].add_column(
                    "retrieved_documents", [[] for _ in range(len(self.dataset[split]))]
                )

            if "title" not in self.dataset[split].column_names:
                self.dataset[split] = self.dataset[split].add_column(
                    "title", ["" for _ in range(len(self.dataset[split]))]
                )

            # if "instructions" not in self.dataset[split].column_names:
            #     self.dataset[split] = self.dataset[split].add_column(
            #         "instructions", [TASK_INSTRUCTIONS[self.task_name] for _ in range(len(self.dataset[split]))]
            #     )

            if "meta" not in self.dataset[split].column_names:
                self.dataset[split] = self.dataset[split].add_column(
                    "meta", [[] for _ in range(len(self.dataset[split]))]
                )

            # Filter out the empty examples.
            self.dataset[split] = self.dataset[split].filter(lambda x: len(x["input"]) > 0)
            self.dataset[split] = self.dataset[split].filter(lambda x: len(x["edits"]) > 0)

        # Adds an empty dictionary if one of the partitions train, dev, and test doesn't exist
        for col in SPLITS:
            if col not in list(self.dataset.column_names.keys()):
                warnings.warn(f"WARNING: {col} is missing from {self.dataset_name}")
                self.dataset[col] = Dataset.from_dict(
                    dict(
                        zip(
                            CORE_FEATURES,
                            [[] for _ in range(len(CORE_FEATURES))],
                        )
                    )
                )
        print(f"CORE FEATURES: {CORE_FEATURES}")

    @abstractmethod
    def download_and_process(self) -> None:
        pass


class ExternalProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()

    @classmethod
    def parse_from_json(
        cls,
        path: str = None,
        split_col_name: Optional[str] = None,
        split_name: Optional[str] = None,
        return_dataset_dict: bool = False,
    ) -> DatasetDict:
        dataset = {}
        with open(path, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                example = json.loads(line)

                if split_name is None:
                    split_name = example[split_col_name]

                if split_name not in dataset:
                    dataset[split_name] = {}

                for feature, value in example.items():
                    if feature not in dataset[split_name]:
                        dataset[split_name][feature] = []
                    dataset[split_name][feature].append(value)
        if not return_dataset_dict:
            return dataset
        dataset_dict = {}
        for split, dataset_objects in dataset.items():
            dataset_dict[split] = Dataset.from_dict(dataset_objects)
        dataset_dict = DatasetDict(dataset_dict)
        return dataset_dict

    @abstractmethod
    def download_and_process(self) -> None:
        pass
