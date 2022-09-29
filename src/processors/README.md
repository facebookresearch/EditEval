# How to add a dataset not in the Hugging Face library to EditEval

## Introduction
To this we will create a subclass of the [ExternalProcessor](https://github.com/fairinternal/side/blob/b322bc1/projects/edit_eval/src/base_processor.py#L116) class which inherits from the [BaseProcessor](https://github.com/fairinternal/side/blob/b322bc1/projects/edit_eval/src/base_processor.py#L10) class. The processor's function download_and_process will be called from [_download_and_process_files](https://github.com/fairinternal/side/blob/b322bc1/projects/edit_eval/src/dataset.py#L80) in src/dataset.py which first will check to see if the processed files are on the local machine. If not, the class you are about to implement will be used for downloading and processing the raw files.

The data type of the processed dataset will be DatasetDict from Hugging Face (essentially a dictionary of dictionaries). A really simple example of creating a DatasetDict would look like:
```
DatasetDict({'train': {'input': ['cheese'], 'output': ['Cheese!']}})
```

The purpose of the processor is to download the raw files and create a DatasetDict where the first set of keys are the dataset splits (e.g., train, test) and the second set of keys define the dataset features, each of which map to a list of the examples. 

Note: There is already functionality to do the following:
1. Normalize the data, if needed. 
2. Filter empty examples.
3. Map the dictionary keys to a standardized set of names.
4. Adding placeholders for core fields ["id", "title", "input", "edits", "retrieved_documents", "tasks"] and for any splits that do not exist in ["train", "dev", "test"].
5. Save the dataset.

# Let's get started!

## Simple setup
1. Create a python file in projects/edit_eval/src/processors/{your_dataset}.py
2. Implement the class {YourDatasetProcessor} (e.g., `class WNCProcessor(ExternalProcessor)`)
3. Define the initializer as
```
def __init__(self, raw_path):
    super().__init__()
```

## Define several properties of the class in the initializer

1. Set the raw path to the input raw path: `self.raw_path = raw_path`
2. Set the dataset name: self.dataset_name = "{your_dataset}"
3. Set the boolean self._needs_normalize. Put True if the spaces around punctuation need to be removed, for example.
4. Set the task_name (e.g., `self.task_name = 'grammar'`). It should be one of ['clarity', 'coherence', 'simplification', 'paraphrasing', 'neutralization', 'fluency', 'updating'].

## Define the mapping to the standardized naming conventions
You can skip this step if you choose to convert the naming yourself in the `download_and_process` function. All the datasets are processed into a standardized naming convention, where the dataset split names must be in the list src/utils.SPLITS and the dataset features should be renamed as those in src/utils.CORE_FEATURES, if possible. You can define self.split_mapping and self.features_mapping which maps from the unstandardized names to the standardized names, if they are different (e.g., self.split_mapping = {'validation': 'dev'}).

```
SPLITS = ["train", "dev", "test"]
CORE_FEATURES = ["id", "input", "edits", "retrieved_documents", "tasks"]
```

Example from running in dataset.py

```
>>> jfleg_data = EditDataset("jfleg")
>>> print(jfleg_data.dataset)

DatasetDict({
    test: Dataset({
        features: ['input', 'edits', 'tasks', 'id', 'retrieved_documents'],
        num_rows: 747
    })
    dev: Dataset({
        features: ['input', 'edits', 'tasks', 'id', 'retrieved_documents'],
        num_rows: 754
    })
    train: Dataset({
        features: ['id', 'input', 'edits', 'retrieved_documents', 'tasks'],
        num_rows: 0
    })
})
```

```
>>> jfleg_data = EditDataset("jfleg")
>>> print(jfleg_data.datasets[0]["test"][:2])
{
    'input': ['New and new technology has been introduced to the society.', 'One possible outcome is that an environmentally-induced reduction in motorization levels in the richer countries will outweigh any rise in motorization levels in the poorer countries.'], 
    'edits': [['New technology has been introduced to society.', 'New technology has been introduced into the society.', 'Newer and newer technology has been introduced into society.', 'Newer and newer technology has been introduced to the society.'], ['One possible outcome is that an environmentally-induced reduction in motorization levels in richer countries will outweigh any rise in motorization levels in poorer countries.', 'One possible outcome is that an environmentally-induced reduction in motorization levels in the richer countries will outweigh any rise in motorization levels in the poorer countries.', 'One possible outcome is that an environmentally induced reduction in motorization levels in the richer countries will outweigh any rise in motorization levels in the poorer countries.', 'One possible outcome is that an environmentally-induced reduction in motorization levels in the richer countries will outweigh any rise in motorization levels in the poorer countries.']], 
    'tasks': ['grammar', 'grammar'], 
    'id': ['jfleg-test-0', 'jfleg-test-1'], 'retrieved_documents': [[], []]
}
```

## Implement the method download_and_process 

This function should download the relevant files to the directory `raw_path` and then process it into a DatasetDict. Some resources for this:

- Importing from a [csv file](https://huggingface.co/docs/datasets/v1.12.0/_modules/datasets/dataset_dict.html#DatasetDict.from_csv).
- Importing from a [json file](https://huggingface.co/docs/datasets/v1.12.0/_modules/datasets/dataset_dict.html#DatasetDict.from_json). If your data files are in the format of a json, where each line is a dictionary and you know which key of the dictionary corresponds to the split name, you can call [ExternalProcessor.parse_from_json(path, split_col_name)](https://github.com/fairinternal/side/blob/b322bc1/projects/edit_eval/src/base_processor.py#L121) <-- NOT TESTED.
- From pandas dataframes (see example in the [WNCProcessor](https://github.com/fairinternal/side/blob/b322bc1/projects/edit_eval/src/processors/wnc.py#L41).

## Incorporate into EditEval
1. Add your dataset to src/dataset.PROCESSORS
2. Add your dataset to src/utils.CUSTOM_DATASETS
