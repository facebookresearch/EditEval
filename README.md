<!-- ![EditEval logo](./logo.png | width = 100) -->
<p align="center">
<img src="./logo.png" width="700">
</p>


# The Instruction-Based Benchmark for Text Improvements

The EditEval benchmark is described in the following paper:

<!-- ```bibtex
@inproceedings{petroni-etal-2021-kilt,
    title = "{KILT}: a Benchmark for Knowledge Intensive Language Tasks",
    author = {Petroni, Fabio  and Piktus, Aleksandra  and
      Fan, Angela  and Lewis, Patrick  and
      Yazdani, Majid  and De Cao, Nicola  and
      Thorne, James  and Jernite, Yacine  and
      Karpukhin, Vladimir  and Maillard, Jean  and
      Plachouras, Vassilis  and Rockt{\"a}schel, Tim  and
      Riedel, Sebastian},
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association 
                 for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.200",
    doi = "10.18653/v1/2021.naacl-main.200",
    pages = "2523--2544",
}
```

[https://arxiv.org/abs/2009.02252](https://arxiv.org/abs/2009.02252) -->

## Leaderboard

The leaderboard for this benchmark can be found on [EvalAI](https://eval.ai/web/challenges/challenge-page/1866/manage).  

## Installation

```
conda create -n editeval -y python=3.7 && conda activate editeval
pip install -e .
```
## Additional dependencies

The [FRUIT](https://github.com/google-research/language/tree/master/language/fruit) dataset requires that you install [gsutil](https://cloud.google.com/storage/docs/gsutil_install).

## Downloading datasets

This will download to the directory /data. To specify a different output directory use ```output_directory={path_to_output_dir}```.

For a single dataset run: 

    python main.py --dataset_name {dataset_name}

For all datasets run: 

    python main.py --dataset_name all

## Writing datasets to jsonl files

For a single dataset run:

    python main.py --dataset_name {dataset_name} --write_to_jsonl

For all datasets run:

    python main.py --dataset_name all --write_to_jsonl

## Sampling datasets

    python main.py --dataset_name jfleg --sample {num_examples_to_sample}

## Running evaluation for a dataset

    python main.py --dataset_name {dataset_name}  --prediction_file {path_to_jsonl}

To specify certain metrics (e.g., gleu and sari): 

    python main.py --dataset_name {dataset_name}  --prediction_file {path_to_jsonl} --metrics gleu sari

To turn off normalization during evaluation, specify ```--no_normalization```.

# Current tasks and datasets
- Fluency
    - jfleg
    - iterater_fluency
- Clarity
    - iterater_clarity
- Coherence 
    - iterater_coherence
- Paraphrasing
    - stsb_multi_mt
- Simplification
    - turk
    - asset
- Neutralization
    - wnc
- Updating
    - fruit
    - wafer_insert
    
# Current metrics
- sari
- em
- em_diff
- bleu
- ibleu
- gleu
- rouge
- update_rouge
- bert_score

## Licensing

See our LICENSE file for licensing details.
