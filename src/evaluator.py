# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from src.preprocessing import tokenize_text, normalize_text
from typing import List, Optional, Dict, Union, Tuple
from collections import defaultdict
from datasets import Metric, load_metric
from src.utils import HUGGINGFACE_METRICS, CUSTOM_METRICS
import re
from transformers import AutoTokenizer
from src.metrics.custom_metrics import CustomMetric


def unwikify(
    text: str,
    remove_wikilinks: bool = True,
    remove_targets: bool = True,
    remove_sections: bool = True,
    remove_other_markup: bool = True,
    remove_categories: bool = True,
) -> str:
    """Remove Wikipedia-specific syntax from a given piece of text."""

    if remove_categories:
        text = re.sub(r"\[\[Category[^]]+]]", "", text)
    if remove_targets:
        text = re.sub(r"\[\[\[REF[^]]+]]]", "", text)
        text = re.sub(r"\[\[\[0-9]+]]]", "", text)
    if remove_wikilinks:
        text = re.sub(r"\[\[([^|\]]+)\|([^]]+)]]", r"\2", text)
        text = text.replace("[[", "").replace("]]", "")
    if remove_other_markup:
        text = text.replace("'''", "").replace("''", "")
    if remove_sections:
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("==")]
        text = "\n".join(lines)

    return text


class Evaluator:
    """Collection of functions for evaluating given inputs, outputs, and predictions"""

    @classmethod
    def check_input_and_pred_ids_match(cls, input_ids: List[str], pred_ids: List[str]) -> bool:
        for input_id, pred_id in zip(input_ids, pred_ids):
            assert input_id == pred_id, f"The id in the input dict {input_id} does not match the output dict {pred_id}."
        return True

    @classmethod
    def normalize_refs(cls, text: str, targets: List[str], tokenizer: AutoTokenizer, dataset: str, field: str):
        """Text normalization before evaluation"""
        assert field in {"input", "prediction", "reference"}
        if dataset == "fruit" and field in {"prediction", "reference"}:  # remove context indices
            num_contexts = len(targets)
            to_rm = r"|".join(map(re.escape, [f"({i})" for i in range(num_contexts)]))
            text = re.sub(to_rm, "", text)
        if dataset == "fruit":
            text = unwikify(text)
        # encode and decode with tokenizer
        clean_by_tokenizer = lambda text: tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True)
        text = clean_by_tokenizer(text)
        # strip
        text = text.strip()
        return text

    @classmethod
    def format_evaluation_inputs(
        cls,
        reference_documents: List[str],
        inputs: List[str],
        predictions: List[str],
        targets: List[List[str]],
        dataset_name: str = None,
        normalize: bool = False,
    ) -> Tuple[List[str], List[str], List[List[str]], List[List[str]], List[List[str]], List[List[List[str]]],]:
        assert len(predictions) == len(
            targets
        ), f"The number of predictions is {len(predictions)} but reference(s) is {len(targets)}."
        assert len(inputs) == len(
            predictions
        ), f"The number of inputs is {len(inputs)} but reference(s) is {len(predictions)}."

        if dataset_name is not None and (dataset_name == "fruit" or dataset_name == "wafer_insert"):
            # /checkpoint/schick/peer/models/peer_edit_infilling_plans/checkpoint-20000
            # t5-small
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            normalize_func = lambda text, refs, field: cls.normalize_refs(
                text=text, targets=refs, tokenizer=tokenizer, dataset=dataset_name, field=field
            )
            normalize = True
        else:
            normalize_func = lambda text, targets, field: normalize_text(text)  # default normalization function

        if normalize:
            inputs = [normalize_func(inp, refs, field="input") for inp, refs in zip(inputs, reference_documents)]
            predictions = [
                normalize_func(pred, refs, field="prediction") for pred, refs in zip(predictions, reference_documents)
            ]
            targets = [
                [normalize_func(ref, refs, field="reference") for ref in example_refs]
                for example_refs, refs in zip(targets, reference_documents)
            ]

        input_tokens = [tokenize_text(input) if len(input) > 0 else [] for input in inputs]
        prediction_tokens = [tokenize_text(pred) for pred in predictions]
        reference_tokens = [[tokenize_text(ref_text) for ref_text in example_refs] for example_refs in targets]

        return (
            inputs,
            predictions,
            targets,
            input_tokens,
            prediction_tokens,
            reference_tokens,
        )

    @classmethod
    def get_scorers(cls, metrics: Optional[List[str]] = None) -> List[Union[Metric, CustomMetric]]:
        if metrics is None:
            metrics = itertools.chain(HUGGINGFACE_METRICS, CUSTOM_METRICS.keys())
        scorers = []
        for metric_name in metrics:
            if metric_name == "bert_score":
                scorers.append(load_metric("bertscore"))
            elif metric_name in HUGGINGFACE_METRICS:
                scorers.append(load_metric(metric_name))
            else:
                scorers.append(CUSTOM_METRICS[metric_name]())
        return scorers

    @classmethod
    def evaluate(
        cls,
        dataset_name: str = None,
        input_dict: Optional[Dict[str, List]] = None,
        pred_dict: Optional[Dict[str, List]] = None,
        inputs: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
        targets: List[List[str]] = None,
        reference_documents: List[str] = None,
        metrics: Optional[List[str]] = None,
        normalize: Optional[bool] = False,
        categories: Optional[List[str]] = None,
    ) -> List[Dict[str, float]]:
        """Main evaluation function that can accept {input_dict, pred_dict} or {inputs, predictions, targets}.
        The input_dict has keys ["dataset_name", "id", "title", "input", "edits", "tasks", "retrieved_documents"].
        The pred_dict has keys ["id", "predictions"]."""
        if inputs is None:
            if input_dict is not None:
                inputs = input_dict["input"]
            else:
                inputs = ["" for x in range(len(predictions))]

        if reference_documents is None:
            if input_dict is not None:
                reference_documents = input_dict["retrieved_documents"]
            else:
                reference_documents = [[] for x in range(len(inputs))]

        if predictions is None and pred_dict is not None:
            predictions = pred_dict["pred"]

        if input_dict is not None and pred_dict is not None:
            cls.check_input_and_pred_ids_match(input_dict["id"], pred_dict["id"])

        all_metric_scores = {}
        scorers = cls.get_scorers(metrics=metrics)
        (
            inputs,
            predictions,
            targets,
            input_tokens,
            prediction_tokens,
            reference_tokens,
        ) = cls.format_evaluation_inputs(
            reference_documents=reference_documents,
            inputs=inputs,
            predictions=predictions,
            targets=targets,
            dataset_name=dataset_name,
            normalize=normalize,
        )

        for scorer in scorers:
            score_dict = cls.evaluate_metric(
                scorer,
                candidate_tokens=prediction_tokens,
                reference_tokens=reference_tokens,
                original_tokens=input_tokens,
                candidate_text=predictions,
                reference_texts=targets,
                original_text=inputs,
                categories=categories,
            )
            for metric, score in score_dict.items():
                if metric.startswith("_"):
                    all_metric_scores[metric[1:]] = score
                elif scorer.name in metric:
                    all_metric_scores[metric] = score
        return all_metric_scores

    @classmethod
    def evaluate_metric(
        cls,
        scorer: Union[Metric, CustomMetric],
        candidate_tokens: Optional[List[List[str]]] = None,
        reference_tokens: Optional[List[List[List[str]]]] = None,
        original_tokens: Optional[List[List[str]]] = None,
        candidate_text: Optional[List[str]] = None,
        reference_texts: Optional[List[List[str]]] = None,
        original_text: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, float]:

        categories = categories if categories else [""] * len(reference_texts)
        cate2data: Dict[str, Tuple[List, List, List, List, List, List]] = defaultdict(lambda: ([], [], [], [], [], []))
        overall_dict = {}

        for i, c in enumerate(categories):
            cate2data[c][0].append(candidate_tokens[i] if candidate_tokens else None)
            cate2data[c][1].append(reference_tokens[i] if reference_tokens else None)
            cate2data[c][2].append(original_tokens[i] if original_tokens else None)
            cate2data[c][3].append(candidate_text[i] if candidate_text else None)
            cate2data[c][4].append(reference_texts[i] if reference_texts else None)
            cate2data[c][5].append(original_text[i] if original_text else None)

        for c, (ct, rt, ot, cx, rx, ox) in cate2data.items():
            if isinstance(scorer, Metric):
                if scorer.name in ["sari"]:
                    score_dict = scorer.compute(sources=ox, predictions=cx, references=rx)
                elif scorer.name in ["meteor"]:
                    score_dict = scorer.compute(predictions=cx, references=rx)
                elif scorer.name in ["bleu", "exact_match"]:
                    score_dict = scorer.compute(predictions=ct, references=rt)
                elif scorer.name in ["bert_score"]:
                    score = scorer.compute(predictions=cx, references=rx, lang="en")["f1"][0]
                    score_dict = {"bert_score": score}
                elif scorer.name in ["rouge"]:
                    scores = scorer.compute(predictions=cx, references=rx)
                    score_dict = {
                        "rouge1": scores["rouge1"].mid.fmeasure,
                        "rouge2": scores["rouge2"].mid.fmeasure,
                        "rougeL": scores["rougeL"].mid.fmeasure,
                    }
                else:
                    score = scorer.compute(predictions=cx, references=rx)["score"]
                    score_dict = {scorer.name: score}
            else:
                score = scorer.evaluate_safe(
                    originals=ox,
                    predictions=cx,
                    targets=rx,
                    original_tokens=ot,
                    prediction_tokens=ct,
                    target_tokens=rt,
                )
                if type(score) is dict:
                    score_dict = score
                else:
                    score_dict = {scorer.name: score}

            overall_dict.update({(f"{c}-{k}" if c else k): v for k, v in score_dict.items()})

        return overall_dict
