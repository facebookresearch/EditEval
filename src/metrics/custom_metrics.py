# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import abc
from typing import List, Tuple, Optional
from src.diff_utils import source_target_to_diffs
from datasets import load_metric


class CustomMetric(abc.ABC):
    def __init__(self, name: str):
        self.name = name

    def evaluate_safe(
        self,
        originals: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
        targets: Optional[List[List[str]]] = None,
        original_tokens: Optional[List[List[str]]] = None,
        prediction_tokens: Optional[List[List[str]]] = None,
        target_tokens: Optional[List[List[List[str]]]] = None,
    ) -> float:
        assert (originals is not None and predictions is not None and targets is not None) or (
            original_tokens is not None and prediction_tokens is not None and target_tokens is not None
        )
        assert len(originals) == len(predictions) == len(targets)
        assert all(isinstance(x, list) for x in targets)
        assert len(original_tokens) == len(prediction_tokens) == len(target_tokens)
        assert all(isinstance(x, list) for x in target_tokens)
        return self.evaluate(
            originals=originals,
            predictions=predictions,
            targets=targets,
            original_tokens=original_tokens,
            prediction_tokens=prediction_tokens,
            target_tokens=target_tokens,
        )

    @abc.abstractmethod
    def evaluate(
        self,
        originals: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
        targets: Optional[List[List[str]]] = None,
        original_tokens: Optional[List[List[str]]] = None,
        prediction_tokens: Optional[List[List[str]]] = None,
        target_tokens: Optional[List[List[List[str]]]] = None,
        **kwargs,
    ) -> float:
        raise NotImplementedError()


class ExactMatchDiffMetric(CustomMetric):
    def __init__(self):
        super().__init__(name="em_diff")

    def evaluate(
        self,
        originals: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
        targets: Optional[List[List[str]]] = None,
        original_tokens: Optional[List[List[str]]] = None,
        prediction_tokens: Optional[List[List[str]]] = None,
        target_tokens: Optional[List[List[List[str]]]] = None,
        **kwargs,
    ) -> float:
        score = 0

        for input, output, target_list in zip(originals, predictions, targets):
            output_diff = source_target_to_diffs(input, output, num_context_words=0)
            target_diffs = [source_target_to_diffs(input, target, num_context_words=0) for target in target_list]

            score += max(
                len(set(output_diff).intersection(target_diff)) / max((1, len(output_diff), len(target_diff)))
                for target_diff in target_diffs
            )

        return score / len(targets)


class ExactMatchMetric(CustomMetric):
    def __init__(self):
        super().__init__(name="em")

    def evaluate(
        self,
        originals: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
        targets: Optional[List[List[str]]] = None,
        original_tokens: Optional[List[List[str]]] = None,
        prediction_tokens: Optional[List[List[str]]] = None,
        target_tokens: Optional[List[List[List[str]]]] = None,
        **kwargs,
    ) -> float:
        return sum(1 for pred, target_list in zip(predictions, targets) if pred in target_list) / len(targets)


class InsertDeleteMetric(CustomMetric):
    def __init__(self):
        super().__init__(name="id")

    @staticmethod
    def get_new_ngrams(source: List[str], target: List[str], n: int) -> List[Tuple[str, ...]]:
        source_ngrams = InsertDeleteMetric.get_ngrams(source, n)
        target_ngrams = InsertDeleteMetric.get_ngrams(target, n)
        new_ngrams = [x for x in target_ngrams if x not in source_ngrams]
        return new_ngrams

    @staticmethod
    def get_ngrams(input_list: List[str], n: int) -> List[Tuple[str, ...]]:
        return list(zip(*[input_list[i:] for i in range(n)]))

    @staticmethod
    def overlap_score(source: List, target: List) -> float:
        if not source and not target:
            return 1
        return len(set(source) & set(target)) / max(len(set(source)), len(set(target)))

    @staticmethod
    def get_insert_delete_ngram_score(source: str, target: str, prediction: str) -> float:
        score = 0

        source = source.split()
        target = target.split()
        prediction = prediction.split()

        # We just take 1-, 2-, and 3-grams.
        for n in [1, 2, 3]:
            target_insertions = InsertDeleteMetric.get_new_ngrams(source, target, n)
            predicted_insertions = InsertDeleteMetric.get_new_ngrams(source, prediction, n)

            target_deletions = InsertDeleteMetric.get_new_ngrams(target, source, n)
            predicted_deletions = InsertDeleteMetric.get_new_ngrams(prediction, source, n)

            ins_score = InsertDeleteMetric.overlap_score(target_insertions, predicted_insertions)
            del_score = InsertDeleteMetric.overlap_score(target_deletions, predicted_deletions)

            score += (ins_score + del_score) / 2

        score /= 3
        return score

    def evaluate(
        self,
        originals: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
        targets: Optional[List[List[str]]] = None,
        original_tokens: Optional[List[List[str]]] = None,
        prediction_tokens: Optional[List[List[str]]] = None,
        target_tokens: Optional[List[List[List[str]]]] = None,
        **kwargs,
    ) -> float:
        score = sum(
            max(InsertDeleteMetric.get_insert_delete_ngram_score(input, target, output) for target in target_list)
            for input, target_list, output in zip(originals, targets, predictions)
        ) / len(targets)
        return score


class EasseSariMetric(CustomMetric):
    def __init__(self):
        super().__init__(name="sari")

    def evaluate(
        self,
        originals: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
        targets: Optional[List[List[str]]] = None,
        original_tokens: Optional[List[List[str]]] = None,
        prediction_tokens: Optional[List[List[str]]] = None,
        target_tokens: Optional[List[List[List[str]]]] = None,
        **kwargs,
    ) -> float:
        from src.metrics.easse_sari import corpus_sari

        # Easse expects targets in a transposed way.
        targets = list(map(list, zip(*targets)))
        return corpus_sari(
            orig_sents=originals,
            sys_sents=predictions,
            refs_sents=targets,
        )


class GLEUMetric(CustomMetric):
    def __init__(self):
        super().__init__(name="gleu")

    def evaluate(
        self,
        originals: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
        targets: Optional[List[List[str]]] = None,
        original_tokens: Optional[List[List[str]]] = None,
        prediction_tokens: Optional[List[List[str]]] = None,
        target_tokens: Optional[List[List[List[str]]]] = None,
        **kwargs,
    ) -> float:
        import src.metrics.gleu as gleu

        gleu_calculator = gleu.GLEU(n=4)
        gleu_calculator.load_inputs(original_tokens)
        gleu_calculator.load_references(target_tokens)
        return gleu_calculator.run_iterations(hypotheses=prediction_tokens, num_iterations=500, per_sent=False)


class UpdateRougeMetric(CustomMetric):
    def __init__(self):
        super().__init__(name="update_rouge")

    def evaluate(
        self,
        originals: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
        targets: Optional[List[List[str]]] = None,
        original_tokens: Optional[List[List[str]]] = None,
        prediction_tokens: Optional[List[List[str]]] = None,
        target_tokens: Optional[List[List[List[str]]]] = None,
        **kwargs,
    ) -> float:
        import src.metrics.update_rouge as update_rouge

        scores_1 = []
        scores_2 = []
        scores_L = []

        target_length = len(targets[0])
        for i in range(target_length):
            # print(update_rouge.update_rouge(originals, [t[i] for t in targets], predictions))
            scores = update_rouge.update_rouge(originals, [t[i] for t in targets], predictions)
            scores_1.append(scores["update_rouge1"])
            scores_2.append(scores["update_rouge2"])
            scores_L.append(scores["update_rougeLsum"])
        return {
            "update_rouge1": sum(scores_1) / len(scores_1),
            "update_rouge2": sum(scores_2) / len(scores_2),
            "update_rougeLsum": sum(scores_L) / len(scores_L),
        }


class iBLEUMetric(CustomMetric):
    """Given a candidate sentence O, human references R and input text I,
    iBLEU is defined as: iBLEU = α × BLEU(O, R) (1) −(1 − α) × BLEU(O, I).
    Originally proposed by (Sun and Zhou, 2012)."""

    def __init__(self, alpha=0.9):
        super().__init__(name="ibleu")
        self.alpha = alpha

    def evaluate(
        self,
        originals: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
        targets: Optional[List[List[str]]] = None,
        original_tokens: Optional[List[List[str]]] = None,
        prediction_tokens: Optional[List[List[str]]] = None,
        target_tokens: Optional[List[List[List[str]]]] = None,
        **kwargs,
    ) -> float:
        bleu_scorer = load_metric("bleu")
        ref_scores = bleu_scorer.compute(predictions=prediction_tokens, references=target_tokens)
        ref_bleu = ref_scores["bleu"]

        wrapped_originals = [[original_input] for original_input in original_tokens]
        input_scores = bleu_scorer.compute(predictions=prediction_tokens, references=wrapped_originals)
        input_bleu = input_scores["bleu"]

        return self.alpha * ref_bleu - (1 - self.alpha) * input_bleu
