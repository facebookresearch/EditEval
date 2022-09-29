import unittest
from typing import List
import datasets
from src.evaluator import Evaluator
from src.metrics.custom_metrics import iBLEUMetric
from src.preprocessing import tokenize_text

SARI_TEST_SOURCES: List[str] = ["About 95 species are currently accepted ."]
SARI_TEST_PREDICTIONS: List[str] = ["About 95 you now get in ."]
SARI_TEST_REFERENCES: List[List[str]] = [
    [
        "About 95 species are currently known .",
        "About 95 species are now accepted .",
        "95 species are now accepted .",
    ]
]
TEST_PREDICTION_TOKENS: List[List[str]] = [["hello", "there", "general", "kenobi"]]
TEST_REFERENCE_TOKENS: List[List[str]] = [[["hello", "there", "general", "kenobi"]]]
TEST_ORIGINAL_TOKENS: List[List[str]] = [["hello", "there", "general", "kenobi"]]
DIRECT_SCORE: float = 0.8
EVAL_VAL: float = 0.8
ROUND_TO_THREE: int = 3
ROUND_TO_SIX: int = 6
BERT_TEST_PREDICTIONS: List[str] = ["hello there"]
BERT_TEST_REFERENCES: List[List[str]] = [["hello there"]]
ROUGE_TEST_PREDICTIONS: List[str] = ["hello goodbye", "ankh morpork"]
ROUGE_TEST_REFERENCES: List[List[str]] = [["hello goodbye"], ["ankh morpork"]]
GLEU_TEST_ORIGINAL_TOKENS: List[List[str]] = [["About", "95", "species", "are", "currently", "accepted", "."], ["About", "95", "species", "are", "currently", "accepted", "."]]
GLEU_TEST_PREDICTION_TOKENS: List[List[str]] = [["About", "95", "species", "are", "currently", "known", "."],["About", "95", "you", "now", "get", "in", "."]]
GLEU_TEST_TARGET_TOKENS: List[List[List[str]]] = [
    [
        ["About", "95", "species", "are", "currently", "known", "."],
        ["About", "95", "species", "are", "now", "accepted", "."],
        ["95", "species", "are", "now", "accepted", "."],
    ],
    [
        ["About", "95", "species", "are", "currently", "known", "."],
        ["About", "95", "species", "are", "now", "accepted", "."],
        ["95", "species", "are", "now", "accepted", "."],
    ]
]
GLEU_VAL: float = 0.168629
SARI_VAL: float = 31.35


class TestMetrics(unittest.TestCase):
    """
    Unit tests for EditEval suite of metrics: sari, bleu, ibleu, sacrebleu, bertscore, rouge, gleu.
    """

    def test_sari(self):

        sari = datasets.load_metric("sari")

        eval_sari = round(
            Evaluator().evaluate(
                predictions=SARI_TEST_PREDICTIONS,
                targets=SARI_TEST_REFERENCES,
                inputs=SARI_TEST_SOURCES,
                metrics=["sari"],
            )["sari"],
            ROUND_TO_THREE,
        )

        self.assertEqual(
            eval_sari, SARI_VAL, f"The sari custom metric's value is {eval_sari} while the correct one is {SARI_VAL}"
        )

    def test_bleu(self):

        bleu = datasets.load_metric("bleu")
        huggingface_bleu = round(
            bleu.compute(predictions=TEST_PREDICTION_TOKENS, references=TEST_REFERENCE_TOKENS)["bleu"], ROUND_TO_THREE
        )
        predictions = [" ".join(pred) for pred in TEST_PREDICTION_TOKENS]
        references = [[" ".join(ref_i) for ref_i in ref_example] for ref_example in TEST_REFERENCE_TOKENS]
        eval_bleu = round(
            Evaluator().evaluate(predictions=predictions, targets=references, metrics=["bleu"])["bleu"],
            ROUND_TO_THREE,
        )
        self.assertEqual(
            eval_bleu, huggingface_bleu, f"Huggingface bleu is {huggingface_bleu} while yours is {eval_bleu}"
        )

    def test_ibleu(self):

        predictions = [" ".join(pred) for pred in TEST_PREDICTION_TOKENS]
        originals = [" ".join(o) for o in TEST_ORIGINAL_TOKENS]
        references = [[" ".join(ref_i) for ref_i in ref_example] for ref_example in TEST_REFERENCE_TOKENS]

        direct_metric = iBLEUMetric()
        direct_score = direct_metric.evaluate_safe(
            predictions=predictions,
            originals=originals,
            targets=references,
            prediction_tokens=TEST_PREDICTION_TOKENS,
            target_tokens=TEST_REFERENCE_TOKENS,
            original_tokens=TEST_ORIGINAL_TOKENS,
        )
        self.assertEqual(direct_score, DIRECT_SCORE)

        eval_metrics = round(
            Evaluator().evaluate(predictions=predictions, targets=references, inputs=originals, metrics=["ibleu"])[
                "ibleu"
            ],
            ROUND_TO_THREE,
        )
        self.assertEqual(eval_metrics, EVAL_VAL)

    def test_sacrebleu(self):

        sacrebleu = datasets.load_metric("sacrebleu")
        huggingface_sacrebleu = round(
            sacrebleu.compute(predictions=TEST_PREDICTION_TOKENS, references=TEST_REFERENCE_TOKENS)["score"],
            ROUND_TO_THREE,
        )
        predictions = [" ".join(pred) for pred in TEST_PREDICTION_TOKENS]
        references = [[" ".join(ref_i) for ref_i in ref_example] for ref_example in TEST_REFERENCE_TOKENS]
        eval_sacrebleu = round(
            Evaluator().evaluate(predictions=predictions, targets=references, metrics=["sacrebleu"])["sacrebleu"],
            ROUND_TO_THREE,
        )
        self.assertEqual(
            eval_sacrebleu,
            huggingface_sacrebleu,
            f"Huggingface sari is {huggingface_sacrebleu} while yours is {eval_sacrebleu}",
        )

    def test_bertscore(self):

        bertscore = datasets.load_metric("bertscore")
        huggingface_bert = round(
            bertscore.compute(predictions=BERT_TEST_PREDICTIONS, references=BERT_TEST_REFERENCES, lang="en")["f1"][0],
            ROUND_TO_THREE,
        )

        eval_bert = round(
            Evaluator().evaluate(
                predictions=BERT_TEST_PREDICTIONS, targets=BERT_TEST_REFERENCES, metrics=["bert_score"]
            )["bert_score"],
            ROUND_TO_THREE,
        )
        self.assertEqual(
            eval_bert, huggingface_bert, f"Huggingface bert is {huggingface_bert} while yours is {eval_bert}"
        )

    def test_rouge(self):

        rouge = datasets.load_metric("rouge")
        huggingface_rouge = round(
            rouge.compute(predictions=ROUGE_TEST_PREDICTIONS, references=ROUGE_TEST_REFERENCES)["rouge1"].mid.fmeasure,
            ROUND_TO_THREE,
        )

        eval_rouge = round(
            Evaluator().evaluate(predictions=ROUGE_TEST_PREDICTIONS, targets=ROUGE_TEST_REFERENCES, metrics=["rouge"])[
                "rouge1"
            ],
            ROUND_TO_THREE,
        )
        self.assertEqual(
            eval_rouge, huggingface_rouge, f"Huggingface rouge is {huggingface_rouge} while yours is {eval_rouge}"
        )

    def test_gleu(self):

        predictions = [" ".join(pred) for pred in GLEU_TEST_PREDICTION_TOKENS]
        originals = [" ".join(o) for o in GLEU_TEST_ORIGINAL_TOKENS]
        references = [[" ".join(ref_i) for ref_i in ref_example] for ref_example in GLEU_TEST_TARGET_TOKENS]

        results = Evaluator().evaluate(inputs=originals, predictions=predictions, targets=references, metrics=["gleu"], normalize=True)[
            "gleu"
        ]
        eval_gleu = round(results, ROUND_TO_SIX)
        self.assertEqual(
            eval_gleu, GLEU_VAL, f"The gleu custom metric's value is {results} while the correct one is {GLEU_VAL}"
        )
