from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from training.dataset import prepare_train_features
from training.flatten_squad import flatten_squad_records
from training import metrics as qa_metrics
from training.trainer import _is_better_metric


class MockTokenizedBatch(dict):
    def sequence_ids(self, batch_index: int):
        return self["sequence_ids"][batch_index]


class MockTokenizer:
    padding_side = "right"
    cls_token_id = 101

    def __call__(self, text, text_pair, truncation, max_length, stride, return_overflowing_tokens, return_offsets_mapping, padding):
        return MockTokenizedBatch(
            {
                "input_ids": [[101, 11, 102, 21, 22, 23, 102]],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1]],
                "offset_mapping": [[(0, 0), (0, 0), (0, 0), (0, 2), (3, 5), (6, 8), (0, 0)]],
                "overflow_to_sample_mapping": [0],
                "sequence_ids": [[None, 0, None, 1, 1, 1, None]],
            }
        )


class TestPipeline(unittest.TestCase):
    def test_flatten_squad_records(self):
        raw = {
            "data": [
                {
                    "title": "demo",
                    "paragraphs": [
                        {
                            "context": "Ha Noi la thu do Viet Nam.",
                            "qas": [
                                {
                                    "id": "q1",
                                    "question": "Thu do nuoc nao?",
                                    "answers": [{"text": "Viet Nam", "answer_start": 14}],
                                    "is_impossible": False,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        rows = flatten_squad_records(raw)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "q1")
        self.assertEqual(rows[0]["answers"]["text"], ["Viet Nam"])
        self.assertEqual(rows[0]["answers"]["answer_start"], [14])

    def test_prepare_train_features_answer_in_context(self):
        examples = {
            "question": ["Q?"],
            "context": ["aa bb cc"],
            "answers": [{"text": ["bb"], "answer_start": [3]}],
            "is_impossible": [False],
        }
        output = prepare_train_features(
            examples=examples,
            tokenizer=MockTokenizer(),
            use_vietnamese_segmentation=False,
            max_length=32,
            doc_stride=8,
        )
        self.assertEqual(output["start_positions"][0], 4)
        self.assertEqual(output["end_positions"][0], 4)

    def test_prepare_train_features_unanswerable_uses_cls(self):
        examples = {
            "question": ["Q?"],
            "context": ["aa bb cc"],
            "answers": [{"text": [], "answer_start": []}],
            "is_impossible": [True],
        }
        output = prepare_train_features(
            examples=examples,
            tokenizer=MockTokenizer(),
            use_vietnamese_segmentation=False,
            max_length=32,
            doc_stride=8,
        )
        self.assertEqual(output["start_positions"][0], 0)
        self.assertEqual(output["end_positions"][0], 0)

    def test_compute_metrics_handles_empty_reference_answers(self):
        class DummyMetric:
            def compute(self, predictions, references):
                self.last_predictions = predictions
                self.last_references = references
                return {"exact_match": 0.0, "f1": 0.0}

        eval_preds = {
            "start_logits": np.array([[0.1, 2.0, 0.3]]),
            "end_logits": np.array([[0.1, 1.8, 0.2]]),
            "offset_mapping": [[(0, 0), (0, 1), (1, 2)]],
            "contexts": ["ab"],
            "example_ids": ["0"],
            "references": [{"id": "0", "answers": {"text": [], "answer_start": []}}],
            "max_answer_length": 30,
        }

        dummy = DummyMetric()
        with patch.object(qa_metrics, "_load_squad_metric", return_value=dummy):
            result = qa_metrics.compute_metrics(eval_preds)

        self.assertIn("exact_match", result)
        self.assertIn("f1", result)
        self.assertEqual(dummy.last_references[0]["answers"]["text"], [""])
        self.assertEqual(dummy.last_references[0]["answers"]["answer_start"], [0])

    def test_best_model_selection_prefers_larger_metric(self):
        self.assertTrue(_is_better_metric(0.9, 0.8, greater_is_better=True))
        self.assertFalse(_is_better_metric(0.7, 0.8, greater_is_better=True))
        self.assertTrue(_is_better_metric(0.7, 0.8, greater_is_better=False))


if __name__ == "__main__":
    unittest.main()
