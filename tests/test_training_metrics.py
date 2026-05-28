from __future__ import annotations

import unittest

import numpy as np

from training import metrics


class _FakeSquadMetric:
    def __init__(self) -> None:
        self.predictions: list[dict] = []
        self.references: list[dict] = []

    def compute(self, *, predictions: list[dict], references: list[dict]) -> dict[str, float]:
        self.predictions = predictions
        self.references = references
        reference_by_id = {str(ref["id"]): ref for ref in references}

        exact_scores = []
        f1_scores = []
        for prediction in predictions:
            ref = reference_by_id[str(prediction["id"])]
            answer_texts = list(ref["answers"]["text"])
            exact, f1 = metrics.compute_exact_and_f1(
                prediction["prediction_text"],
                answer_texts,
            )
            exact_scores.append(exact)
            f1_scores.append(f1)

        return {
            "exact_match": 100.0 * sum(exact_scores) / len(exact_scores),
            "f1": 100.0 * sum(f1_scores) / len(f1_scores),
        }


class ComputeMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_metric = metrics._SQUAD_METRIC
        self.fake_metric = _FakeSquadMetric()
        metrics._SQUAD_METRIC = self.fake_metric

    def tearDown(self) -> None:
        metrics._SQUAD_METRIC = self._previous_metric

    def test_sliding_window_features_are_grouped_by_example_id(self) -> None:
        result = metrics.compute_metrics(
            {
                "start_logits": np.array(
                    [
                        [0.0, 5.0, 0.0],
                        [0.0, 8.0, 0.0],
                    ]
                ),
                "end_logits": np.array(
                    [
                        [0.0, 5.0, 0.0],
                        [0.0, 8.0, 0.0],
                    ]
                ),
                "offset_mapping": [
                    [None, (0, 5), None],
                    [None, (6, 10), None],
                ],
                "contexts": ["wrong gold", "wrong gold"],
                "example_ids": ["0", "0"],
                "references": [
                    {
                        "id": "0",
                        "answers": {"text": ["gold"], "answer_start": [6]},
                    }
                ],
            }
        )

        self.assertEqual(result, {"exact_match": 100.0, "f1": 100.0})
        self.assertEqual(
            self.fake_metric.predictions,
            [{"id": "0", "prediction_text": "gold"}],
        )

    def test_missing_example_prediction_is_scored_as_empty_answer(self) -> None:
        metrics.compute_metrics(
            {
                "start_logits": np.array([[0.0, 5.0, 0.0]]),
                "end_logits": np.array([[0.0, 5.0, 0.0]]),
                "offset_mapping": [[None, (0, 4), None]],
                "contexts": ["gold"],
                "example_ids": ["0"],
                "references": [
                    {
                        "id": "0",
                        "answers": {"text": ["gold"], "answer_start": [0]},
                    },
                    {
                        "id": "1",
                        "answers": {"text": ["missing"], "answer_start": [0]},
                    },
                ],
            }
        )

        self.assertEqual(
            self.fake_metric.predictions,
            [
                {"id": "0", "prediction_text": "gold"},
                {"id": "1", "prediction_text": ""},
            ],
        )


if __name__ == "__main__":
    unittest.main()
