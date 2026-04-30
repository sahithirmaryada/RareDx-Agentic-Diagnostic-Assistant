import unittest

from scripts.calibrate import (
    calibrate_examples,
    example_from_row,
    reliability,
)


class CalibrationTests(unittest.TestCase):
    def test_example_from_row_extracts_top_candidate_features(self):
        row = {
            "case_id": "RD001",
            "strategy": "fusion_60_40",
            "difficulty": "hard",
            "status": "ok",
            "top1_correct": "1",
            "top_score": "1.8",
            "num_candidates": "2",
            "graph_candidate_count": "1",
            "citation_candidate_count": "2",
            "icd10_candidate_count": "1",
            "top_disease": "Pompe disease",
            "top_orphacode": "ORPHA:365",
            "ranked_candidates_json": """
            [
              {
                "rank": 1,
                "score": 1.8,
                "graph_path_count": 2,
                "citation_count": 3,
                "icd10": "E74.0",
                "evidence_badges": ["Neo4j Path Found", "PubMed Grounded"],
                "matched_symptoms": ["Muscle weakness"],
                "matched_genes": ["GAA"]
              },
              {"rank": 2, "score": 1.2}
            ]
            """,
        }

        example = example_from_row(1, row)

        self.assertEqual(example.label, 1)
        self.assertEqual(example.top_score, 1.8)
        self.assertAlmostEqual(example.score_margin, 0.6)
        self.assertEqual(example.features[6], 2.0)
        self.assertEqual(example.features[7], 3.0)
        self.assertEqual(example.features[11], 1.0)

    def test_calibrate_examples_returns_predictions_and_metrics(self):
        examples = []
        for index, label in enumerate([1, 1, 1, 0, 0, 1], start=1):
            row = {
                "case_id": f"RD{index:03d}",
                "strategy": "fusion_60_40",
                "difficulty": "easy" if label else "hard",
                "status": "ok",
                "top1_correct": str(label),
                "top_score": str(2.0 if label else 0.5),
                "num_candidates": "5",
                "graph_candidate_count": "2",
                "citation_candidate_count": "5",
                "icd10_candidate_count": "4",
                "top_disease": "Disease",
                "top_orphacode": "ORPHA:1",
                "ranked_candidates_json": "[]",
            }
            examples.append(example_from_row(index, row))

        calibration = calibrate_examples(
            examples,
            folds=3,
            bins=3,
            epochs=20,
            learning_rate=0.05,
            l2=0.1,
            seed=7,
        )

        self.assertEqual(len(calibration["predictions"]), len(examples))
        self.assertIn("brier_score", calibration["metrics"])
        self.assertIn("expected_calibration_error", calibration["metrics"])
        self.assertIn("model", calibration)

    def test_reliability_includes_probability_one_in_last_bin(self):
        bins = reliability([1], [1.0], bins=5)

        self.assertEqual(bins[-1]["count"], 1)
        self.assertEqual(bins[-1]["observed_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
