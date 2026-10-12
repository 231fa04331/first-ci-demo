
import json
import unittest
from pathlib import Path

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_outputs_exist(self):
        for name in [
            "student_result_model.pkl",
            "metrics.json",
            "student_performance_raw.csv"
        ]:
            self.assertTrue(Path(name).exists(), f"Missing {name}")

    def test_metrics(self):
        metrics = json.loads(Path("metrics.json").read_text())
        self.assertTrue(0 <= metrics["accuracy"] <= 1)
        self.assertTrue(0 <= metrics["precision"] <= 1)
        self.assertTrue(0 <= metrics["recall"] <= 1)
        self.assertTrue(0 <= metrics["f1_score"] <= 1)

    def test_prediction(self):
        bundle = joblib.load("student_result_model.pkl")
        model = bundle["model"]
        sample = pd.read_csv("student_performance_raw.csv").head(1)
        sample = sample[bundle["features"]]
        result = model.predict(sample)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
