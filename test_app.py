
import unittest
import pandas as pd
from app import app, load_bundle


class TestPredictionAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_prediction(self):
        bundle = load_bundle()
        data = pd.read_csv("student_performance_raw.csv")
        sample = data[bundle["features"]].iloc[0].to_dict()

        response = self.client.post("/predict", json=sample)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.get_json())

    def test_missing_fields(self):
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
