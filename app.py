
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path("student_result_model.pkl")


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file is missing. Train the model first.")
    return joblib.load(MODEL_PATH)


@app.get("/")
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "student-performance-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    try:
        bundle = load_bundle()
        features = bundle["features"]
        missing = [name for name in features if name not in data]

        if missing:
            return jsonify({
                "error": "Missing feature values",
                "missing_fields": missing
            }), 400

        sample = pd.DataFrame(
            [{name: data[name] for name in features}],
            columns=features
        )

        prediction = bundle["model"].predict(sample)[0]

        return jsonify({
            "prediction": str(prediction),
            "target_column": bundle["target_column"]
        })
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
