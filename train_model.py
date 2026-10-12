
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("student_performance_raw.csv")
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "G3")
MODEL_PATH = Path("student_result_model.pkl")


def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Upload student_performance_raw.csv to the repository root."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' not found. "
            f"Available columns: {df.columns.tolist()}. "
            "Change TARGET_COLUMN to your actual target column."
        )

    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    y_raw = df.pop(TARGET_COLUMN)

    # For the common Portuguese student dataset, G3 is the final grade.
    if TARGET_COLUMN == "G3":
        y = (pd.to_numeric(y_raw, errors="coerce") >= 10).astype(int)
        valid = pd.to_numeric(y_raw, errors="coerce").notna()
        df, y = df.loc[valid], y.loc[valid]
    else:
        y = y_raw

    if y.nunique() < 2:
        raise ValueError("The target must contain at least two classes.")

    X = df.copy()
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1] + 1),
        "target_column": TARGET_COLUMN,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "features": X.columns.tolist(),
        "classes": [str(c) for c in model.classes_]
    }

    joblib.dump({
        "model": model,
        "features": X.columns.tolist(),
        "target_column": TARGET_COLUMN
    }, MODEL_PATH)

    Path("metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("Model saved to", MODEL_PATH)


if __name__ == "__main__":
    train_model()
