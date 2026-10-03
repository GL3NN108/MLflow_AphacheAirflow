"""Train a tiny iris classifier and log it to MLflow."""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")


def train_and_log(n_estimators: int = 80, max_depth: int = 4) -> dict:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("iris-classifier")

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
    )

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        accuracy = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="macro"))

        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        mlflow.log_metrics({"accuracy": accuracy, "f1_macro": f1})
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=x_train[:2],
        )

        return {
            "run_id": run.info.run_id,
            "accuracy": accuracy,
            "f1_macro": f1,
            "model_name": MODEL_NAME,
        }


if __name__ == "__main__":
    result = train_and_log()
    print(result)
