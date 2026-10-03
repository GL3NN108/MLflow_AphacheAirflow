from __future__ import annotations

import os
from pathlib import Path

import httpx
import mlflow
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.train import train_and_log

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080").rstrip("/")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
AIRFLOW_DAG_ID = os.getenv("AIRFLOW_DAG_ID", "train_iris_model")
MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
IRIS_LABELS = ["setosa", "versicolor", "virginica"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="model-ops")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., ge=0)
    sepal_width: float = Field(..., ge=0)
    petal_length: float = Field(..., ge=0)
    petal_width: float = Field(..., ge=0)


def _airflow_client() -> httpx.Client:
    return httpx.Client(
        base_url=AIRFLOW_BASE_URL,
        auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
        timeout=8.0,
    )


def _mlflow_ok() -> bool:
    try:
        httpx.get(f"{MLFLOW_TRACKING_URI.rstrip('/')}/health", timeout=3.0).raise_for_status()
        return True
    except Exception:
        return False


def _airflow_ok() -> bool:
    try:
        with _airflow_client() as client:
            r = client.get("/health")
            return r.status_code < 500
    except Exception:
        return False


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
def status():
    return {
        "mlflow": {"ok": _mlflow_ok(), "url": MLFLOW_TRACKING_URI},
        "airflow": {"ok": _airflow_ok(), "url": AIRFLOW_BASE_URL, "dag_id": AIRFLOW_DAG_ID},
        "model_name": MODEL_NAME,
    }


@app.get("/api/runs")
def list_runs():
    if not _mlflow_ok():
        return {"runs": [], "error": "MLflow is not reachable"}
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("iris-classifier")
    if experiment is None:
        return {"runs": []}
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=12,
    )
    payload = []
    for run in runs:
        payload.append(
            {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "accuracy": run.data.metrics.get("accuracy"),
                "f1_macro": run.data.metrics.get("f1_macro"),
                "params": run.data.params,
            }
        )
    return {"runs": payload}


@app.get("/api/dag-runs")
def dag_runs():
    try:
        with _airflow_client() as client:
            r = client.get(
                f"/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns",
                params={"order_by": "-execution_date", "limit": 8},
            )
            r.raise_for_status()
            items = []
            for item in r.json().get("dag_runs", []):
                items.append(
                    {
                        "dag_run_id": item.get("dag_run_id"),
                        "state": item.get("state"),
                        "start_date": item.get("start_date"),
                        "end_date": item.get("end_date"),
                    }
                )
            return {"dag_runs": items}
    except Exception as exc:
        return {"dag_runs": [], "error": str(exc)}


@app.post("/api/train/airflow")
def trigger_airflow_train():
    try:
        with _airflow_client() as client:
            client.patch(
                f"/api/v1/dags/{AIRFLOW_DAG_ID}",
                json={"is_paused": False},
            )
            r = client.post(
                f"/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns",
                json={"conf": {}},
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Airflow trigger failed: {exc}") from exc


@app.post("/api/train/direct")
def train_direct():
    if not _mlflow_ok():
        raise HTTPException(status_code=503, detail="MLflow is not reachable")
    try:
        return train_and_log()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict")
def predict(body: PredictRequest):
    if not _mlflow_ok():
        raise HTTPException(status_code=503, detail="MLflow is not reachable")
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise HTTPException(status_code=404, detail="No registered model yet. Run training first.")
    latest = max(versions, key=lambda version: int(version.version))
    model_uri = f"models:/{MODEL_NAME}/{latest.version}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Could not load the registered model.") from exc
    features = [[
        body.sepal_length,
        body.sepal_width,
        body.petal_length,
        body.petal_width,
    ]]
    pred = int(model.predict(features)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[0]
        proba = {label: float(score) for label, score in zip(IRIS_LABELS, scores)}
    return {
        "label": IRIS_LABELS[pred],
        "class_id": pred,
        "probabilities": proba,
        "model_uri": model_uri,
    }
