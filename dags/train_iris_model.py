from datetime import datetime
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

SRC = Path("/opt/airflow/src")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _train(**context):
    from train import train_and_log

    result = train_and_log()
    context["ti"].xcom_push(key="train_result", value=result)


with DAG(
    dag_id="train_iris_model",
    description="Train iris model and log metrics/artifacts to MLflow",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=["mlflow", "training"],
) as dag:
    PythonOperator(
        task_id="train_and_log",
        python_callable=_train,
    )
