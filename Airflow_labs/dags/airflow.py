from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.lab import load_data, split_data, train_model, predict_and_evaluate

# ── DAG default args ─────────────────────────────────────────────────
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# ── Wrapper callables (pull XCom as native Python objects) ───────────
def _split_data(**context):
    ti = context["ti"]
    data = ti.xcom_pull(task_ids="load_data")
    return split_data(data)


def _train_model(**context):
    ti = context["ti"]
    split = ti.xcom_pull(task_ids="split_data")
    return train_model(split)


def _predict_and_evaluate(**context):
    ti = context["ti"]
    split = ti.xcom_pull(task_ids="split_data")
    model_path = ti.xcom_pull(task_ids="train_model")
    return predict_and_evaluate(split, model_path)


# ── DAG definition ───────────────────────────────────────────────────
with DAG(
    dag_id="wine_classification_pipeline",
    default_args=default_args,
    description="Wine dataset ML pipeline using GradientBoostingClassifier",
    start_date=datetime(2025, 2, 10),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "wine", "classification"],
) as dag:

    task_load = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    task_split = PythonOperator(
        task_id="split_data",
        python_callable=_split_data,
    )

    task_train = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    task_evaluate = PythonOperator(
        task_id="predict_and_evaluate",
        python_callable=_predict_and_evaluate,
    )

    # ── Pipeline ordering ────────────────────────────────────────────
    task_load >> task_split >> task_train >> task_evaluate

if __name__ == "__main__":
    dag.test()