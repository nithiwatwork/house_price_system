
from datetime import datetime, timedelta
import os
import sys

# Add project root to sys.path for Airflow workers
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from airflow import DAG
    from airflow.decorators import task
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="house_price_training_pipeline",
        default_args=default_args,
        description="Production House Price Valuation Training DAG",
        schedule_interval=None,  # Triggered manually or by monitoring DAG
        catchup=False,
        tags=["mlops", "regression", "king_county"],
    ) as dag:

        @task
        def task_ingest():
            from src.ingestion import ingest_data
            return ingest_data()

        @task
        def task_validate(split_info: dict):
            from src.validation import validate_file
            validate_file(split_info["train_path"])
            validate_file(split_info["test_path"])
            return split_info

        @task
        def task_train(split_info: dict):
            from src.train import train_models
            return train_models(train_path=split_info["train_path"], val_path=split_info["val_path"])

        @task
        def task_evaluate(split_info: dict, train_results: dict):
            from src.evaluate import evaluate_test_set
            return evaluate_test_set(test_path=split_info["test_path"])

        @task
        def task_bless_and_push(eval_report: dict, train_results: dict):
            from src.registry import bless_and_register_model
            run_id = train_results.get("champion", {}).get("run_id", "airflow_run")
            return bless_and_register_model(evaluation_report=eval_report, run_id=run_id)

        # DAG Workflow definition
        info = task_ingest()
        valid_info = task_validate(info)
        training_res = task_train(valid_info)
        eval_res = task_evaluate(valid_info, training_res)
        blessed = task_bless_and_push(eval_res, training_res)
