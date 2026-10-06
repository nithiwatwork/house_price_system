from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from airflow import DAG
    from airflow.decorators import task
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="house_price_monitoring_pipeline",
        default_args=default_args,
        description="Continuous Monitoring & Automated Retraining DAG",
        schedule_interval="@daily",
        catchup=False,
        tags=["monitoring", "drift", "auto-retrain"],
    ) as dag:

        @task
        def task_check_drift():
            from monitoring.drift_detector import detect_drift
            drift_report = detect_drift()
            return drift_report

        @task.short_circuit
        def task_needs_retraining(drift_report: dict) -> bool:
            """Short circuit if drift is within acceptable limits."""
            needs_retrain = drift_report.get("needs_retrain", False)
            print(f"[Monitoring DAG] Drift check result: needs_retrain={needs_retrain}")
            return needs_retrain

        trigger_retraining = TriggerDagRunOperator(
            task_id="trigger_house_price_training",
            trigger_dag_id="house_price_training_pipeline",
            wait_for_completion=False,
        )

        report = task_check_drift()
        should_retrain = task_needs_retraining(report)
        should_retrain >> trigger_retraining
