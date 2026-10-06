"""DAG หลัก — ไปป์ไลน์เทรนโมเดลแบบครบวงจร ตั้งชื่อขั้นตอนตามคอมโพเนนต์ของ TFX

ชุดข้อมูล King County House Sales: เป้าหมายคือทายราคาบ้าน (House Price Regression)

ExampleGen -> StatisticsGen -> SchemaGen -> ExampleValidator -> Transform
          -> Trainer -> Evaluator -> (ด่านอนุมัติ 3 เงื่อนไข) -> Pusher

สิ่งที่ไฟล์นี้แสดงให้เห็น
  - TaskFlow API (@dag, @task) ซึ่งทำให้ค่าที่ return กลายเป็นเส้นเชื่อมของ DAG โดยอัตโนมัติ
  - การแตกสาขาด้วย @task.branch เพื่อทำด่าน "blessing" แบบเดียวกับ Evaluator ของ TFX
  - trigger_rule สำหรับ task ที่ต้องรันไม่ว่าสาขาไหนจะถูกเลือก
  - retries และ params ที่ปรับได้ตอนสั่งรัน โดยไม่ต้องแก้โค้ด
  - task ใหม่ของกลุ่ม: data_card ซึ่งสรุปโปรไฟล์ข้อมูลขนานกับ trainer ได้เพราะไม่พึ่งโมเดล
"""
from __future__ import annotations

import os
import sys

# เพิ่ม dags directory และ project root ใน sys.path
_dags_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_dags_dir, ".."))
for _p in [_dags_dir, _project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import pendulum
except ImportError:
    from datetime import datetime, timedelta, timezone

    class _PendulumMock:
        @staticmethod
        def datetime(year, month, day, tz="UTC"):
            return datetime(year, month, day, tzinfo=timezone.utc)

        @staticmethod
        def duration(seconds=0, minutes=0):
            return timedelta(seconds=seconds, minutes=minutes)

    pendulum = _PendulumMock()  # type: ignore

try:
    from airflow.providers.standard.operators.empty import EmptyOperator
    from airflow.sdk import Param, dag, task
    from airflow.sdk.exceptions import AirflowFailException
    from airflow.task.trigger_rule import TriggerRule
    AIRFLOW_AVAILABLE = True
except ImportError:
    try:
        from airflow.decorators import dag, task
        from airflow.exceptions import AirflowFailException
        from airflow.models.param import Param
        from airflow.operators.empty import EmptyOperator
        from airflow.utils.trigger_rule import TriggerRule
        AIRFLOW_AVAILABLE = True
    except ImportError:
        AIRFLOW_AVAILABLE = False

if not AIRFLOW_AVAILABLE:
    class _DummyDag:
        pass

    def dag(*args, **kwargs):
        def decorator(f):
            return lambda *a, **kw: _DummyDag()
        return decorator

    def task(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

    task.branch = task  # type: ignore

    class Param:  # type: ignore
        def __init__(self, default=None, **kwargs):
            self.default = default

    class EmptyOperator:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class AirflowFailException(Exception):
        pass

    class TriggerRule:  # type: ignore
        NONE_FAILED_MIN_ONE_SUCCESS = "none_failed_min_one_success"

from ml import steps

DEFAULT_ARGS = {
    "owner": "mlops-cp413008",
    "retries": 2,                                  # งานที่ล้มเพราะเหตุชั่วคราวให้ลองใหม่เอง
    "retry_delay": pendulum.duration(seconds=30),
}


@dag(
    dag_id="house_price_training_pipeline",
    description="ไปป์ไลน์เทรนโมเดล King County House Price ตั้งแต่ข้อมูลดิบจนถึงการอนุมัติขึ้นใช้งาน",
    schedule="0 2 * * 1",                          # ทุกวันจันทร์ ตี 2 (UTC) หรือสั่ง Trigger ด้วยตนเอง
    # นาที  ชั่วโมง  วันที่   เดือน   วันในสัปดาห์
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    catchup=False,                                 # ไม่ต้องย้อนรันอดีตทั้งหมดตอนเปิดใช้ครั้งแรก
    max_active_runs=1,                             # กันสอง run เขียนทับโมเดลพร้อมกัน
    default_args=DEFAULT_ARGS,
    tags=["mlops", "training", "lab11", "house_price", "regression"],
    params={
        # ปรับค่าเหล่านี้ได้จากหน้าเว็บตอนกด Trigger DAG w/ config โดยไม่ต้องแก้โค้ด
        "data_file": Param(
            "kc_house_data.csv", type="string",
            description="ชื่อไฟล์ข้อมูลในโฟลเดอร์ data/ หรือ data/raw/ ใช้สลับไปไฟล์ข้อมูลเสียเพื่อสาธิตด่านตรวจ",
        ),
        "max_iter": Param(200, type="integer", minimum=1, description="จำนวนรอบการเทรน"),
        "fail_on_anomaly": Param(True, type="boolean", description="ให้หยุด pipeline เมื่อข้อมูลผิดปกติ"),
    },
)
def house_price_training_pipeline():
    @task(task_id="example_gen")
    def t_example_gen(**context) -> dict:
        return steps.example_gen(context["run_id"], context["params"]["data_file"])

    @task(task_id="statistics_gen")
    def t_statistics_gen(examples: dict, **context) -> str:
        return steps.statistics_gen(context["run_id"], examples)

    @task(task_id="schema_gen")
    def t_schema_gen(statistics_path: str, **context) -> str:
        return steps.schema_gen(context["run_id"], statistics_path)

    @task(task_id="example_validator")
    def t_example_validator(examples: dict, schema_path: str, **context) -> dict:
        result = steps.example_validator(examples, schema_path)
        for a in result["anomalies"]:
            print("ANOMALY:", a)
        if not result["ok"] and context["params"]["fail_on_anomaly"]:
            # AirflowFailException = ล้มทันทีโดยไม่ retry เพราะข้อมูลเสียลองใหม่กี่ครั้งก็เสียเหมือนเดิม
            raise AirflowFailException(f"ข้อมูลไม่ผ่านการตรวจ: {result['anomalies']}")
        return result

    @task(task_id="transform")
    def t_transform(examples: dict, **context) -> str:
        return steps.transform(context["run_id"], examples)

    @task(task_id="data_card")                        # task ใหม่ของกลุ่ม — ขนานกับ trainer
    def t_data_card(examples: dict, **context) -> str:
        return steps.data_card_gen(context["run_id"], examples)

    @task(task_id="trainer")
    def t_trainer(examples: dict, transform_path: str, **context) -> str:
        return steps.trainer(context["run_id"], examples, transform_path, context["params"]["max_iter"])

    @task(task_id="evaluator")
    def t_evaluator(examples: dict, model_path: str, **context) -> dict:
        metrics = steps.evaluator(context["run_id"], examples, model_path)
        print("metrics:", metrics)
        return metrics

    @task.branch(task_id="blessing_gate")
    def t_blessing_gate(metrics: dict) -> str:
        """เลือกเส้นทางถัดไปจากผลการประเมิน — นี่คือด่านอนุมัติก่อนขึ้นใช้งาน"""
        if metrics["blessed"]:
            return "pusher"
        print("โมเดลไม่ผ่านด่าน:", metrics)
        return "skip_push"

    @task(task_id="pusher")
    def t_pusher(model_path: str, metrics: dict, schema_path: str, **context) -> str:
        return steps.pusher(context["run_id"], model_path, metrics, schema_path)

    skip_push = EmptyOperator(task_id="skip_push")

    @task(task_id="report", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def t_report(metrics: dict, **context) -> None:
        """สรุปผลของการรัน — ต้องรันเสมอไม่ว่าจะ push หรือไม่

        trigger_rule ค่าเริ่มต้นคือ all_success ซึ่งจะทำให้ task นี้ถูกข้าม
        เพราะสาขาที่ไม่ถูกเลือกจะมีสถานะ skipped จึงต้องเปลี่ยนเป็น none_failed_min_one_success
        (งาน 4.2 ใช้ช่องนี้สาธิตความต่างของ trigger_rule)
        """
        status = "PUSHED" if metrics["blessed"] else "NOT PUSHED"
        print(f"run_id={context['run_id']} | {status} | mape={metrics['mape']} "
              f"| rmse={metrics['rmse']} | r2={metrics['r2']} "
              f"| baseline_mape={metrics['baseline_mape']} | improvement={metrics['improvement']}")

    # ------------------------------------------------------------ ผูกลำดับงาน
    examples = t_example_gen()
    stats = t_statistics_gen(examples)
    schema = t_schema_gen(stats)
    validation = t_example_validator(examples, schema)

    transform_path = t_transform(examples)
    data_card = t_data_card(examples)  # noqa: F841     # ขึ้นกับตัวอย่างข้อมูลเท่านั้น จึงรันขนานกับ trainer
    model_path = t_trainer(examples, transform_path)
    metrics = t_evaluator(examples, model_path)
    gate = t_blessing_gate(metrics)
    push = t_pusher(model_path, metrics, schema)
    report = t_report(metrics)

    # ต้องผ่านด่านตรวจข้อมูลก่อนจึงจะเริ่มแปลงข้อมูลได้
    # เส้นนี้ต้องเขียนเอง เพราะ transform ไม่ได้รับค่าจาก validation โดยตรง
    validation >> transform_path

    gate >> [push, skip_push]
    [push, skip_push] >> report


house_price_training_pipeline()
