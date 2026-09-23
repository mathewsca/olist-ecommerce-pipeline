"""
FabricaIA - Olist ETL Pipeline DAG

Orchestrates the full Olist Brazilian E-Commerce ETL: download from Kaggle
(or reuse CSVs already in data/raw/) -> load into the Postgres "raw" schema
-> clean into "staging" -> build the star-schema "dw" -> validate.

Requires the Airflow connection "olist_dw_postgres" pointing at the olist_dw
database
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup

try:
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import PythonOperator

default_args = {
    "owner": "fabricaia",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "olist_etl_pipeline",
    default_args=default_args,
    description="Olist Brazilian E-Commerce ETL: raw -> staging -> star-schema DW",
    schedule=timedelta(days=1),
    catchup=False,
    tags=["etl", "olist", "fabricaia"],
)


def _ensure_project_path():
    import os
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if "/opt/airflow" not in sys.path:
        sys.path.append("/opt/airflow")
    try:
        os.chdir(project_root)
    except Exception:
        pass
    return project_root


def check_or_download_kaggle_data(**context):
    """Download the Olist dataset from Kaggle, unless the CSVs are already present."""
    _ensure_project_path()
    from src.etl.kaggle_ingestion import KaggleIngestion

    dest_dir = Variable.get("kaggle_dataset_path", default_var="data/raw")
    ingestion = KaggleIngestion(
        dataset_slug=Variable.get("kaggle_dataset_slug", default_var="olistbr/brazilian-ecommerce")
    )
    files = ingestion.download_dataset(dest_dir=dest_dir)
    return {"dest_dir": dest_dir, "files": files}


def init_schemas(**context):
    """Create the raw/staging/dw schemas and (re)create the DW tables."""
    _ensure_project_path()
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    conn_id = Variable.get("dw_postgres_conn_id", default_var="olist_dw_postgres")
    hook = PostgresHook(postgres_conn_id=conn_id)
    for path in (
        "sql/raw/create_raw_schema.sql",
        "sql/staging/create_staging_schema.sql",
        "sql/dw/create_dw_schema.sql",
    ):
        with open(path, "r", encoding="utf-8") as sql_file:
            hook.run(sql_file.read())


def _make_raw_load_callable(table_name: str, csv_filename: str):
    def _load(**context):
        _ensure_project_path()
        from src.etl.db import get_engine
        from src.etl.raw_loader import RawLoader

        loader = RawLoader(get_engine(), source_dir=Variable.get("kaggle_dataset_path", default_var="data/raw"))
        loader.ensure_schema()
        return loader.load_csv_to_raw(csv_filename, table_name)

    return _load


def build_staging_tables(**context):
    _ensure_project_path()
    from src.etl.db import get_engine
    from src.etl.staging_transform import StagingTransformer

    transformer = StagingTransformer(get_engine())
    return transformer.build_all()


def build_dw_dimensions(**context):
    _ensure_project_path()
    from src.etl.db import get_engine
    from src.etl.dw_builder import DWBuilder

    builder = DWBuilder(get_engine())
    return builder.build_all_dimensions()


def build_dw_facts(**context):
    _ensure_project_path()
    from src.etl.db import get_engine
    from src.etl.dw_builder import DWBuilder

    builder = DWBuilder(get_engine())
    return builder.build_all_facts()


def validate_dw(**context):
    _ensure_project_path()
    from src.etl.db import get_engine
    from src.etl.validations import DWValidator

    result = DWValidator(get_engine()).run_all()
    if not result.passed:
        raise ValueError(f"DW validation failed: {result.failures}")
    return {"checks_run": result.checks_run}


download_task = PythonOperator(
    task_id="check_or_download_kaggle_data",
    python_callable=check_or_download_kaggle_data,
    dag=dag,
)

init_schemas_task = PythonOperator(
    task_id="init_schemas",
    python_callable=init_schemas,
    dag=dag,
)

RAW_TABLES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

with TaskGroup(group_id="load_raw", dag=dag) as load_raw_group:
    for table_name, csv_filename in RAW_TABLES.items():
        PythonOperator(
            task_id=f"load_raw_{table_name}",
            python_callable=_make_raw_load_callable(table_name, csv_filename),
            dag=dag,
        )

staging_task = PythonOperator(
    task_id="build_staging_tables",
    python_callable=build_staging_tables,
    dag=dag,
)

dw_dimensions_task = PythonOperator(
    task_id="build_dw_dimensions",
    python_callable=build_dw_dimensions,
    dag=dag,
)

dw_facts_task = PythonOperator(
    task_id="build_dw_facts",
    python_callable=build_dw_facts,
    dag=dag,
)

validate_task = PythonOperator(
    task_id="validate_dw",
    python_callable=validate_dw,
    dag=dag,
)

cleanup_task = BashOperator(
    task_id="cleanup_temp_files",
    bash_command='find /tmp -name "*.tmp" -user $(whoami) -delete 2>/dev/null || true',
    dag=dag,
)

(
    download_task
    >> init_schemas_task
    >> load_raw_group
    >> staging_task
    >> dw_dimensions_task
    >> dw_facts_task
    >> validate_task
    >> cleanup_task
)
