"""
FabricaIA - Maintenance DAG: clear old logs, Airflow metadata and temp files.

Airflow keeps every DAG/task run forever by default (dag_run, task_instance, log, xcom,
job... rows in its metadata DB) and writes one log file per task try under logs/ that is
never deleted on its own. Left alone, both grow without bound and slow the webserver/
scheduler down over time. This DAG runs on its own schedule to keep that under control:

  1. clean_airflow_metadata - purge old rows from the metadata DB (raw SQL via PostgresHook).
  2. vacuum_metadata_db      - reclaim the disk space Postgres freed up in step 1.
  3. clean_task_logs         - delete task log files older than the retention window.
  4. clean_temp_files        - sweep /tmp and data/processed/tmp* leftovers.

Retention is controlled by the Airflow Variable "log_retention_days" (default: 30).
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable

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
    "clear_logs_and_metadata",
    default_args=default_args,
    description="Manutencao: limpa metadados antigos do Airflow, logs de tasks e arquivos temporarios",
    schedule=timedelta(days=7),
    catchup=False,
    tags=["maintenance", "fabricaia"],
)

# Metadata tables to purge, with the timestamp column each one is aged by, in
# child-before-parent order (task_instance/xcom/log reference dag_run by FK, so
# dag_run must be deleted last or the DELETE fails on the constraint). Picked from
# Airflow's own metadata schema; left out on purpose: "variable"/"connection"/"dag",
# which are configuration, not run history. Existence is checked via to_regclass,
# since some (celery_taskmeta/tasksetmeta) only exist when CeleryExecutor is configured.
METADATA_TABLES = {
    "log": "dttm",
    "xcom": "timestamp",
    "sla_miss": "timestamp",
    "import_error": "timestamp",
    "task_reschedule": "start_date",
    "celery_taskmeta": "date_done",
    "celery_tasksetmeta": "date_done",
    "task_instance_history": "start_date",
    "task_instance": "start_date",
    "job": "start_date",
    "dag_run": "start_date",
}

METADATA_CONN_ID = "airflow_metadata_db"


def _retention_days() -> int:
    return int(Variable.get("log_retention_days", default_var="30"))


def _metadata_conn():
    """
    Raw psycopg2 connection to Airflow's OWN metadata DB, via a Connection resolved
    from AIRFLOW_CONN_AIRFLOW_METADATA_DB (see docker-compose.yml). Airflow 3 tasks
    cannot use airflow.settings.engine / create_session() directly - that raises
    "Direct database access via the ORM is not allowed in Airflow 3.0" - but a Hook
    resolving its own Connection (the same mechanism used to reach any other Postgres
    database) is the sanctioned way in and works fine.
    """
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    return PostgresHook(postgres_conn_id=METADATA_CONN_ID).get_conn()


def vacuum_metadata_db(**context) -> dict:
    """Reclaim the disk space clean_airflow_metadata freed up. VACUUM can't run inside
    a transaction block, so autocommit is set explicitly before running it."""
    conn = _metadata_conn()
    try:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("VACUUM (ANALYZE)")
    finally:
        conn.close()
    return {"status": "vacuumed"}


def clean_airflow_metadata(**context) -> dict:
    """Delete rows older than the retention window from each table in METADATA_TABLES."""
    cutoff_days = _retention_days()
    deleted = {}
    conn = _metadata_conn()
    try:
        with conn.cursor() as cursor:
            for table, date_column in METADATA_TABLES.items():
                cursor.execute("SELECT to_regclass(%s)", (table,))
                if cursor.fetchone()[0] is None:
                    continue  # table not present in this Airflow version/executor - skip
                cursor.execute(
                    f'DELETE FROM "{table}" WHERE "{date_column}" < now() - %s * interval \'1 day\'',
                    (cutoff_days,),
                )
                deleted[table] = cursor.rowcount
        conn.commit()
    finally:
        conn.close()
    return {"retention_days": cutoff_days, "deleted_rows": deleted}


def clean_task_logs(**context) -> dict:
    """Delete task log files (and the now-empty directories under them) older than the retention window."""
    import time
    from pathlib import Path

    logs_root = Path(Variable.get("airflow_logs_path", default_var="/opt/airflow/logs"))
    cutoff = time.time() - _retention_days() * 86400
    deleted_files = deleted_dirs = 0

    if logs_root.exists():
        for path in sorted(logs_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted_files += 1
                elif path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
                    deleted_dirs += 1
            except OSError:
                continue  # file/dir removed or locked concurrently - skip, not fatal

    return {"logs_root": str(logs_root), "retention_days": _retention_days(),
            "deleted_files": deleted_files, "deleted_empty_dirs": deleted_dirs}


clean_metadata_task = PythonOperator(
    task_id="clean_airflow_metadata",
    python_callable=clean_airflow_metadata,
    dag=dag,
)

vacuum_task = PythonOperator(
    task_id="vacuum_metadata_db",
    python_callable=vacuum_metadata_db,
    dag=dag,
)

clean_logs_task = PythonOperator(
    task_id="clean_task_logs",
    python_callable=clean_task_logs,
    dag=dag,
)

clean_temp_task = BashOperator(
    task_id="clean_temp_files",
    bash_command=(
        'find /tmp -type f -mtime +{{ var.value.get("log_retention_days", 30) }} -delete 2>/dev/null || true; '
        'find data/processed -type f -name "tmp*" -mtime +{{ var.value.get("log_retention_days", 30) }} -delete 2>/dev/null || true'
    ),
    dag=dag,
)

clean_metadata_task >> vacuum_task
[vacuum_task, clean_logs_task, clean_temp_task]
