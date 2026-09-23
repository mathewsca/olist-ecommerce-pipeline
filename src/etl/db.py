"""
FabricaIA - ETL Database Connection Module

Centralizes the SQLAlchemy connection to the Olist Data Warehouse (Postgres),
so the Airflow DAG, notebooks and the Streamlit dashboard all resolve the
connection the same way: config/config.yaml DATABASE section, overridden by
DW_* environment variables.
"""

import csv
import io
import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine.url import URL

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/config.yaml"


def load_database_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    Load the DATABASE section from config.yaml, with DW_* env vars taking
    precedence over the YAML values (same override convention as MLFLOW).

    Args:
        config_path: Path to the YAML config file.

    Returns:
        dict with keys: host, port, name, user, password,
        schema_raw, schema_staging, schema_dw.
    """
    yaml_config = {}
    if Path(config_path).exists():
        with open(config_path, "r") as file:
            full_config = yaml.safe_load(file) or {}
        yaml_config = full_config.get("DATABASE", {})

    return {
        "host": os.environ.get("DW_HOST", yaml_config.get("host", "localhost")),
        "port": int(os.environ.get("DW_PORT", yaml_config.get("port", 5432))),
        "name": os.environ.get("DW_NAME", yaml_config.get("name", "olist_dw")),
        "user": os.environ.get("DW_USER", yaml_config.get("user", "airflow")),
        "password": os.environ.get("DW_PASSWORD", yaml_config.get("password", "airflow")),
        "schema_raw": yaml_config.get("schema_raw", "raw"),
        "schema_staging": yaml_config.get("schema_staging", "staging"),
        "schema_dw": yaml_config.get("schema_dw", "dw"),
    }


def get_engine(config_path: str = DEFAULT_CONFIG_PATH, echo: bool = False) -> Engine:
    """
    Build a SQLAlchemy engine for the Olist Data Warehouse.

    Args:
        config_path: Path to the YAML config file.
        echo: Whether SQLAlchemy should log emitted SQL (useful for teaching).

    Returns:
        A SQLAlchemy Engine connected to the configured Postgres database.
    """
    db_config = load_database_config(config_path)
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["name"],
    )
    logger.info(
        "Creating DW engine for %s:%s/%s", db_config["host"], db_config["port"], db_config["name"]
    )
    return create_engine(url, echo=echo)


_NULL_MARKER = r"\N"  # PostgreSQL's conventional NULL token in COPY text/csv
_NULL_LITERAL = "'" + _NULL_MARKER + "'"


def _csv_value(value):
    """
    Serialize one cell for COPY. None -> NULL marker. Integral floats become ints because
    nullable-integer columns (Int64 with NULLs) reach us as 37.0, which COPY rejects for INT
    columns while NUMERIC/FLOAT columns accept the plain integer form just as well.
    """
    if value is None:
        return _NULL_MARKER
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return int(value)
    return value


def _copy_insert(table, conn, keys, data_iter) -> None:
    """
    pandas `to_sql(method=...)` callable that bulk-loads through PostgreSQL COPY.

    Much faster than row-by-row INSERTs on the 1M-row geolocation table. NULLs are sent
    with an explicit marker so empty strings stay empty strings.
    """
    columns = ", ".join(f'"{key}"' for key in keys)
    qualified = f'"{table.schema}"."{table.name}"' if table.schema else f'"{table.name}"'
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for row in data_iter:
        writer.writerow([_csv_value(value) for value in row])
    buffer.seek(0)
    with conn.connection.cursor() as cursor:
        cursor.copy_expert(f"COPY {qualified} ({columns}) FROM STDIN WITH (FORMAT csv, NULL {_NULL_LITERAL})", buffer)


def to_sql_fast(df, name: str, engine: Engine, schema: Optional[str] = None, if_exists: str = "fail") -> int:
    """
    Write a DataFrame to Postgres with COPY (see _copy_insert). Same semantics as
    DataFrame.to_sql(index=False): creates the table when missing, or replaces/appends per `if_exists`.
    """
    df.to_sql(name, engine, schema=schema, if_exists=if_exists, index=False, method=_copy_insert, chunksize=100_000)
    return len(df)
