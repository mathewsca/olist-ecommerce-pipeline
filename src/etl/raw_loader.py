"""
FabricaIA - Raw Loader Module

Loads each Olist CSV, near-verbatim, into the Postgres "raw" schema. This is
the first landing zone: minimal typing, no cleaning — that happens later in
src/etl/staging_transform.py.
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import Engine, text

from src.etl.db import to_sql_fast

logger = logging.getLogger(__name__)

# Maps raw.<table_name> to the source CSV filename in data/raw/.
RAW_TABLE_MAP: Dict[str, str] = {
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


class RawLoader:
    """Loads Olist CSVs into the raw schema of the Data Warehouse."""

    def __init__(self, engine: Engine, source_dir: str = "data/raw", schema: str = "raw"):
        self.engine = engine
        self.source_dir = Path(source_dir)
        self.schema = schema

    def ensure_schema(self) -> None:
        """Create the raw schema if it doesn't exist yet."""
        with self.engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))

    def load_csv_to_raw(self, csv_filename: str, table_name: str) -> int:
        """
        Load a single CSV into raw.<table_name>, replacing it if it exists.

        Args:
            csv_filename: File name inside source_dir.
            table_name: Destination table name (without schema prefix).

        Returns:
            Number of rows loaded.
        """
        csv_path = self.source_dir / csv_filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        to_sql_fast(df, table_name, self.engine, schema=self.schema, if_exists="replace")
        logger.info("Loaded %s rows into %s.%s", len(df), self.schema, table_name)
        return len(df)

    def load_all(self) -> Dict[str, int]:
        """
        Load every table in RAW_TABLE_MAP into the raw schema.

        Returns:
            Mapping of table_name -> row count loaded.
        """
        self.ensure_schema()
        row_counts = {}
        for table_name, csv_filename in RAW_TABLE_MAP.items():
            row_counts[table_name] = self.load_csv_to_raw(csv_filename, table_name)
        return row_counts
