"""
FabricaIA - Data Warehouse Builder Module

Builds the star-schema Data Warehouse (dw schema) from the cleaned staging
tables: dimensions first (with surrogate keys), then fact tables that look
up those surrogate keys via the Olist natural keys.
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import Engine, text

from src.etl.db import to_sql_fast

logger = logging.getLogger(__name__)

DDL_PATHS = {
    "raw": "sql/raw/create_raw_schema.sql",
    "staging": "sql/staging/create_staging_schema.sql",
    "dw": "sql/dw/create_dw_schema.sql",
}

MONTH_NAMES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}
DAY_NAMES_PT = {
    0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 4: "Sexta", 5: "Sábado", 6: "Domingo",
}


def date_key(value: pd.Timestamp):
    """Convert a timestamp to the YYYYMMDD integer key used by dim_date, or None."""
    if pd.isna(value):
        return None
    return int(value.strftime("%Y%m%d"))


def date_keys(series: pd.Series) -> pd.Series:
    """Vectorized date_key: a datetime Series -> nullable Int64 YYYYMMDD keys."""
    return pd.to_numeric(series.dt.strftime("%Y%m%d"), errors="coerce").astype("Int64")


class DWBuilder:
    """Builds the dw schema (dimensions + facts) from staging.*."""

    def __init__(self, engine: Engine, schema_staging: str = "staging", schema_dw: str = "dw"):
        self.engine = engine
        self.schema_staging = schema_staging
        self.schema_dw = schema_dw

    def run_ddl(self, sql_path: str) -> None:
        """Execute a .sql DDL file against the DW database."""
        sql_text = Path(sql_path).read_text(encoding="utf-8")
        with self.engine.begin() as conn:
            conn.execute(text(sql_text))
        logger.info("Executed DDL: %s", sql_path)

    def init_schemas(self) -> None:
        """Run the raw/staging/dw DDL files, in order."""
        for name in ("raw", "staging", "dw"):
            self.run_ddl(DDL_PATHS[name])

    def _read_staging(self, table_name: str) -> pd.DataFrame:
        return pd.read_sql_table(table_name, self.engine, schema=self.schema_staging)

    def _append_dw(self, df: pd.DataFrame, table_name: str) -> int:
        """
        Truncate then insert: every build reads a full snapshot of staging, so this must
        stay idempotent under Airflow retries (a retry re-runs the whole task from
        scratch, and a plain append would violate unique constraints on the second try).
        CASCADE also clears any fact rows still pointing at the dimension being rebuilt,
        since dimensions are rebuilt before facts within the same DAG run.
        """
        qualified = f'"{self.schema_dw}"."{table_name}"'
        with self.engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {qualified} RESTART IDENTITY CASCADE"))
        to_sql_fast(df, table_name, self.engine, schema=self.schema_dw, if_exists="append")
        logger.info("Inserted %s rows into %s.%s", len(df), self.schema_dw, table_name)
        return len(df)

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def build_dim_customers(self) -> int:
        df = self._read_staging("stg_customers")[
            [
                "customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city",
                "customer_state", "customer_region", "zip_in_geolocation", "zip_state_mismatch",
            ]
        ]
        return self._append_dw(df, "dim_customers")

    def build_dim_sellers(self) -> int:
        df = self._read_staging("stg_sellers")[
            [
                "seller_id", "seller_zip_code_prefix", "seller_city", "seller_state", "seller_region",
                "zip_in_geolocation", "zip_state_mismatch",
            ]
        ]
        return self._append_dw(df, "dim_sellers")

    def build_dim_products(self) -> int:
        df = self._read_staging("stg_products")[
            [
                "product_id",
                "product_category_name",
                "product_category_name_english",
                "category_group",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
                "product_volume_cm3",
                "product_photos_qty",
                "product_name_length",
                "product_description_length",
                "has_listing_metadata",
                "dimensions_imputed",
            ]
        ]
        return self._append_dw(df, "dim_products")

    def build_dim_geolocation(self) -> int:
        df = self._read_staging("stg_geolocation").rename(
            columns={
                "geolocation_zip_code_prefix": "zip_code_prefix",
                "geolocation_city": "city",
                "geolocation_state": "state",
                "geolocation_region": "region",
            }
        )[["zip_code_prefix", "city", "state", "region", "avg_lat", "avg_lng", "n_points", "n_points_discarded"]]
        return self._append_dw(df, "dim_geolocation")

    def build_dim_date(self) -> int:
        """
        Build a date spine covering every date referenced by orders and reviews
        (Olist data spans 2016-2018), so all *_date_key foreign keys resolve.
        """
        orders = self._read_staging("stg_orders")
        reviews = self._read_staging("stg_order_reviews")
        date_cols = [
            orders["order_purchase_timestamp"],
            orders["order_approved_at"],
            orders["order_delivered_carrier_date"],
            orders["order_delivered_customer_date"],
            orders["order_estimated_delivery_date"],
            reviews["review_creation_date"],
            reviews["review_answer_timestamp"],
        ]
        all_dates = pd.concat(date_cols).dropna().dt.normalize()
        start, end = all_dates.min(), all_dates.max()
        spine = pd.DataFrame({"full_date": pd.date_range(start, end, freq="D")})
        spine["date_key"] = date_keys(spine["full_date"])
        spine["year"] = spine["full_date"].dt.year
        spine["quarter"] = spine["full_date"].dt.quarter
        spine["month"] = spine["full_date"].dt.month
        spine["day"] = spine["full_date"].dt.day
        spine["day_of_week"] = spine["full_date"].dt.dayofweek
        spine["is_weekend"] = spine["day_of_week"].isin([5, 6])
        spine["year_month"] = spine["full_date"].dt.strftime("%Y-%m")
        spine["month_name"] = spine["month"].map(MONTH_NAMES_PT)
        spine["day_name"] = spine["day_of_week"].map(DAY_NAMES_PT)
        spine["week_of_year"] = spine["full_date"].dt.isocalendar().week.astype(int)
        return self._append_dw(spine, "dim_date")

    def build_all_dimensions(self) -> Dict[str, int]:
        return {
            "dim_customers": self.build_dim_customers(),
            "dim_sellers": self.build_dim_sellers(),
            "dim_products": self.build_dim_products(),
            "dim_geolocation": self.build_dim_geolocation(),
            "dim_date": self.build_dim_date(),
        }

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    def build_fact_order_items(self) -> int:
        items = self._read_staging("stg_order_items")
        orders = self._read_staging("stg_orders")[["order_id", "customer_id", "order_purchase_timestamp"]]
        dim_customers = self._read_staging_dw("dim_customers")[["customer_key", "customer_id"]]
        dim_sellers = self._read_staging_dw("dim_sellers")[["seller_key", "seller_id"]]
        dim_products = self._read_staging_dw("dim_products")[["product_key", "product_id"]]

        df = items.merge(orders, on="order_id", how="left")
        df = df.merge(dim_customers, on="customer_id", how="left")
        df = df.merge(dim_sellers, on="seller_id", how="left")
        df = df.merge(dim_products, on="product_id", how="left")
        df["order_purchase_date_key"] = date_keys(df["order_purchase_timestamp"])

        out = df[
            [
                "order_id",
                "order_item_id",
                "customer_key",
                "seller_key",
                "product_key",
                "order_purchase_date_key",
                "price",
                "freight_value",
                "is_price_outlier",
                "is_freight_above_price",
            ]
        ]
        return self._append_dw(out, "fact_order_items")

    def build_fact_orders(self) -> int:
        orders = self._read_staging("stg_orders")
        items = self._read_staging("stg_order_items")
        dim_customers = self._read_staging_dw("dim_customers")[["customer_key", "customer_id"]]

        df = orders.merge(dim_customers, on="customer_id", how="left")
        df["order_purchase_date_key"] = date_keys(df["order_purchase_timestamp"])
        df["order_delivered_date_key"] = date_keys(df["order_delivered_customer_date"])
        df["order_estimated_delivery_date_key"] = date_keys(df["order_estimated_delivery_date"])
        df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour.astype("Int64")

        seconds_per_hour, seconds_per_day = 3600.0, 86400.0
        df["approval_time_hours"] = (
            (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / seconds_per_hour
        ).round(2)
        df["carrier_handoff_days"] = (
            (df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / seconds_per_day
        ).round(2)

        # Delivery metrics only make sense for orders that really were delivered and whose
        # timeline is not contradictory (canceled orders with a delivery date are excluded).
        delivered = (
            (df["order_status"] == "delivered")
            & df["order_delivered_customer_date"].notna()
            & (df["order_delivered_customer_date"] >= df["order_purchase_timestamp"])
        )
        delivery_days = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / seconds_per_day
        delay_days = (
            df["order_delivered_customer_date"].dt.normalize() - df["order_estimated_delivery_date"].dt.normalize()
        ).dt.days
        df["delivery_time_days"] = delivery_days.round(2).where(delivered)
        df["delivery_delay_days"] = delay_days.where(delivered)
        df["is_late"] = (delay_days > 0).where(delivered).astype("boolean")

        rollup = items.groupby("order_id").agg(
            items_count=("order_item_id", "count"),
            items_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
        )
        df = df.merge(rollup, left_on="order_id", right_index=True, how="left")
        # Orders without items (canceled/unavailable) have zero items and zero value, not unknown ones.
        df[["items_count", "items_value", "freight_value"]] = df[["items_count", "items_value", "freight_value"]].fillna(0)
        df["items_count"] = df["items_count"].astype(int)

        out = df[
            [
                "order_id",
                "customer_key",
                "order_status",
                "order_purchase_date_key",
                "order_delivered_date_key",
                "order_estimated_delivery_date_key",
                "purchase_hour",
                "approval_time_hours",
                "carrier_handoff_days",
                "delivery_time_days",
                "delivery_delay_days",
                "is_late",
                "has_date_anomaly",
                "items_count",
                "items_value",
                "freight_value",
            ]
        ]
        return self._append_dw(out, "fact_orders")

    def build_fact_payments(self) -> int:
        df = self._read_staging("stg_order_payments")[
            [
                "order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value",
                "is_valid_payment",
            ]
        ]
        return self._append_dw(df, "fact_payments")

    def build_fact_reviews(self) -> int:
        df = self._read_staging("stg_order_reviews").copy()
        df["review_creation_date_key"] = date_keys(df["review_creation_date"])
        df["review_answer_date_key"] = date_keys(df["review_answer_timestamp"])
        df["has_comment_title"] = df["review_comment_title"].notna()
        df["has_comment_message"] = df["review_comment_message"].notna()
        df["response_time_hours"] = (
            (df["review_answer_timestamp"] - df["review_creation_date"]).dt.total_seconds() / 3600.0
        ).round(2)

        out = df[
            [
                "review_id",
                "order_id",
                "review_score",
                "review_creation_date_key",
                "review_answer_date_key",
                "has_comment_title",
                "has_comment_message",
                "review_comment_title",
                "review_comment_message",
                "comment_length",
                "is_low_information",
                "is_shouting",
                "is_latest_for_order",
                "response_time_hours",
            ]
        ]
        return self._append_dw(out, "fact_reviews")

    def build_all_facts(self) -> Dict[str, int]:
        return {
            "fact_order_items": self.build_fact_order_items(),
            "fact_orders": self.build_fact_orders(),
            "fact_payments": self.build_fact_payments(),
            "fact_reviews": self.build_fact_reviews(),
        }

    def _read_staging_dw(self, table_name: str) -> pd.DataFrame:
        return pd.read_sql_table(table_name, self.engine, schema=self.schema_dw)
