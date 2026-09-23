"""
FabricaIA - DW Validation Module

Post-load sanity checks for the Data Warehouse. Two severities:
  - failures: integrity violations (row counts, orphan keys, NULL keys, impossible values,
    staging-vs-dw reconciliation). They fail the Airflow task.
  - warnings: data-quality signals worth looking at (unknown categories, date anomalies,
    late deliveries, CEPs without coordinates...). They are reported but never fail the run.
Used as an Airflow task and from the validation notebook.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    checks_run: int = 0
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


class DWValidator:
    """Runs integrity checks against the dw schema."""

    EXPECTED_MIN_ROWS = {
        "dim_customers": 1,
        "dim_sellers": 1,
        "dim_products": 1,
        "dim_geolocation": 1,
        "dim_date": 1,
        "fact_order_items": 1,
        "fact_orders": 1,
        "fact_payments": 1,
        "fact_reviews": 1,
    }

    ORPHAN_FK_CHECKS = [
        # (fact_table, fk_column, dim_table, dim_key_column)
        ("fact_order_items", "customer_key", "dim_customers", "customer_key"),
        ("fact_order_items", "seller_key", "dim_sellers", "seller_key"),
        ("fact_order_items", "product_key", "dim_products", "product_key"),
        ("fact_orders", "customer_key", "dim_customers", "customer_key"),
    ]

    # Facts that reference an order by its natural key (no surrogate FK constraint).
    ORDER_REFERENCE_CHECKS = [
        ("fact_order_items", "order_id"),
        ("fact_payments", "order_id"),
        ("fact_reviews", "order_id"),
    ]

    NOT_NULL_KEY_CHECKS = [
        ("fact_order_items", "customer_key"),
        ("fact_order_items", "seller_key"),
        ("fact_order_items", "product_key"),
        ("fact_order_items", "order_purchase_date_key"),
        ("fact_orders", "customer_key"),
        ("fact_orders", "order_purchase_date_key"),
    ]

    # (table, SQL predicate that must be TRUE for every row that has a value, description)
    VALUE_RANGE_CHECKS = [
        ("fact_order_items", "price > 0", "price must be > 0"),
        ("fact_order_items", "freight_value >= 0", "freight_value must be >= 0"),
        ("fact_payments", "payment_value >= 0", "payment_value must be >= 0"),
        ("fact_payments", "payment_installments >= 1", "payment_installments must be >= 1"),
        ("fact_reviews", "review_score BETWEEN 1 AND 5", "review_score must be between 1 and 5"),
        ("fact_orders", "delivery_time_days >= 0", "delivery_time_days must be >= 0"),
    ]

    # (staging table, dw table): both sides must have the same number of rows.
    RECONCILIATION_CHECKS = [
        ("stg_customers", "dim_customers"),
        ("stg_sellers", "dim_sellers"),
        ("stg_products", "dim_products"),
        ("stg_geolocation", "dim_geolocation"),
        ("stg_orders", "fact_orders"),
        ("stg_order_items", "fact_order_items"),
        ("stg_order_payments", "fact_payments"),
        ("stg_order_reviews", "fact_reviews"),
    ]

    # (description, SQL returning a percentage 0-100, threshold above which we warn)
    WARNING_RATE_CHECKS = [
        ("products with unknown category (%)",
         "SELECT 100.0 * AVG((product_category_name = 'unknown')::int) FROM {dw}.dim_products", 5.0),
        ("orders with inconsistent timestamps (%)",
         "SELECT 100.0 * AVG(has_date_anomaly::int) FROM {dw}.fact_orders", 5.0),
        ("delivered orders that arrived late (%)",
         "SELECT 100.0 * AVG(is_late::int) FROM {dw}.fact_orders WHERE is_late IS NOT NULL", 15.0),
        ("customers whose CEP has no coordinates (%)",
         "SELECT 100.0 * AVG((NOT COALESCE(zip_in_geolocation, FALSE))::int) FROM {dw}.dim_customers", 2.0),
        ("payments flagged invalid (%)",
         "SELECT 100.0 * AVG((NOT is_valid_payment)::int) FROM {dw}.fact_payments", 0.5),
    ]

    def __init__(self, engine: Engine, schema_dw: str = "dw", schema_staging: str = "staging"):
        self.engine = engine
        self.schema_dw = schema_dw
        self.schema_staging = schema_staging

    def _scalar(self, sql: str):
        with self.engine.connect() as conn:
            return conn.execute(text(sql)).scalar_one()

    def _row_count(self, table_name: str, schema: str = None) -> int:
        return self._scalar(f"SELECT COUNT(*) FROM {schema or self.schema_dw}.{table_name}")

    def check_row_counts(self, result: ValidationResult) -> None:
        for table_name, min_rows in self.EXPECTED_MIN_ROWS.items():
            result.checks_run += 1
            count = self._row_count(table_name)
            if count < min_rows:
                result.failures.append(
                    f"{table_name} has {count} rows, expected at least {min_rows}"
                )

    def check_orphan_foreign_keys(self, result: ValidationResult) -> None:
        for fact_table, fk_column, dim_table, dim_key in self.ORPHAN_FK_CHECKS:
            result.checks_run += 1
            orphan_count = self._scalar(
                f"""
                SELECT COUNT(*)
                FROM {self.schema_dw}.{fact_table} f
                LEFT JOIN {self.schema_dw}.{dim_table} d ON f.{fk_column} = d.{dim_key}
                WHERE f.{fk_column} IS NOT NULL AND d.{dim_key} IS NULL
                """
            )
            if orphan_count > 0:
                result.failures.append(
                    f"{fact_table}.{fk_column} has {orphan_count} orphan rows not found in "
                    f"{dim_table}.{dim_key}"
                )

    def check_order_references(self, result: ValidationResult) -> None:
        for fact_table, column in self.ORDER_REFERENCE_CHECKS:
            result.checks_run += 1
            orphan_count = self._scalar(
                f"""
                SELECT COUNT(*)
                FROM {self.schema_dw}.{fact_table} f
                LEFT JOIN {self.schema_dw}.fact_orders o ON f.{column} = o.order_id
                WHERE o.order_id IS NULL
                """
            )
            if orphan_count > 0:
                result.failures.append(
                    f"{fact_table}.{column} has {orphan_count} rows pointing to a missing order in fact_orders"
                )

    def check_not_null_keys(self, result: ValidationResult) -> None:
        for table_name, column in self.NOT_NULL_KEY_CHECKS:
            result.checks_run += 1
            nulls = self._scalar(f"SELECT COUNT(*) FROM {self.schema_dw}.{table_name} WHERE {column} IS NULL")
            if nulls > 0:
                result.failures.append(f"{table_name}.{column} has {nulls} NULL values, expected none")

    def check_value_ranges(self, result: ValidationResult) -> None:
        for table_name, predicate, description in self.VALUE_RANGE_CHECKS:
            result.checks_run += 1
            # NULLs are allowed here (missing != invalid); only rows that HAVE a value can violate the rule.
            violations = self._scalar(
                f"SELECT COUNT(*) FROM {self.schema_dw}.{table_name} WHERE NOT COALESCE(({predicate}), TRUE)"
            )
            if violations > 0:
                result.failures.append(f"{table_name}: {violations} rows violate '{description}'")

    def check_staging_reconciliation(self, result: ValidationResult) -> None:
        for staging_table, dw_table in self.RECONCILIATION_CHECKS:
            result.checks_run += 1
            staging_rows = self._row_count(staging_table, self.schema_staging)
            dw_rows = self._row_count(dw_table)
            if staging_rows != dw_rows:
                result.failures.append(
                    f"{staging_table} has {staging_rows} rows but {dw_table} has {dw_rows}: rows were lost or duplicated"
                )

    def check_quality_warnings(self, result: ValidationResult) -> None:
        for description, sql, threshold in self.WARNING_RATE_CHECKS:
            result.checks_run += 1
            value = self._scalar(sql.format(dw=self.schema_dw))
            if value is not None and float(value) > threshold:
                result.warnings.append(f"{description}: {float(value):.2f}% (threshold {threshold}%)")

    def run_all(self) -> ValidationResult:
        result = ValidationResult()
        self.check_row_counts(result)
        self.check_orphan_foreign_keys(result)
        self.check_order_references(result)
        self.check_not_null_keys(result)
        self.check_value_ranges(result)
        self.check_staging_reconciliation(result)
        self.check_quality_warnings(result)
        if result.passed:
            logger.info("DW validation passed: %s checks run, %s warnings.", result.checks_run, len(result.warnings))
        else:
            logger.error("DW validation failed: %s", result.failures)
        for warning in result.warnings:
            logger.warning("DW data-quality warning: %s", warning)
        return result
