"""
Dashboard query layer.

Every analysis reads the star schema through one of these functions. Filters (period and customer
state) are applied uniformly: `Filters` is hashable, so Streamlit caches each (query, filters) pair.

Business definitions used everywhere:
  - "valid order"  = status is not canceled/unavailable.
  - revenue (GMV)  = items price + freight of valid orders.
  - late           = delivered after the estimated date (day granularity), delivered orders only.
  - review of an order = its most recent review (fact_reviews.is_latest_for_order).
"""

from dataclasses import dataclass
from datetime import date
from typing import Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import Engine, bindparam, text

VALID = "o.order_status NOT IN ('canceled', 'unavailable')"

ORDER_FROM = """
FROM dw.fact_orders o
JOIN dw.dim_date d ON o.order_purchase_date_key = d.date_key
JOIN dw.dim_customers c ON o.customer_key = c.customer_key
"""

ITEM_FROM = """
FROM dw.fact_order_items f
JOIN dw.fact_orders o ON f.order_id = o.order_id
JOIN dw.dim_date d ON f.order_purchase_date_key = d.date_key
JOIN dw.dim_customers c ON f.customer_key = c.customer_key
JOIN dw.dim_products p ON f.product_key = p.product_key
"""

DELAY_BUCKET = """
CASE WHEN o.delivery_delay_days <= 0 THEN '1. No prazo'
     WHEN o.delivery_delay_days <= 3 THEN '2. Atraso 1-3 dias'
     WHEN o.delivery_delay_days <= 7 THEN '3. Atraso 4-7 dias'
     ELSE '4. Atraso 8+ dias' END
"""


@dataclass(frozen=True)
class Filters:
    date_from: date
    date_to: date
    states: Tuple[str, ...] = ()


def _run(engine: Engine, sql: str, f: Filters, **params) -> pd.DataFrame:
    """Execute `sql`, replacing {where} with the period/state filter."""
    where = "WHERE d.full_date BETWEEN :date_from AND :date_to"
    values = {"date_from": f.date_from, "date_to": f.date_to, **params}
    statement_text = sql.replace("{where}", where + (" AND c.customer_state IN :states" if f.states else ""))
    statement = text(statement_text)
    if f.states:
        statement = statement.bindparams(bindparam("states", expanding=True))
        values["states"] = list(f.states)
    return pd.read_sql(statement, engine, params=values)


def _plain(engine: Engine, sql: str, **params) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine, params=params)


cached = st.cache_data(ttl=600, show_spinner=False)


# ----------------------------------------------------------------------
# Filter options
# ----------------------------------------------------------------------


@cached
def filter_bounds(_engine: Engine) -> pd.DataFrame:
    return _plain(
        _engine,
        """
        SELECT MIN(d.full_date) AS first_day, MAX(d.full_date) AS last_day
        FROM dw.fact_orders o JOIN dw.dim_date d ON o.order_purchase_date_key = d.date_key
        """,
    )


@cached
def state_options(_engine: Engine) -> list:
    df = _plain(_engine, "SELECT DISTINCT customer_state FROM dw.dim_customers WHERE customer_state IS NOT NULL ORDER BY 1")
    return df["customer_state"].tolist()


# ----------------------------------------------------------------------
# Headline numbers and revenue over time
# ----------------------------------------------------------------------


@cached
def kpis(_engine: Engine, f: Filters) -> pd.Series:
    df = _run(
        _engine,
        f"""
        SELECT
            COUNT(*) AS pedidos,
            COUNT(*) FILTER (WHERE {VALID}) AS pedidos_validos,
            COUNT(*) FILTER (WHERE o.order_status = 'canceled') AS cancelados,
            COALESCE(SUM(o.items_value + o.freight_value) FILTER (WHERE {VALID}), 0) AS receita,
            COALESCE(SUM(o.freight_value) FILTER (WHERE {VALID}), 0) AS frete,
            AVG(o.items_count) FILTER (WHERE {VALID} AND o.items_count > 0) AS itens_por_pedido,
            AVG(o.delivery_time_days) AS entrega_media,
            100.0 * AVG(o.is_late::int) FILTER (WHERE o.is_late IS NOT NULL) AS pct_atraso,
            COUNT(DISTINCT c.customer_unique_id) AS clientes
        {ORDER_FROM}
        {{where}}
        """,
        f,
    )
    return df.iloc[0]


@cached
def review_kpis(_engine: Engine, f: Filters) -> pd.Series:
    df = _run(
        _engine,
        f"""
        SELECT AVG(r.review_score) AS nota_media,
               100.0 * AVG((r.review_score >= 4)::int) AS pct_positivas,
               100.0 * AVG((r.review_score <= 2)::int) AS pct_negativas,
               COUNT(*) AS avaliacoes
        {ORDER_FROM}
        JOIN dw.fact_reviews r ON r.order_id = o.order_id AND r.is_latest_for_order
        {{where}} AND r.review_score IS NOT NULL
        """,
        f,
    )
    return df.iloc[0]


@cached
def monthly(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT d.year_month, MIN(d.full_date) AS mes,
               SUM(o.items_value + o.freight_value) AS receita,
               COUNT(*) AS pedidos,
               COUNT(DISTINCT c.customer_unique_id) AS clientes
        {ORDER_FROM}
        {{where}} AND {VALID}
        GROUP BY d.year_month ORDER BY d.year_month
        """,
        f,
    )


@cached
def weekday_hour(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT d.day_of_week, o.purchase_hour AS hora, COUNT(*) AS pedidos
        {ORDER_FROM}
        {{where}} AND {VALID} AND o.purchase_hour IS NOT NULL
        GROUP BY d.day_of_week, o.purchase_hour
        """,
        f,
    )


@cached
def status_counts(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT o.order_status AS status, COUNT(*) AS pedidos
        {ORDER_FROM}
        {{where}}
        GROUP BY o.order_status ORDER BY pedidos DESC
        """,
        f,
    )


# ----------------------------------------------------------------------
# Categories, payments, freight
# ----------------------------------------------------------------------


@cached
def categories(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT p.category_group AS grupo, p.product_category_name AS categoria_pt, p.product_category_name_english AS categoria,
               SUM(f.price) AS receita, SUM(f.freight_value) AS frete, COUNT(*) AS itens,
               COUNT(DISTINCT f.order_id) AS pedidos
        {ITEM_FROM}
        {{where}} AND {VALID}
        GROUP BY p.category_group, p.product_category_name, p.product_category_name_english
        """,
        f,
    )


@cached
def payments(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT pay.payment_type AS tipo, pay.payment_installments AS parcelas, pay.payment_value AS valor
        FROM dw.fact_payments pay
        JOIN dw.fact_orders o ON pay.order_id = o.order_id
        JOIN dw.dim_date d ON o.order_purchase_date_key = d.date_key
        JOIN dw.dim_customers c ON o.customer_key = c.customer_key
        {{where}} AND pay.is_valid_payment AND {VALID}
        """,
        f,
    )


# ----------------------------------------------------------------------
# Delivery
# ----------------------------------------------------------------------


@cached
def delivery_by_state(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT c.customer_state AS estado, c.customer_region AS regiao,
               COUNT(*) AS pedidos_entregues,
               AVG(o.delivery_time_days) AS entrega_media,
               AVG((de.full_date - d.full_date)) AS prazo_prometido,
               100.0 * AVG(o.is_late::int) AS pct_atraso,
               SUM(o.freight_value) / NULLIF(SUM(o.items_value), 0) * 100 AS frete_pct_preco
        {ORDER_FROM}
        JOIN dw.dim_date de ON o.order_estimated_delivery_date_key = de.date_key
        {{where}} AND o.is_late IS NOT NULL
        GROUP BY c.customer_state, c.customer_region
        """,
        f,
    )


@cached
def delivery_time_hist(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT LEAST(FLOOR(o.delivery_time_days), 60)::int AS dias, COUNT(*) AS pedidos
        {ORDER_FROM}
        {{where}} AND o.delivery_time_days IS NOT NULL
        GROUP BY 1 ORDER BY 1
        """,
        f,
    )


@cached
def delay_vs_review(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT {DELAY_BUCKET} AS faixa, AVG(r.review_score) AS nota_media, COUNT(*) AS pedidos,
               100.0 * AVG((r.review_score <= 2)::int) AS pct_negativas
        {ORDER_FROM}
        JOIN dw.fact_reviews r ON r.order_id = o.order_id AND r.is_latest_for_order
        {{where}} AND o.is_late IS NOT NULL AND r.review_score IS NOT NULL
        GROUP BY 1 ORDER BY 1
        """,
        f,
    )


# ----------------------------------------------------------------------
# Reviews
# ----------------------------------------------------------------------


@cached
def review_scores(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT r.review_score AS nota, COUNT(*) AS avaliacoes
        {ORDER_FROM}
        JOIN dw.fact_reviews r ON r.order_id = o.order_id AND r.is_latest_for_order
        {{where}} AND r.review_score IS NOT NULL
        GROUP BY 1 ORDER BY 1
        """,
        f,
    )


@cached
def review_by_group(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT p.category_group AS grupo, AVG(r.review_score) AS nota_media,
               100.0 * AVG((r.review_score <= 2)::int) AS pct_negativas, COUNT(DISTINCT f.order_id) AS pedidos
        {ITEM_FROM}
        JOIN dw.fact_reviews r ON r.order_id = f.order_id AND r.is_latest_for_order
        {{where}} AND {VALID} AND r.review_score IS NOT NULL
        GROUP BY p.category_group
        """,
        f,
    )


@cached
def review_comments(_engine: Engine, f: Filters) -> pd.DataFrame:
    """Cleaned comments (informative ones only) with their score, for the word analysis."""
    return _run(
        _engine,
        f"""
        SELECT r.review_score AS nota, r.review_comment_message AS mensagem
        {ORDER_FROM}
        JOIN dw.fact_reviews r ON r.order_id = o.order_id AND r.is_latest_for_order
        {{where}} AND r.review_comment_message IS NOT NULL AND NOT r.is_low_information
        """,
        f,
    )


# ----------------------------------------------------------------------
# Geography, sellers, customers
# ----------------------------------------------------------------------


@cached
def revenue_by_state(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT c.customer_state AS estado, c.customer_region AS regiao,
               SUM(o.items_value + o.freight_value) AS receita, COUNT(*) AS pedidos,
               SUM(o.freight_value) / NULLIF(SUM(o.items_value), 0) * 100 AS frete_pct_preco
        {ORDER_FROM}
        {{where}} AND {VALID}
        GROUP BY c.customer_state, c.customer_region
        """,
        f,
    )


@cached
def map_points(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT g.zip_code_prefix, g.city AS cidade, g.state AS estado, g.avg_lat AS lat, g.avg_lng AS lng,
               COUNT(*) AS pedidos, SUM(o.items_value + o.freight_value) AS receita
        {ORDER_FROM}
        JOIN dw.dim_geolocation g ON g.zip_code_prefix = c.customer_zip_code_prefix
        {{where}} AND {VALID} AND g.avg_lat IS NOT NULL
        GROUP BY g.zip_code_prefix, g.city, g.state, g.avg_lat, g.avg_lng
        ORDER BY pedidos DESC LIMIT 3000
        """,
        f,
    )


@cached
def sellers(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT s.seller_key, s.seller_id, s.seller_state AS estado,
               SUM(f.price) AS receita, COUNT(*) AS itens, COUNT(DISTINCT f.order_id) AS pedidos,
               AVG(r.review_score) AS nota_media, 100.0 * AVG(o.is_late::int) AS pct_atraso
        {ITEM_FROM}
        JOIN dw.dim_sellers s ON f.seller_key = s.seller_key
        LEFT JOIN dw.fact_reviews r ON r.order_id = f.order_id AND r.is_latest_for_order
        {{where}} AND {VALID}
        GROUP BY s.seller_key, s.seller_id, s.seller_state
        """,
        f,
    )


@cached
def repeat_customers(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT c.customer_unique_id, COUNT(*) AS pedidos, SUM(o.items_value + o.freight_value) AS receita
        {ORDER_FROM}
        {{where}} AND {VALID}
        GROUP BY c.customer_unique_id
        """,
        f,
    )


# ----------------------------------------------------------------------
# Exploratory analysis (item / order grain, deliberately without fan-out)
# ----------------------------------------------------------------------


@cached
def item_grain(_engine: Engine, f: Filters) -> pd.DataFrame:
    return _run(
        _engine,
        f"""
        SELECT f.price, f.freight_value, f.is_price_outlier, p.category_group AS grupo,
               p.product_category_name_english AS categoria, p.product_weight_g AS peso_g,
               p.product_volume_cm3 AS volume_cm3, c.customer_state AS estado
        {ITEM_FROM}
        {{where}} AND {VALID}
        """,
        f,
    )


@cached
def order_grain(_engine: Engine, f: Filters) -> pd.DataFrame:
    """One row per order: the latest review is joined once, so order totals are never multiplied."""
    return _run(
        _engine,
        f"""
        SELECT o.order_id, o.order_status, o.delivery_time_days, o.delivery_delay_days, o.approval_time_hours,
               o.items_count, o.items_value AS price_total, o.freight_value AS freight_total,
               c.customer_state AS estado, c.customer_region AS regiao, r.review_score, o.has_date_anomaly
        {ORDER_FROM}
        LEFT JOIN dw.fact_reviews r ON r.order_id = o.order_id AND r.is_latest_for_order
        {{where}} AND {VALID}
        """,
        f,
    )


# ----------------------------------------------------------------------
# Data quality (written by the staging layer; not filtered by period/state)
# ----------------------------------------------------------------------


@cached
def dq_issues(_engine: Engine) -> pd.DataFrame:
    return _plain(
        _engine,
        """
        SELECT table_name, column_name, rule, description, severity, action,
               rows_affected, total_rows, pct_affected, examples, checked_at
        FROM staging.dq_issues
        """,
    )


@cached
def dq_profile(_engine: Engine) -> pd.DataFrame:
    return _plain(
        _engine,
        "SELECT table_name, stage, column_name, dtype, total_rows, null_count, null_pct, distinct_count FROM staging.dq_column_profile",
    )


@cached
def date_anomalies(_engine: Engine) -> pd.DataFrame:
    """Order-level date anomaly counts by kind, straight from staging.stg_orders."""
    return _plain(
        _engine,
        """
        SELECT 'Transportadora antes da aprovação' AS anomalia, SUM(flag_carrier_before_approval::int) AS pedidos FROM staging.stg_orders
        UNION ALL SELECT 'Transportadora antes da compra', SUM(flag_carrier_before_purchase::int) FROM staging.stg_orders
        UNION ALL SELECT 'Entrega antes da transportadora', SUM(flag_delivered_before_carrier::int) FROM staging.stg_orders
        UNION ALL SELECT 'Entregue sem data de entrega', SUM(flag_delivered_without_date::int) FROM staging.stg_orders
        UNION ALL SELECT 'Não entregue, mas com data de entrega', SUM(flag_undelivered_with_delivery_date::int) FROM staging.stg_orders
        UNION ALL SELECT 'Aprovação antes da compra', SUM(flag_approved_before_purchase::int) FROM staging.stg_orders
        """,
    )


@cached
def dw_row_counts(_engine: Engine) -> pd.DataFrame:
    """Rows per warehouse table, keyed by the raw/staging table name it came from."""
    return _plain(
        _engine,
        """
        SELECT 'customers' AS t, (SELECT COUNT(*) FROM dw.dim_customers) AS dw UNION ALL
        SELECT 'sellers', (SELECT COUNT(*) FROM dw.dim_sellers) UNION ALL
        SELECT 'products', (SELECT COUNT(*) FROM dw.dim_products) UNION ALL
        SELECT 'geolocation', (SELECT COUNT(*) FROM dw.dim_geolocation) UNION ALL
        SELECT 'orders', (SELECT COUNT(*) FROM dw.fact_orders) UNION ALL
        SELECT 'order_items', (SELECT COUNT(*) FROM dw.fact_order_items) UNION ALL
        SELECT 'order_payments', (SELECT COUNT(*) FROM dw.fact_payments) UNION ALL
        SELECT 'order_reviews', (SELECT COUNT(*) FROM dw.fact_reviews)
        """,
    )
