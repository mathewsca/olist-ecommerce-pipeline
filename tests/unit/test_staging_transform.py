"""
FabricaIA ETL - Tests for the pure (no-DB) staging transform functions.
"""

import numpy as np
import pandas as pd
import pytest

from src.etl.staging_transform import StagingTransformer


class TestCleanCustomers:
    def test_dedupes_and_normalizes(self):
        df = pd.DataFrame(
            {
                "customer_id": ["c1", "c1", "c2"],
                "customer_unique_id": ["u1", "u1", "u2"],
                "customer_zip_code_prefix": [1001, 1001, 2],
                "customer_city": ["sao paulo", "sao paulo", "rio"],
                "customer_state": ["sp", "sp", "rj"],
            }
        )
        out = StagingTransformer.clean_customers(df)

        assert len(out) == 2
        assert out["customer_zip_code_prefix"].tolist() == ["01001", "00002"]
        assert set(out["customer_state"]) == {"SP", "RJ"}


class TestCleanOrders:
    def test_parses_timestamps_and_lowercases_status(self):
        df = pd.DataFrame(
            {
                "order_id": ["o1", "o1"],
                "order_status": ["Delivered", "Delivered"],
                "order_purchase_timestamp": ["2018-01-01 10:00:00", "2018-01-01 10:00:00"],
                "order_approved_at": [None, None],
                "order_delivered_carrier_date": [None, None],
                "order_delivered_customer_date": ["2018-01-05 10:00:00", "2018-01-05 10:00:00"],
                "order_estimated_delivery_date": ["2018-01-10 00:00:00", "2018-01-10 00:00:00"],
            }
        )
        out = StagingTransformer.clean_orders(df)

        assert len(out) == 1  # deduped on order_id
        assert out["order_status"].iloc[0] == "delivered"
        assert pd.api.types.is_datetime64_any_dtype(out["order_purchase_timestamp"])


class TestCleanOrderItems:
    def test_casts_numeric_and_date_columns(self):
        df = pd.DataFrame(
            {
                "order_id": ["o1"],
                "order_item_id": [1],
                "price": ["100.50"],
                "freight_value": ["10.00"],
                "shipping_limit_date": ["2018-01-01 00:00:00"],
            }
        )
        out = StagingTransformer.clean_order_items(df)

        assert out["price"].iloc[0] == pytest.approx(100.50)
        assert out["freight_value"].iloc[0] == pytest.approx(10.00)
        assert pd.api.types.is_datetime64_any_dtype(out["shipping_limit_date"])


class TestCleanOrderPayments:
    def test_casts_and_lowercases_payment_type(self):
        df = pd.DataFrame(
            {
                "order_id": ["o1"],
                "payment_sequential": [1],
                "payment_type": ["Credit_Card"],
                "payment_installments": ["3"],
                "payment_value": ["150.00"],
            }
        )
        out = StagingTransformer.clean_order_payments(df)

        assert out["payment_type"].iloc[0] == "credit_card"
        assert int(out["payment_installments"].iloc[0]) == 3
        assert out["payment_value"].iloc[0] == pytest.approx(150.0)


class TestCleanOrderReviews:
    def test_dedupes_and_parses_dates(self):
        df = pd.DataFrame(
            {
                "review_id": ["r1", "r1"],
                "review_score": ["5", "5"],
                "review_creation_date": ["2018-01-01", "2018-01-01"],
                "review_answer_timestamp": ["2018-01-02 00:00:00", "2018-01-02 00:00:00"],
            }
        )
        out = StagingTransformer.clean_order_reviews(df)

        assert len(out) == 1
        assert int(out["review_score"].iloc[0]) == 5
        assert pd.api.types.is_datetime64_any_dtype(out["review_creation_date"])


class TestCleanProducts:
    def test_merges_category_translation_and_fills_unknowns(self):
        products = pd.DataFrame(
            {
                "product_id": ["p1", "p2"],
                "product_category_name": ["moveis", None],
                "product_weight_g": ["500", "700"],
                "product_length_cm": ["10", "20"],
                "product_height_cm": ["10", "20"],
                "product_width_cm": ["10", "20"],
                "product_photos_qty": ["1", "2"],
            }
        )
        translation = pd.DataFrame(
            {
                "product_category_name": ["moveis"],
                "product_category_name_english": ["furniture"],
            }
        )
        out = StagingTransformer.clean_products(products, translation)

        assert out.loc[out["product_id"] == "p1", "product_category_name_english"].iloc[0] == "furniture"
        assert out.loc[out["product_id"] == "p2", "product_category_name_english"].iloc[0] == "unknown"
        assert pd.api.types.is_numeric_dtype(out["product_weight_g"])


class TestCleanGeolocation:
    def test_aggregates_duplicate_points_per_zip_prefix(self):
        df = pd.DataFrame(
            {
                "geolocation_zip_code_prefix": [1, 1, 2],
                "geolocation_city": ["sp", "sp", "rj"],
                "geolocation_state": ["sp", "sp", "rj"],
                "geolocation_lat": [-23.0, -23.2, -22.9],
                "geolocation_lng": [-46.0, -46.2, -43.2],
            }
        )
        out = StagingTransformer.clean_geolocation(df)

        assert len(out) == 2  # one row per zip prefix
        row = out[out["geolocation_zip_code_prefix"] == "00001"].iloc[0]
        assert row["avg_lat"] == pytest.approx(-23.1)
        assert row["avg_lng"] == pytest.approx(-46.1)
