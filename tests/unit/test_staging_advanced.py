"""
FabricaIA ETL - Advanced staging cleaning tests.

Each case reproduces a behavior found while profiling the real Olist data
(shared review ids, timeline anomalies, untranslated categories, GPS outliers...).
"""

import pandas as pd
import pytest

from src.etl.data_quality import DataQualityLog
from src.etl.staging_transform import GeoReference, StagingTransformer


def _rules(dq: DataQualityLog) -> dict:
    return {issue.rule: issue.rows_affected for issue in dq.issues}


class TestCleanReviewsAdvanced:
    def _frame(self):
        return pd.DataFrame(
            {
                "review_id": ["r1", "r1", "r2", "r3"],
                "order_id": ["o1", "o2", "o3", "o3"],  # r1 is shared by two orders; o3 has two reviews
                "review_score": [5, 5, 1, 4],
                "review_comment_title": [None, None, "  Ruim ", None],
                "review_comment_message": ["ok", "Chegou\r\nrápido  demais", "PRODUTO QUEBRADO", None],
                "review_creation_date": ["2018-01-01"] * 4,
                "review_answer_timestamp": [
                    "2018-01-02 10:00:00", "2018-01-02 10:00:00", "2018-01-03 10:00:00", "2018-01-05 10:00:00",
                ],
            }
        )

    def test_same_review_id_on_different_orders_is_kept(self):
        dq = DataQualityLog()
        out = StagingTransformer.clean_order_reviews(self._frame(), dq)

        assert len(out) == 4  # deduplicating on review_id alone would keep only 2
        assert _rules(dq)["review_id_shared_by_several_orders"] == 2

    def test_true_duplicates_of_the_pair_are_dropped(self):
        df = pd.concat([self._frame(), self._frame().iloc[[0]]], ignore_index=True)
        assert len(StagingTransformer.clean_order_reviews(df)) == 4

    def test_latest_review_per_order_is_flagged(self):
        out = StagingTransformer.clean_order_reviews(self._frame())
        o3 = out[out["order_id"] == "o3"].set_index("review_score")
        assert bool(o3.loc[4, "is_latest_for_order"]) and not bool(o3.loc[1, "is_latest_for_order"])

    def test_comment_text_is_sanitized_and_featurized(self):
        out = StagingTransformer.clean_order_reviews(self._frame())
        assert out["review_comment_message"].iloc[1] == "Chegou rápido demais"
        assert out["review_comment_title"].iloc[2] == "Ruim"
        assert out["is_low_information"].tolist() == [True, False, False, False]
        assert bool(out["is_shouting"].iloc[2])
        assert out["review_comment_message_norm"].iloc[1] == "chegou rapido demais"


class TestCleanOrdersAdvanced:
    def test_timeline_anomalies_are_flagged_not_dropped(self):
        df = pd.DataFrame(
            {
                "order_id": ["ok", "carrier_first", "delivered_no_date", "canceled_but_delivered"],
                "customer_id": ["c"] * 4,
                "order_status": ["delivered", "delivered", "delivered", "canceled"],
                "order_purchase_timestamp": ["2018-01-01 10:00:00"] * 4,
                "order_approved_at": ["2018-01-01 11:00:00"] * 4,
                "order_delivered_carrier_date": ["2018-01-02", "2018-01-01 10:30:00", "2018-01-02", None],
                "order_delivered_customer_date": ["2018-01-05", "2018-01-05", None, "2018-01-06"],
                "order_estimated_delivery_date": ["2018-01-10"] * 4,
            }
        )
        out = StagingTransformer.clean_orders(df, DataQualityLog()).set_index("order_id")

        assert len(out) == 4
        assert out["has_date_anomaly"].to_dict() == {
            "ok": False, "carrier_first": True, "delivered_no_date": True, "canceled_but_delivered": True,
        }
        assert bool(out.loc["carrier_first", "flag_carrier_before_approval"])
        assert bool(out.loc["delivered_no_date", "flag_delivered_without_date"])
        assert bool(out.loc["canceled_but_delivered", "flag_undelivered_with_delivery_date"])

    def test_unknown_status_is_reported(self):
        df = pd.DataFrame(
            {
                "order_id": ["a"], "customer_id": ["c"], "order_status": ["Teleported "],
                "order_purchase_timestamp": ["2018-01-01"], "order_approved_at": [None],
                "order_delivered_carrier_date": [None], "order_delivered_customer_date": [None],
                "order_estimated_delivery_date": ["2018-01-10"],
            }
        )
        dq = DataQualityLog()
        StagingTransformer.clean_orders(df, dq)
        assert _rules(dq)["unknown_status"] == 1


class TestCleanItemsAndPaymentsAdvanced:
    def test_items_nullify_invalid_values_and_flag_outliers(self):
        df = pd.DataFrame(
            {
                "order_id": ["o1"] * 6,
                "order_item_id": [1, 2, 3, 4, 5, 6],
                "product_id": ["p"] * 6,
                "seller_id": ["s"] * 6,
                "shipping_limit_date": ["2018-01-01", "2020-02-05", "2018-01-01", "2018-01-01", "2018-01-01", "2018-01-01"],
                "price": ["10", "0", "12", "11", "13", "9999"],
                "freight_value": ["20", "5", "-1", "3", "3", "3"],
            }
        )
        out = StagingTransformer.clean_order_items(df).set_index("order_item_id")

        assert pd.isna(out.loc[2, "price"])                 # price must be > 0
        assert pd.isna(out.loc[3, "freight_value"])         # freight must be >= 0
        assert pd.isna(out.loc[2, "shipping_limit_date"])   # 2020 is outside the plausible window
        assert bool(out.loc[6, "is_price_outlier"])
        assert bool(out.loc[1, "is_freight_above_price"])

    def test_payments_fix_zero_installments_and_flag_undefined_type(self):
        df = pd.DataFrame(
            {
                "order_id": ["o1", "o2", "o3"],
                "payment_sequential": [1, 1, 1],
                "payment_type": ["Credit_Card", "not_defined", "voucher"],
                "payment_installments": [0, 1, 1],
                "payment_value": ["100.005", "0", "0"],
            }
        )
        dq = DataQualityLog()
        out = StagingTransformer.clean_order_payments(df, dq)

        assert out["payment_installments"].tolist() == [1, 1, 1]
        assert out["is_valid_payment"].tolist() == [True, False, True]
        assert _rules(dq)["zero_installments"] == 1


class TestCleanProductsAdvanced:
    def test_missing_translation_fallback_group_and_typo_rename(self):
        products = pd.DataFrame(
            {
                "product_id": ["p1", "p2", "p3"],
                "product_category_name": ["pc_gamer", "Cama Mesa Banho", None],
                "product_name_lenght": [40, 50, None],
                "product_description_lenght": [100, 200, None],
                "product_photos_qty": [1, 2, None],
                "product_weight_g": [1000, 500, 300],
                "product_length_cm": [10, 10, 10],
                "product_height_cm": [10, 10, 10],
                "product_width_cm": [10, 10, 10],
            }
        )
        translation = pd.DataFrame(
            {"product_category_name": ["cama_mesa_banho"], "product_category_name_english": ["bed_bath_table"]}
        )
        out = StagingTransformer.clean_products(products, translation).set_index("product_id")

        assert out.loc["p1", "product_category_name_english"] == "pc_gamer"   # not 'unknown'
        assert out.loc["p2", "product_category_name"] == "cama_mesa_banho"
        assert out.loc["p2", "category_group"] == "Casa e Decoração"
        assert out.loc["p3", "product_category_name"] == "unknown"
        assert out.loc["p3", "category_group"] == "Sem categoria"
        assert not bool(out.loc["p3", "has_listing_metadata"])
        assert "product_name_length" in out.columns and "product_name_lenght" not in out.columns

    def test_zero_weight_is_nullified_then_imputed_with_category_median(self):
        products = pd.DataFrame(
            {
                "product_id": ["a", "b", "c"],
                "product_category_name": ["x", "x", "x"],
                "product_weight_g": [0, 100, 300],
                "product_length_cm": [10, 10, 10],
                "product_height_cm": [5, 5, 5],
                "product_width_cm": [4, 4, 4],
                "product_photos_qty": [1, 1, 1],
            }
        )
        translation = pd.DataFrame({"product_category_name": ["x"], "product_category_name_english": ["x_en"]})
        out = StagingTransformer.clean_products(products, translation).set_index("product_id")

        assert out.loc["a", "product_weight_g"] == 200            # median of 100 and 300
        assert bool(out.loc["a", "dimensions_imputed"]) and not bool(out.loc["b", "dimensions_imputed"])
        assert out.loc["a", "product_volume_cm3"] == 200


class TestCleanGeolocationAdvanced:
    def test_outlier_points_are_discarded_and_mode_city_wins(self):
        df = pd.DataFrame(
            {
                "geolocation_zip_code_prefix": [1046] * 5,
                "geolocation_lat": [-23.55, -23.56, -23.54, 41.6, -34.6],   # last two: Europe / Argentina
                "geolocation_lng": [-46.63, -46.64, -46.62, -8.4, -58.7],
                "geolocation_city": ["são paulo", "sao paulo", "sao paulo", "sao paulo", "sao paulo"],
                "geolocation_state": ["SP"] * 5,
            }
        )
        dq = DataQualityLog()
        out = StagingTransformer.clean_geolocation(df, dq).iloc[0]

        assert out["geolocation_zip_code_prefix"] == "01046"
        assert out["geolocation_city"] == "sao paulo"
        assert out["n_points"] == 3 and out["n_points_discarded"] == 2
        assert out["avg_lat"] == pytest.approx(-23.55)
        assert _rules(dq)["coordinates_outside_brazil"] == 2
        assert out["geolocation_region"] == "Sudeste"

    def test_zip_with_two_states_keeps_the_most_frequent(self):
        df = pd.DataFrame(
            {
                "geolocation_zip_code_prefix": [20040] * 3,
                "geolocation_lat": [-22.9, -22.9, -22.9],
                "geolocation_lng": [-43.2, -43.2, -43.2],
                "geolocation_city": ["rio de janeiro"] * 3,
                "geolocation_state": ["RJ", "RJ", "SP"],
            }
        )
        dq = DataQualityLog()
        out = StagingTransformer.clean_geolocation(df, dq)

        assert out["geolocation_state"].tolist() == ["RJ"]
        assert _rules(dq)["zip_with_multiple_states"] == 1


class TestCustomersAndSellersWithGeoReference:
    def _geo(self):
        geo = pd.DataFrame(
            {
                "geolocation_zip_code_prefix": ["09710", "01310"],
                "geolocation_city": ["sao bernardo do campo", "sao paulo"],
                "geolocation_state": ["SP", "SP"],
            }
        )
        return GeoReference.from_geolocation(geo)

    def test_seller_city_is_repaired_from_the_zip_and_flags_are_set(self):
        sellers = pd.DataFrame(
            {
                "seller_id": ["s1", "s2", "s3", "s4"],
                "seller_zip_code_prefix": [9710, 1310, 1310, 99999],
                "seller_city": ["sbc/sp", "vendas@loja.com.br", "sao paulo", "cidade x"],
                "seller_state": ["SP", "SP", "SP", "RS"],
            }
        )
        dq = DataQualityLog()
        out = StagingTransformer.clean_sellers(sellers, dq, self._geo()).set_index("seller_id")

        assert out.loc["s1", "seller_city"] == "sao bernardo do campo"   # 'sbc' does not exist in SP
        assert out.loc["s2", "seller_city"] == "sao paulo"                # e-mail typed as city, repaired by CEP
        assert out.loc["s3", "seller_city"] == "sao paulo"
        assert out["seller_region"].tolist() == ["Sudeste", "Sudeste", "Sudeste", "Sul"]
        assert out["zip_in_geolocation"].tolist() == [True, True, True, False]
        assert _rules(dq)["zip_not_in_geolocation"] == 1

    def test_customers_without_reference_still_clean_and_keep_unique_id_repeats(self):
        customers = pd.DataFrame(
            {
                "customer_id": ["c1", "c2"],
                "customer_unique_id": ["u1", "u1"],
                "customer_zip_code_prefix": [1310, 1310],
                "customer_city": ["São Paulo", "sao paulo"],
                "customer_state": ["sp", "SP"],
            }
        )
        dq = DataQualityLog()
        out = StagingTransformer.clean_customers(customers, dq)

        assert out["customer_city"].tolist() == ["sao paulo", "sao paulo"]
        assert out["zip_in_geolocation"].isna().all()
        assert len(out) == 2  # same person, one row per order: kept on purpose
        assert _rules(dq)["same_person_several_customer_ids"] == 2
