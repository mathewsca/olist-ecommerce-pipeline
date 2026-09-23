"""
FabricaIA - Staging Transform Module

Cleans, types and deduplicates the raw Olist tables into the "staging"
schema. The clean_* methods are pure DataFrame-in/DataFrame-out functions
(no DB access) so they can be unit tested without a live Postgres; the
build_stg_* methods are thin wrappers that read from raw.* and write to
staging.*.

Every clean_* method accepts an optional DataQualityLog. When given, it records
what was wrong (rule, rows affected, action, examples); the build_stg_* wrappers
persist that log to staging.dq_issues together with a raw-vs-staging column
profile (staging.dq_column_profile), which the dashboard reads.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import Engine, inspect, text

from src.etl import cleaning as cl
from src.etl.db import to_sql_fast
from src.etl.data_quality import DataQualityLog, persist_profile, profile_dataframe, sample_values

logger = logging.getLogger(__name__)


@dataclass
class GeoReference:
    """
    Lookups derived from the cleaned geolocation table, used to repair and validate the
    city/CEP of customers and sellers.
    """

    zip_to_city: Dict[str, str] = field(default_factory=dict)
    cities_by_state: Dict[str, set] = field(default_factory=dict)

    @classmethod
    def from_geolocation(cls, geo: pd.DataFrame) -> "GeoReference":
        valid = geo.dropna(subset=["geolocation_city"])
        return cls(
            zip_to_city=dict(zip(valid["geolocation_zip_code_prefix"], valid["geolocation_city"])),
            cities_by_state={
                uf: set(group["geolocation_city"]) for uf, group in valid.groupby("geolocation_state")
            },
        )

    def knows_zip(self, zip_series: pd.Series) -> pd.Series:
        return zip_series.isin(self.zip_to_city.keys())


class StagingTransformer:
    """Builds the staging layer from the raw Olist tables."""

    def __init__(self, engine: Engine, schema_raw: str = "raw", schema_staging: str = "staging"):
        self.engine = engine
        self.schema_raw = schema_raw
        self.schema_staging = schema_staging

    def ensure_schema(self) -> None:
        """Create the staging schema if it doesn't exist yet."""
        with self.engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema_staging}"))

    def _read_raw(self, table_name: str) -> pd.DataFrame:
        return pd.read_sql_table(table_name, self.engine, schema=self.schema_raw)

    def _write_staging(self, df: pd.DataFrame, table_name: str) -> int:
        to_sql_fast(df, table_name, self.engine, schema=self.schema_staging, if_exists="replace")
        logger.info("Loaded %s rows into %s.%s", len(df), self.schema_staging, table_name)
        return len(df)

    def _finalize(self, name: str, raw: pd.DataFrame, clean: pd.DataFrame, dq: DataQualityLog) -> int:
        """Persist findings + before/after profile, then write staging.stg_<name>."""
        dq.persist(self.engine, self.schema_staging)
        profile = pd.concat(
            [profile_dataframe(raw, name, "raw"), profile_dataframe(clean, name, "staging")], ignore_index=True
        )
        persist_profile(self.engine, profile, self.schema_staging)
        return self._write_staging(clean, f"stg_{name}")

    def _geo_reference(self) -> Optional[GeoReference]:
        """GeoReference from staging.stg_geolocation, or None when it was not built yet."""
        if not inspect(self.engine).has_table("stg_geolocation", schema=self.schema_staging):
            logger.warning("stg_geolocation not found: city/CEP repair will be skipped")
            return None
        geo = pd.read_sql_table("stg_geolocation", self.engine, schema=self.schema_staging)
        return GeoReference.from_geolocation(geo)

    # ------------------------------------------------------------------
    # Pure transforms (unit-testable without a database)
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_place_table(
        df: pd.DataFrame,
        prefix: str,
        table: str,
        key: str,
        dq: Optional[DataQualityLog],
        geo: Optional[GeoReference],
    ) -> pd.DataFrame:
        """Shared logic of customers/sellers: key, CEP, UF, city and geographic flags."""
        out = df.copy()
        out[key] = cl.as_string_id(out[key])
        out = cl.report_duplicates(out, [key], table, dq)
        zip_col, city_col, state_col = f"{prefix}_zip_code_prefix", f"{prefix}_city", f"{prefix}_state"

        out[zip_col] = cl.standardize_zip(out[zip_col], table, zip_col, dq)
        out[state_col] = cl.standardize_state(out[state_col], table, state_col, dq)
        out[city_col] = cl.standardize_city(out[city_col], out[state_col], table, city_col, dq)
        if geo is not None:
            out[city_col] = cl.fill_city_from_zip(out[city_col], out[zip_col], geo.zip_to_city, table, city_col, dq)
            out[city_col] = cl.repair_city_with_reference(
                out[city_col], out[state_col], out[zip_col], geo.zip_to_city, geo.cities_by_state,
                table, city_col, dq,
            )
        out[f"{prefix}_region"] = cl.state_to_region(out[state_col])
        out["zip_state_mismatch"] = cl.zip_state_mismatch(out[zip_col], out[state_col])
        out["zip_in_geolocation"] = (
            geo.knows_zip(out[zip_col]).astype("boolean") if geo is not None
            else pd.Series(pd.NA, index=out.index, dtype="boolean")
        )
        if dq is not None:
            dq.add(table, zip_col, "zip_state_mismatch", int(out["zip_state_mismatch"].sum()), len(out), "flagged",
                   "UF do cadastro difere da UF que a faixa de CEP dos Correios indica.", "medium")
            if geo is not None:
                missing_geo = ~out["zip_in_geolocation"].fillna(False).astype(bool) & out[zip_col].notna()
                dq.add(table, zip_col, "zip_not_in_geolocation", int(missing_geo.sum()), len(out), "flagged",
                       "CEP inexistente na geolocation: não dá para plotar no mapa nem reparar a cidade.", "low",
                       sample_values(out.loc[missing_geo, zip_col]))
        return out

    @staticmethod
    def clean_customers(
        df: pd.DataFrame, dq: Optional[DataQualityLog] = None, geo: Optional[GeoReference] = None
    ) -> pd.DataFrame:
        out = StagingTransformer._clean_place_table(df, "customer", "customers", "customer_id", dq, geo)
        if "customer_unique_id" in out.columns:
            out["customer_unique_id"] = cl.as_string_id(out["customer_unique_id"])
            repeated = out["customer_unique_id"].duplicated(keep=False)
            if dq is not None:
                dq.add("customers", "customer_unique_id", "same_person_several_customer_ids", int(repeated.sum()),
                       len(out), "kept",
                       "O Olist cria um customer_id novo a cada pedido: a mesma pessoa (customer_unique_id) aparece "
                       "em várias linhas. Mantido de propósito - é o que permite medir recompra.", "info")
        return out

    @staticmethod
    def clean_sellers(
        df: pd.DataFrame, dq: Optional[DataQualityLog] = None, geo: Optional[GeoReference] = None
    ) -> pd.DataFrame:
        return StagingTransformer._clean_place_table(df, "seller", "sellers", "seller_id", dq, geo)

    @staticmethod
    def clean_orders(df: pd.DataFrame, dq: Optional[DataQualityLog] = None) -> pd.DataFrame:
        out = df.copy()
        out["order_id"] = cl.as_string_id(out["order_id"])
        out = cl.report_duplicates(out, ["order_id"], "orders", dq)
        if "customer_id" in out.columns:
            out["customer_id"] = cl.as_string_id(out["customer_id"])

        timestamp_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
        for col in timestamp_cols:
            if col in out.columns:
                out[col] = cl.parse_datetime(out[col], "orders", col, dq, min_date="2016-01-01", max_date="2019-12-31")

        out["order_status"] = out["order_status"].astype("string").str.lower().str.strip()
        unknown_status = out["order_status"].notna() & ~out["order_status"].isin(cl.ORDER_STATUSES)
        if dq is not None:
            dq.add("orders", "order_status", "unknown_status", int(unknown_status.sum()), len(out), "flagged",
                   "Status fora dos 8 valores conhecidos do Olist.", "high", sample_values(out.loc[unknown_status, "order_status"]))

        # Timeline consistency: purchase <= approval <= carrier <= customer. Flagged, never dropped -
        # the anomaly is a real fact about the business (or a system clock issue) worth showing.
        p, a, c, d = (out.get(k) for k in timestamp_cols[:4])
        flags = {
            "flag_approved_before_purchase": (a < p, "Aprovação anterior à compra."),
            "flag_carrier_before_purchase": (c < p, "Postagem na transportadora anterior à compra."),
            "flag_carrier_before_approval": (c < a, "Postagem na transportadora anterior à aprovação do pagamento."),
            "flag_delivered_before_carrier": (d < c, "Entrega ao cliente anterior à postagem na transportadora."),
            "flag_delivered_without_date": (
                (out["order_status"] == "delivered") & d.isna(), "Status 'delivered' mas sem data de entrega."),
            "flag_undelivered_with_delivery_date": (
                (out["order_status"] != "delivered") & d.notna(), "Pedido não entregue (ex.: cancelado) mas com data de entrega."),
        }
        for name, (mask, description) in flags.items():
            out[name] = mask.fillna(False).astype(bool)
            if dq is not None:
                dq.add("orders", name.replace("flag_", ""), name.replace("flag_", ""), int(out[name].sum()), len(out),
                       "flagged", description, "medium")
        out["has_date_anomaly"] = out[list(flags)].any(axis=1)
        return out

    @staticmethod
    def clean_order_items(df: pd.DataFrame, dq: Optional[DataQualityLog] = None) -> pd.DataFrame:
        out = df.copy()
        out["order_id"] = cl.as_string_id(out["order_id"])
        out["order_item_id"] = pd.to_numeric(out["order_item_id"], errors="coerce").astype("Int64")
        out = cl.report_duplicates(out, ["order_id", "order_item_id"], "order_items", dq)
        for col in ("product_id", "seller_id"):
            if col in out.columns:
                out[col] = cl.as_string_id(out[col])

        out["price"] = cl.coerce_numeric(out["price"], "order_items", "price", dq, minimum=0, min_inclusive=False, decimals=2)
        out["freight_value"] = cl.coerce_numeric(
            out["freight_value"], "order_items", "freight_value", dq, minimum=0, decimals=2
        )
        if "shipping_limit_date" in out.columns:
            out["shipping_limit_date"] = cl.parse_datetime(
                out["shipping_limit_date"], "order_items", "shipping_limit_date", dq,
                min_date="2016-01-01", max_date="2019-12-31",
            )
        out["is_price_outlier"] = cl.iqr_outlier_mask(out["price"], k=3.0)
        out["is_freight_above_price"] = (out["freight_value"] > out["price"]).fillna(False)
        if dq is not None:
            dq.add("order_items", "price", "price_extreme_outlier", int(out["is_price_outlier"].sum()), len(out), "flagged",
                   "Preço acima de Q3 + 3*IQR. Mantido (pode ser produto caro legítimo), mas sinalizado para "
                   "não distorcer médias.", "low")
            dq.add("order_items", "freight_value", "freight_above_price", int(out["is_freight_above_price"].sum()), len(out),
                   "flagged", "Frete maior que o preço do item.", "low")
        return out

    @staticmethod
    def clean_order_payments(df: pd.DataFrame, dq: Optional[DataQualityLog] = None) -> pd.DataFrame:
        out = df.copy()
        out["order_id"] = cl.as_string_id(out["order_id"])
        out["payment_sequential"] = pd.to_numeric(out["payment_sequential"], errors="coerce").astype("Int64")
        out = cl.report_duplicates(out, ["order_id", "payment_sequential"], "order_payments", dq)

        out["payment_value"] = cl.coerce_numeric(out["payment_value"], "order_payments", "payment_value", dq, minimum=0, decimals=2)
        out["payment_type"] = out["payment_type"].astype("string").str.lower().str.strip()
        installments = cl.coerce_numeric(out["payment_installments"], "order_payments", "payment_installments", dq, minimum=0)
        # Every payment has at least one installment; 0 is a data-entry artifact.
        zero_installments = (installments == 0).fillna(False)
        out["payment_installments"] = installments.mask(zero_installments, 1).round().astype("Int64")
        out["is_valid_payment"] = (
            out["payment_type"].notna() & (out["payment_type"] != "not_defined") & out["payment_value"].notna()
        ).astype(bool)
        if dq is not None:
            dq.add("order_payments", "payment_installments", "zero_installments", int(zero_installments.sum()), len(out), "fixed",
                   "Parcelas = 0 é impossível (mínimo 1); ajustado para 1.", "medium")
            dq.add("order_payments", "payment_type", "not_defined_payment_type", int((out["payment_type"] == "not_defined").sum()),
                   len(out), "flagged", "Tipo de pagamento 'not_defined' (valor zero): marcado como inválido e "
                   "excluído das análises de pagamento.", "medium")
            dq.add("order_payments", "payment_value", "zero_value_payment", int((out["payment_value"] == 0).sum()), len(out),
                   "flagged", "Pagamento de valor zero (normalmente voucher).", "low")
        return out

    @staticmethod
    def clean_order_reviews(df: pd.DataFrame, dq: Optional[DataQualityLog] = None) -> pd.DataFrame:
        out = df.copy()
        has_order = "order_id" in out.columns
        # review_id is NOT a unique key in Olist: the same review_id is attached to different orders.
        # The real key is (review_id, order_id); deduplicating on review_id alone silently drops valid reviews.
        key = ["review_id", "order_id"] if has_order else ["review_id"]
        out["review_id"] = cl.as_string_id(out["review_id"])
        if has_order:
            out["order_id"] = cl.as_string_id(out["order_id"])
            shared = out["review_id"].duplicated(keep=False)
            if dq is not None:
                dq.add("order_reviews", "review_id", "review_id_shared_by_several_orders", int(shared.sum()), len(out), "kept",
                       "O mesmo review_id aparece em pedidos diferentes. A chave correta é (review_id, order_id); "
                       "deduplicar só por review_id descartaria avaliações válidas.", "medium")
        out = cl.report_duplicates(out, key, "order_reviews", dq)

        out["review_score"] = cl.coerce_numeric(out["review_score"], "order_reviews", "review_score", dq, minimum=1, maximum=5).round().astype("Int64")
        out["review_creation_date"] = cl.parse_datetime(out["review_creation_date"], "order_reviews", "review_creation_date", dq)
        out["review_answer_timestamp"] = cl.parse_datetime(out["review_answer_timestamp"], "order_reviews", "review_answer_timestamp", dq)

        for col in ("review_comment_title", "review_comment_message"):
            if col in out.columns:
                out[col] = cl.clean_free_text(out[col], "order_reviews", col, dq)
        if "review_comment_message" in out.columns:
            features = cl.text_features(out["review_comment_message"])
            out = pd.concat([out, features], axis=1)
            out["review_comment_message_norm"] = cl.normalize_for_nlp(out["review_comment_message"])
            if dq is not None:
                dq.add("order_reviews", "review_comment_message", "no_comment_message", int(out["review_comment_message"].isna().sum()),
                       len(out), "kept", "Avaliação sem comentário: normal (a maioria só dá a nota). Mantido como NULL.", "info")
                dq.add("order_reviews", "review_comment_message", "low_information_comment", int(out["is_low_information"].sum()),
                       len(out), "flagged", "Comentário de até 3 caracteres ou só pontuação/números ('.', 'ok', '10'): "
                       "sinalizado para filtrar em análises de texto.", "low",
                       sample_values(out.loc[out["is_low_information"], "review_comment_message"]))
                dq.add("order_reviews", "review_comment_message", "shouting_all_caps", int(out["is_shouting"].sum()), len(out),
                       "flagged", "Comentário todo em MAIÚSCULAS (geralmente reclamação). Mantido como veio.", "info")

        answered_before = (out["review_answer_timestamp"] < out["review_creation_date"]).fillna(False)
        if dq is not None:
            dq.add("order_reviews", "review_answer_timestamp", "answer_before_creation", int(answered_before.sum()), len(out),
                   "flagged", "Resposta anterior à criação da avaliação.", "medium")

        # A few orders have more than one review; mark the most recent so order-level metrics do not double count.
        if has_order:
            ranked = out.sort_values(["order_id", "review_answer_timestamp", "review_creation_date", "review_id"],
                                     ascending=[True, False, False, False], na_position="last")
            latest_index = ranked.drop_duplicates("order_id", keep="first").index
            out["is_latest_for_order"] = out.index.isin(latest_index)
            if dq is not None:
                dq.add("order_reviews", "order_id", "orders_with_multiple_reviews", int(out["order_id"].duplicated(keep=False).sum()),
                       len(out), "flagged", "Pedidos com mais de uma avaliação: a mais recente recebe "
                       "is_latest_for_order = TRUE.", "low")
        else:
            out["is_latest_for_order"] = True
        return out

    @staticmethod
    def clean_products(
        df: pd.DataFrame, category_translation: pd.DataFrame, dq: Optional[DataQualityLog] = None
    ) -> pd.DataFrame:
        out = df.copy()
        out["product_id"] = cl.as_string_id(out["product_id"])
        out = cl.report_duplicates(out, ["product_id"], "products", dq)
        # The source columns are misspelled ("lenght").
        out = out.rename(columns={"product_name_lenght": "product_name_length",
                                  "product_description_lenght": "product_description_length"})

        # Physical attributes: zero/negative weight and dimensions are impossible -> NULL, then imputed.
        limits = {"product_weight_g": (0, 100_000), "product_length_cm": (0, 300),
                  "product_height_cm": (0, 300), "product_width_cm": (0, 300)}
        for col, (low, high) in limits.items():
            if col in out.columns:
                out[col] = cl.coerce_numeric(out[col], "products", col, dq, minimum=low, maximum=high, min_inclusive=False)
        for col in ("product_photos_qty", "product_name_length", "product_description_length"):
            if col in out.columns:
                out[col] = cl.coerce_numeric(out[col], "products", col, dq, minimum=0)

        category_pt = cl.standardize_category(out["product_category_name"], "products", "product_category_name", dq)
        out["category_group"] = cl.category_group(category_pt)
        english = cl.translate_category(category_pt, category_translation, "products", "product_category_name_english", dq)
        out["product_category_name"] = cl.fill_missing(
            category_pt, "unknown", "products", "product_category_name", dq,
            rule="missing_category", description="Produto sem categoria recebe o rótulo 'unknown' (sentinela), "
            "em vez de ficar NULL e sumir dos agrupamentos.")
        out["product_category_name_english"] = english

        if "product_photos_qty" in out.columns:
            out["has_listing_metadata"] = out["product_photos_qty"].notna()
            if dq is not None:
                dq.add("products", "product_photos_qty", "missing_listing_metadata", int((~out["has_listing_metadata"]).sum()), len(out),
                       "flagged", "Anúncio sem fotos/tamanho de título e descrição (mesmos produtos sem categoria). "
                       "Mantido como NULL: 0 fotos seria uma informação falsa.", "medium")

        size_cols = [c for c in limits if c in out.columns]
        out, imputed = cl.impute_by_group_median(out, size_cols, "product_category_name", "products", dq)
        out["dimensions_imputed"] = imputed
        if {"product_length_cm", "product_height_cm", "product_width_cm"} <= set(out.columns):
            out["product_volume_cm3"] = out["product_length_cm"] * out["product_height_cm"] * out["product_width_cm"]
        return out

    @staticmethod
    def clean_geolocation(df: pd.DataFrame, dq: Optional[DataQualityLog] = None) -> pd.DataFrame:
        """
        The raw geolocation table has many duplicate lat/lng rows per zip prefix, some GPS points
        outside Brazil and city names spelled several ways. Aggregate to one row per zip prefix:
        mode of the standardized city/state and mean lat/lng of the points that survive the outlier filter.
        """
        out = df.copy()
        total = len(out)
        exact_duplicates = int(out.duplicated().sum())
        if dq is not None:
            dq.add("geolocation", "*", "exact_duplicate_rows", exact_duplicates, total, "dropped",
                   "Linhas 100% idênticas (mesmo CEP, cidade, lat e lng): um CEP aparece dezenas de vezes.", "high")

        zip_col = "geolocation_zip_code_prefix"
        out[zip_col] = cl.standardize_zip(out[zip_col], "geolocation", zip_col, dq)
        out["geolocation_state"] = cl.standardize_state(out["geolocation_state"], "geolocation", "geolocation_state", dq)
        out["geolocation_city"] = cl.standardize_city(
            out["geolocation_city"], out["geolocation_state"], "geolocation", "geolocation_city", dq,
            repair_with_vocabulary=True,
        )
        out["lat"] = cl.coerce_numeric(out["geolocation_lat"], "geolocation", "geolocation_lat", dq)
        out["lng"] = cl.coerce_numeric(out["geolocation_lng"], "geolocation", "geolocation_lng", dq)
        out = out.dropna(subset=[zip_col]).copy()

        in_brazil = out["lat"].between(*cl.BRAZIL_LAT_RANGE) & out["lng"].between(*cl.BRAZIL_LNG_RANGE)
        keep = cl.filter_gps_outliers(out["lat"], out["lng"], out[zip_col])
        if dq is not None:
            dq.add("geolocation", "geolocation_lat/lng", "coordinates_outside_brazil", int((~in_brazil).sum()), total, "dropped",
                   "Pontos de GPS fora do retângulo do Brasil (ex.: Argentina, Europa) - descartados antes de "
                   "calcular o centro do CEP.", "high")
            dq.add("geolocation", "geolocation_lat/lng", "coordinates_far_from_zip_cluster", int((in_brazil & ~keep).sum()), total,
                   "dropped", "Ponto dentro do Brasil, mas a mais de 1 grau da mediana dos demais pontos do mesmo CEP.", "medium")

        # Mode of city / state per zip via counts (much faster than groupby.apply(mode)).
        def _mode_by_zip(column: str) -> pd.Series:
            counts = out.dropna(subset=[column]).groupby([zip_col, column]).size().reset_index(name="n")
            counts = counts.sort_values([zip_col, "n", column], ascending=[True, False, True])
            return counts.drop_duplicates(zip_col).set_index(zip_col)[column]

        states_per_zip = out.dropna(subset=["geolocation_state"]).groupby(zip_col)["geolocation_state"].nunique()
        cities_per_zip = out.dropna(subset=["geolocation_city"]).groupby(zip_col)["geolocation_city"].nunique()
        if dq is not None:
            dq.add("geolocation", "geolocation_state", "zip_with_multiple_states", int((states_per_zip > 1).sum()), int(len(states_per_zip)),
                   "fixed", "CEP associado a mais de uma UF: fica a UF mais frequente.", "high")
            dq.add("geolocation", "geolocation_city", "zip_with_multiple_cities", int((cities_per_zip > 1).sum()), int(len(cities_per_zip)),
                   "fixed", "CEP com mais de uma grafia/cidade (mesmo depois de padronizar): fica a mais frequente.", "medium")

        points = out.loc[keep].groupby(zip_col).agg(avg_lat=("lat", "mean"), avg_lng=("lng", "mean"), n_points=("lat", "size"))
        agg = pd.DataFrame({
            "geolocation_city": _mode_by_zip("geolocation_city"),
            "geolocation_state": _mode_by_zip("geolocation_state"),
        }).join(points)
        agg["n_points"] = agg["n_points"].fillna(0).astype("Int64")
        agg["n_points_discarded"] = (out.groupby(zip_col).size().reindex(agg.index) - agg["n_points"]).astype("Int64")
        agg = agg.reset_index()
        agg["geolocation_region"] = cl.state_to_region(agg["geolocation_state"])
        agg["zip_state_mismatch"] = cl.zip_state_mismatch(agg[zip_col], agg["geolocation_state"])
        if dq is not None:
            dq.add("geolocation", zip_col, "zip_without_valid_coordinates", int(agg["avg_lat"].isna().sum()), len(agg), "flagged",
                   "CEP em que todos os pontos foram descartados: fica sem latitude/longitude.", "medium")
            dq.add("geolocation", zip_col, "aggregated_to_one_row_per_zip", total - len(agg), total, "dropped",
                   f"{total:,} pontos brutos viram {len(agg):,} linhas (uma por CEP).".replace(",", "."), "info")
        return agg

    # ------------------------------------------------------------------
    # DB-backed builders
    # ------------------------------------------------------------------

    def build_stg_geolocation(self) -> int:
        raw, dq = self._read_raw("geolocation"), DataQualityLog()
        return self._finalize("geolocation", raw, self.clean_geolocation(raw, dq), dq)

    def build_stg_customers(self) -> int:
        raw, dq = self._read_raw("customers"), DataQualityLog()
        return self._finalize("customers", raw, self.clean_customers(raw, dq, self._geo_reference()), dq)

    def build_stg_sellers(self) -> int:
        raw, dq = self._read_raw("sellers"), DataQualityLog()
        return self._finalize("sellers", raw, self.clean_sellers(raw, dq, self._geo_reference()), dq)

    def build_stg_orders(self) -> int:
        raw, dq = self._read_raw("orders"), DataQualityLog()
        return self._finalize("orders", raw, self.clean_orders(raw, dq), dq)

    def build_stg_order_items(self) -> int:
        raw, dq = self._read_raw("order_items"), DataQualityLog()
        return self._finalize("order_items", raw, self.clean_order_items(raw, dq), dq)

    def build_stg_order_payments(self) -> int:
        raw, dq = self._read_raw("order_payments"), DataQualityLog()
        return self._finalize("order_payments", raw, self.clean_order_payments(raw, dq), dq)

    def build_stg_order_reviews(self) -> int:
        raw, dq = self._read_raw("order_reviews"), DataQualityLog()
        return self._finalize("order_reviews", raw, self.clean_order_reviews(raw, dq), dq)

    def build_stg_products(self) -> int:
        raw, dq = self._read_raw("products"), DataQualityLog()
        category_translation = self._read_raw("category_translation")
        return self._finalize("products", raw, self.clean_products(raw, category_translation, dq), dq)

    # ------------------------------------------------------------------
    # Cross-table checks (need several staging tables at once)
    # ------------------------------------------------------------------

    def run_cross_table_checks(self) -> int:
        """
        Referential and business-rule checks that span tables (orders without items, orphan keys,
        payment total vs items total...). Findings are logged under table_name = 'cross_table'.
        """
        read = lambda name: pd.read_sql_table(f"stg_{name}", self.engine, schema=self.schema_staging)  # noqa: E731
        orders, items, payments = read("orders"), read("order_items"), read("order_payments")
        reviews, products, sellers, customers = read("order_reviews"), read("products"), read("sellers"), read("customers")
        dq = DataQualityLog()
        n = len(orders)

        def check(column, rule, mask, description, severity, action="flagged", examples=None):
            dq.add("cross_table", column, rule, int(mask.sum()), len(mask), action, description, severity, examples)

        no_items = ~orders["order_id"].isin(items["order_id"])
        status_mix = orders.loc[no_items, "order_status"].value_counts().head(4)
        check("orders.order_id", "orders_without_items", no_items,
              "Pedidos sem nenhum item (quase todos cancelados/indisponíveis): entram no fato de pedidos, "
              "mas não geram receita.", "medium", examples="; ".join(f"{k}: {v}" for k, v in status_mix.items()))
        check("orders.order_id", "orders_without_payment", ~orders["order_id"].isin(payments["order_id"]),
              "Pedidos sem registro de pagamento.", "medium")
        check("orders.order_id", "orders_without_review", ~orders["order_id"].isin(reviews["order_id"]),
              "Pedidos sem avaliação (normal: nem todo cliente avalia).", "info", "kept")
        check("order_items.order_id", "items_orphan_order", ~items["order_id"].isin(orders["order_id"]),
              "Itens apontando para pedido inexistente.", "high")
        check("order_items.product_id", "items_orphan_product", ~items["product_id"].isin(products["product_id"]),
              "Itens apontando para produto inexistente.", "high")
        check("order_items.seller_id", "items_orphan_seller", ~items["seller_id"].isin(sellers["seller_id"]),
              "Itens apontando para vendedor inexistente.", "high")
        check("order_payments.order_id", "payments_orphan_order", ~payments["order_id"].isin(orders["order_id"]),
              "Pagamentos apontando para pedido inexistente.", "high")
        check("order_reviews.order_id", "reviews_orphan_order", ~reviews["order_id"].isin(orders["order_id"]),
              "Avaliações apontando para pedido inexistente.", "high")
        check("orders.customer_id", "orders_orphan_customer", ~orders["customer_id"].isin(customers["customer_id"]),
              "Pedidos apontando para cliente inexistente.", "high")

        items_total = items.assign(v=items["price"].fillna(0) + items["freight_value"].fillna(0)).groupby("order_id")["v"].sum()
        paid_total = payments[payments["is_valid_payment"]].groupby("order_id")["payment_value"].sum()
        both = pd.concat([items_total.rename("items"), paid_total.rename("paid")], axis=1).dropna()
        diff = (both["paid"] - both["items"]).abs()
        check("payments vs items", "payment_total_differs_from_items_total", diff > 1.0,
              "Soma dos pagamentos difere da soma (preço + frete) dos itens em mais de R$ 1,00. "
              "Costuma ser juros de parcelamento ou desconto/voucher.", "medium",
              examples=f"maior diferença: R$ {diff.max():.2f}")
        dq.add("cross_table", "orders", "orders_analyzed", n, n, "kept", "Total de pedidos avaliados nas checagens acima.", "info")
        return dq.persist(self.engine, self.schema_staging)

    def build_all(self) -> Dict[str, int]:
        """Build every staging table from its raw counterpart (geolocation first: others use it as reference)."""
        self.ensure_schema()
        counts = {
            "stg_geolocation": self.build_stg_geolocation(),
            "stg_customers": self.build_stg_customers(),
            "stg_sellers": self.build_stg_sellers(),
            "stg_orders": self.build_stg_orders(),
            "stg_order_items": self.build_stg_order_items(),
            "stg_order_payments": self.build_stg_order_payments(),
            "stg_order_reviews": self.build_stg_order_reviews(),
            "stg_products": self.build_stg_products(),
        }
        self.run_cross_table_checks()
        return counts
