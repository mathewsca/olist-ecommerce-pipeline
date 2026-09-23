"""
FabricaIA ETL - Integration test for the full raw -> staging -> dw pipeline.

Spins up an ephemeral Postgres via testcontainers (requires Docker to be
available) and runs the pipeline against a small fixture copy of the Olist
CSVs (tests/fixtures/olist_sample/). Run with `make test-integration`.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine

from src.etl.dw_builder import DWBuilder
from src.etl.raw_loader import RawLoader
from src.etl.staging_transform import StagingTransformer
from src.etl.validations import DWValidator

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "olist_sample"


@pytest.fixture(scope="module")
def engine():
    testcontainers_postgres = pytest.importorskip("testcontainers.postgres")
    with testcontainers_postgres.PostgresContainer("postgres:13") as postgres:
        yield create_engine(postgres.get_connection_url())


@pytest.fixture(scope="module")
def populated_dw(engine):
    RawLoader(engine, source_dir=str(FIXTURES_DIR)).load_all()

    builder = DWBuilder(engine)
    builder.init_schemas()

    StagingTransformer(engine).build_all()
    builder.build_all_dimensions()
    builder.build_all_facts()
    return engine


class TestRawToDWPipeline:
    def test_dimensions_are_populated(self, populated_dw):
        import pandas as pd

        customers = pd.read_sql_table("dim_customers", populated_dw, schema="dw")
        products = pd.read_sql_table("dim_products", populated_dw, schema="dw")

        assert len(customers) == 3
        assert len(products) == 2
        assert "furniture_decor" in products["product_category_name_english"].values

    def test_facts_are_populated(self, populated_dw):
        import pandas as pd

        order_items = pd.read_sql_table("fact_order_items", populated_dw, schema="dw")
        orders = pd.read_sql_table("fact_orders", populated_dw, schema="dw")

        assert len(order_items) == 3
        assert len(orders) == 3
        assert order_items["customer_key"].notna().all()
        assert order_items["product_key"].notna().all()

    def test_validation_passes(self, populated_dw):
        result = DWValidator(populated_dw).run_all()

        assert result.passed, result.failures
