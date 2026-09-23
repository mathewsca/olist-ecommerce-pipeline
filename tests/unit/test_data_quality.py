"""
FabricaIA ETL - Tests for the data-quality log (src/etl/data_quality.py).
"""

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.etl.data_quality import (
    DataQualityLog,
    persist_profile,
    profile_dataframe,
    sample_changes,
    sample_values,
)


class TestDataQualityLog:
    def test_add_and_to_frame_compute_percentage(self):
        dq = DataQualityLog()
        dq.add("customers", "zip", "lost_zeros", 25, 100, "fixed", "desc", "medium", "1 -> 01")
        frame = dq.to_frame()

        assert frame.loc[0, "pct_affected"] == 25.0
        assert frame.loc[0, "examples"] == "1 -> 01"
        assert list(frame.columns)[:3] == ["table_name", "column_name", "rule"]

    def test_zero_row_findings_are_kept_as_evidence_of_a_passed_check(self):
        dq = DataQualityLog()
        dq.add("orders", "status", "unknown_status", 0, 500, "flagged")
        assert dq.to_frame()["rows_affected"].tolist() == [0]

    def test_pct_is_zero_when_table_is_empty(self):
        dq = DataQualityLog()
        dq.add("t", "c", "r", 0, 0, "kept")
        assert dq.to_frame().loc[0, "pct_affected"] == 0.0

    def test_invalid_severity_or_action_is_rejected(self):
        dq = DataQualityLog()
        with pytest.raises(ValueError):
            dq.add("t", "c", "r", 1, 1, "fixed", severity="catastrophic")
        with pytest.raises(ValueError):
            dq.add("t", "c", "r", 1, 1, "deleted")

    def test_extend_merges_logs(self):
        first, second = DataQualityLog(), DataQualityLog()
        first.add("a", "c", "r", 1, 1, "fixed")
        second.add("b", "c", "r", 1, 1, "fixed")
        first.extend(second)
        assert [i.table_name for i in first.issues] == ["a", "b"]


class TestPersistence:
    def test_persist_replaces_only_the_tables_being_rerun(self):
        engine = create_engine("sqlite://")
        first = DataQualityLog()
        first.add("customers", "zip", "r1", 5, 10, "fixed")
        first.add("orders", "status", "r2", 1, 10, "flagged")
        first.persist(engine, schema=None)

        rerun = DataQualityLog()
        rerun.add("customers", "zip", "r1", 7, 10, "fixed")
        rerun.persist(engine, schema=None)

        stored = pd.read_sql("SELECT table_name, rule, rows_affected FROM dq_issues ORDER BY table_name", engine)
        assert stored["table_name"].tolist() == ["customers", "orders"]
        assert stored.loc[stored["table_name"] == "customers", "rows_affected"].iloc[0] == 7

    def test_persist_of_empty_log_writes_nothing(self):
        assert DataQualityLog().persist(create_engine("sqlite://"), schema=None) == 0

    def test_profile_persistence_replaces_previous_profile(self):
        engine = create_engine("sqlite://")
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "x", None]})
        profile = pd.concat([profile_dataframe(df, "t", "raw"), profile_dataframe(df, "t", "staging")])
        persist_profile(engine, profile, schema=None)
        persist_profile(engine, profile, schema=None)

        stored = pd.read_sql("SELECT * FROM dq_column_profile", engine)
        assert len(stored) == 4  # 2 columns x 2 stages, not duplicated


class TestHelpers:
    def test_profile_dataframe_counts_nulls_and_distincts(self):
        profile = profile_dataframe(pd.DataFrame({"a": [1, None, 1, 2]}), "t", "raw")
        row = profile.iloc[0]
        assert (row["null_count"], row["null_pct"], row["distinct_count"], row["stage"]) == (1, 25.0, 2, "raw")

    def test_profile_survives_unhashable_cells(self):
        profile = profile_dataframe(pd.DataFrame({"a": [[1], [2]]}), "t", "raw")
        assert profile.iloc[0]["distinct_count"] == -1

    def test_sample_changes_lists_distinct_changed_pairs_and_handles_nulls(self):
        before = pd.Series(["são paulo", "são paulo", "rio", "sp"])
        after = pd.Series(["sao paulo", "sao paulo", "rio", None])
        assert sample_changes(before, after, n=5) == "são paulo -> sao paulo; sp -> <null>"

    def test_sample_values_deduplicates_and_truncates_long_text(self):
        assert sample_values(["a", "a", "b", "c", "d"], n=3) == "a; b; c"
        assert sample_values(["x" * 100], n=1).endswith("...")
        assert sample_values([None], n=1) == "<null>"
