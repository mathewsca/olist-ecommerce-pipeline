"""
FabricaIA - Data Quality Module

Collects every data-quality finding made while cleaning the Olist tables
(what was wrong, how many rows, what was done about it) so the same evidence
can be shown in the notebooks and in the Streamlit "Analise Exploratoria" tab.

Two artifacts are produced and persisted in the staging schema:
  - staging.dq_issues          one row per (table, column, rule) finding
  - staging.dq_column_profile  null / distinct profile per column, raw vs staging

The classes here are pure (no DB access) except DataQualityLog.persist and
persist_profile, which are thin wrappers around DataFrame.to_sql.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import pandas as pd
from sqlalchemy import Engine, bindparam, inspect, text

logger = logging.getLogger(__name__)

SEVERITIES = ("info", "low", "medium", "high")
ACTIONS = ("fixed", "flagged", "dropped", "imputed", "nullified", "kept")

ISSUES_TABLE = "dq_issues"
PROFILE_TABLE = "dq_column_profile"

ISSUE_COLUMNS = [
    "table_name",
    "column_name",
    "rule",
    "description",
    "severity",
    "action",
    "rows_affected",
    "total_rows",
    "pct_affected",
    "examples",
    "checked_at",
]


@dataclass
class DQIssue:
    """One finding: a rule evaluated on a column and how many rows it touched."""

    table_name: str
    column_name: str
    rule: str
    description: str
    severity: str
    action: str
    rows_affected: int
    total_rows: int
    examples: str = ""

    @property
    def pct_affected(self) -> float:
        return round(100.0 * self.rows_affected / self.total_rows, 4) if self.total_rows else 0.0


class DataQualityLog:
    """Accumulates DQIssue records during a cleaning run."""

    def __init__(self) -> None:
        self.issues: List[DQIssue] = []

    def add(
        self,
        table_name: str,
        column_name: str,
        rule: str,
        rows_affected: int,
        total_rows: int,
        action: str,
        description: str = "",
        severity: str = "medium",
        examples: Optional[str] = None,
    ) -> None:
        """
        Record one finding. Findings with rows_affected == 0 are kept on purpose:
        they document that the rule was evaluated and the data passed.
        """
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
        self.issues.append(
            DQIssue(
                table_name=table_name,
                column_name=column_name,
                rule=rule,
                description=description,
                severity=severity,
                action=action,
                rows_affected=int(rows_affected),
                total_rows=int(total_rows),
                examples=examples or "",
            )
        )

    def extend(self, other: "DataQualityLog") -> None:
        self.issues.extend(other.issues)

    def to_frame(self) -> pd.DataFrame:
        checked_at = datetime.now(timezone.utc).replace(tzinfo=None)
        rows = [
            {
                "table_name": i.table_name,
                "column_name": i.column_name,
                "rule": i.rule,
                "description": i.description,
                "severity": i.severity,
                "action": i.action,
                "rows_affected": i.rows_affected,
                "total_rows": i.total_rows,
                "pct_affected": i.pct_affected,
                "examples": i.examples,
                "checked_at": checked_at,
            }
            for i in self.issues
        ]
        return pd.DataFrame(rows, columns=ISSUE_COLUMNS)

    def persist(self, engine: Engine, schema: Optional[str] = "staging") -> int:
        """
        Write the findings to <schema>.dq_issues. Rows previously logged for the
        same table_name are replaced, so re-running a single staging task never
        duplicates or wipes the findings of the others.
        """
        frame = self.to_frame()
        if frame.empty:
            return 0
        _replace_rows_for_tables(engine, schema, ISSUES_TABLE, "table_name", frame["table_name"].unique())
        frame.to_sql(ISSUES_TABLE, engine, schema=schema, if_exists="append", index=False)
        logger.info("Persisted %s data-quality findings to %s.%s", len(frame), schema, ISSUES_TABLE)
        return len(frame)


def sample_changes(before: pd.Series, after: pd.Series, n: int = 3) -> str:
    """
    Build a short "old -> new" string from the first n rows that changed,
    used as the `examples` column of a finding.
    """
    both = pd.DataFrame({"before": before, "after": after})
    b, a = both["before"].astype("string"), both["after"].astype("string")
    differs = (b != a).fillna(False).astype(bool) | (b.isna() != a.isna())
    changed = both[differs].drop_duplicates().head(n)
    pieces = [f"{_short(b)} -> {_short(a)}" for b, a in zip(changed["before"], changed["after"])]
    return "; ".join(pieces)


def sample_values(values: Iterable, n: int = 3) -> str:
    """Return up to n distinct values as a short string, for findings without a fix."""
    seen: List[str] = []
    for value in values:
        shown = _short(value)
        if shown not in seen:
            seen.append(shown)
        if len(seen) >= n:
            break
    return "; ".join(seen)


def _short(value, limit: int = 60) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "<null>"
    shown = str(value).replace("\n", " ").replace("\r", " ")
    return shown if len(shown) <= limit else shown[: limit - 3] + "..."


def profile_dataframe(df: pd.DataFrame, table_name: str, stage: str) -> pd.DataFrame:
    """
    Column-level profile (dtype, nulls, distinct values) of a DataFrame.

    Args:
        df: The table to profile.
        table_name: Logical table name, e.g. "customers".
        stage: "raw" or "staging" - lets the dashboard compare before/after.
    """
    rows = []
    total = len(df)
    for column in df.columns:
        series = df[column]
        nulls = int(series.isna().sum())
        try:
            distinct = int(series.nunique(dropna=True))
        except TypeError:  # unhashable cell values
            distinct = -1
        rows.append(
            {
                "table_name": table_name,
                "stage": stage,
                "column_name": column,
                "dtype": str(series.dtype),
                "total_rows": total,
                "null_count": nulls,
                "null_pct": round(100.0 * nulls / total, 4) if total else 0.0,
                "distinct_count": distinct,
            }
        )
    return pd.DataFrame(rows)


def persist_profile(engine: Engine, profile: pd.DataFrame, schema: Optional[str] = "staging") -> int:
    """Replace the profile rows of the tables present in `profile`."""
    if profile.empty:
        return 0
    _replace_rows_for_tables(engine, schema, PROFILE_TABLE, "table_name", profile["table_name"].unique())
    profile.to_sql(PROFILE_TABLE, engine, schema=schema, if_exists="append", index=False)
    return len(profile)


def _replace_rows_for_tables(engine: Engine, schema: Optional[str], table: str, key_column: str, keys) -> None:
    """DELETE the rows of `keys` from schema.table when the table already exists."""
    if not inspect(engine).has_table(table, schema=schema):
        return
    qualified = f"{schema}.{table}" if schema else table
    statement = text(f"DELETE FROM {qualified} WHERE {key_column} IN :keys").bindparams(
        bindparam("keys", expanding=True)
    )
    with engine.begin() as conn:
        conn.execute(statement, {"keys": list(keys)})
