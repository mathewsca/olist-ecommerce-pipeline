"""Aba 'Dicionário de Dados': modelo dimensional + amostra por tabela."""

import pandas as pd
import streamlit as st
from sqlalchemy import Engine

from src.dashboard.data_dictionary import DW_TABLES, QUALITY_TABLES, STAR_SCHEMA_OVERVIEW


@st.cache_data(ttl=300, show_spinner=False)
def _sample_rows(_engine: Engine, table_name: str, n: int = 5) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM dw.{table_name} LIMIT {n}", _engine)


@st.cache_data(ttl=300, show_spinner=False)
def _row_count(_engine: Engine, table_name: str) -> int:
    return int(pd.read_sql(f"SELECT COUNT(*) AS n FROM dw.{table_name}", _engine)["n"].iloc[0])


def render(engine: Engine) -> None:
    st.markdown(STAR_SCHEMA_OVERVIEW)

    table_name = st.selectbox(
        "Escolha uma tabela do Data Warehouse (schema `dw`)",
        options=list(DW_TABLES.keys()),
        format_func=lambda t: f"{t}  ·  {DW_TABLES[t]['kind']}",
    )
    meta = DW_TABLES[table_name]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### `dw.{table_name}`")
        st.write(meta["description"])
        if meta["kind"] == "Fato":
            st.info(f"**Grao:** {meta['grain']}")
    with col2:
        try:
            st.metric("Linhas na tabela", f"{_row_count(engine, table_name):,}".replace(",", "."))
        except Exception:
            st.metric("Linhas na tabela", "—")

    st.markdown("#### Colunas")
    cols_df = pd.DataFrame(
        [{"coluna": col, "descricao": desc} for col, desc in meta["columns"].items()]
    )
    st.dataframe(cols_df, width="stretch", hide_index=True)

    st.markdown("#### Amostra (5 linhas)")
    try:
        sample = _sample_rows(engine, table_name)
        if sample.empty:
            st.warning("Tabela ainda vazia: rode a DAG `olist_etl_pipeline` no Airflow primeiro")
        else:
            st.dataframe(sample, width="stretch")
    except Exception as exc:
        st.error(f"Não foi possível consultar `dw.{table_name}`: {exc}")

    st.divider()
    st.markdown("#### Tabelas de auditoria da limpeza (schema `staging`)")
    st.caption("Não fazem parte do esquema estrela: registram o que a limpeza encontrou e corrigiu (veja a aba Análise Exploratória).")
    for name, description in QUALITY_TABLES.items():
        st.markdown(f"- `{name}`: {description}")
