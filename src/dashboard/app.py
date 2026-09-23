"""
Fabrica IA - Olist Dashboard (Streamlit)

Three tabs:
  1. Dicionario de Dados - modelo dimensional + amostra por tabela (filtro).
  2. Analise Exploratoria - qualidade dos dados (erros e tratamentos), distribuicoes, correlacoes, comparacoes.
  3. Dashboard - analises com insights prontos para gestores/diretoria.

A barra lateral traz os filtros globais (periodo e estado do cliente) das abas 2 e 3.
"""

from datetime import date

import pandas as pd
import streamlit as st

from src.dashboard import queries as q
from src.dashboard import theme
from src.dashboard.tabs import dictionary_tab, exploratory_tab, insights_tab
from src.etl.db import get_engine

st.set_page_config(page_title="Olist - Fábrica de IA", page_icon="📦", layout="wide")

st.title("📦 Olist - Fábrica de IA")
st.caption("Data Warehouse construído a partir do *Brazilian E-Commerce Public Dataset by Olist*.")

engine = get_engine()

try:
    with engine.connect() as conn:
        summary = pd.read_sql(
            """
            SELECT
                (SELECT COUNT(*) FROM dw.fact_orders) AS pedidos,
                (SELECT COUNT(*) FROM dw.fact_order_items) AS itens,
                (SELECT COUNT(*) FROM dw.dim_customers) AS clientes,
                (SELECT COUNT(*) FROM dw.dim_sellers) AS vendedores,
                (SELECT COUNT(*) FROM dw.dim_products) AS produtos
            """,
            conn,
        ).iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Pedidos", theme.fmt_int(summary["pedidos"]))
    col2.metric("Itens vendidos", theme.fmt_int(summary["itens"]))
    col3.metric("Clientes", theme.fmt_int(summary["clientes"]))
    col4.metric("Vendedores", theme.fmt_int(summary["vendedores"]))
    col5.metric("Produtos", theme.fmt_int(summary["produtos"]))
except Exception as exc:  # noqa: BLE001 - surfaced to the student in the UI
    st.error(
        "Não foi possível consultar o Data Warehouse. Verifique se o Postgres "
        f"está de pé e se a DAG `olist_etl_pipeline` já foi executada.\n\nDetalhe: {exc}"
    )


def _sidebar_filters() -> q.Filters:
    """Global filters (period + customer state) shared by the exploratory and dashboard tabs."""
    st.sidebar.header("Filtros")
    try:
        bounds = q.filter_bounds(engine).iloc[0]
        first, last = pd.to_datetime(bounds["first_day"]).date(), pd.to_datetime(bounds["last_day"]).date()
        states = q.state_options(engine)
    except Exception:  # noqa: BLE001 - DW not built yet; tabs show their own message
        first, last, states = date(2016, 9, 1), date(2018, 10, 31), []

    picked = st.sidebar.date_input("Período da compra", value=(first, last), min_value=first, max_value=last, format="DD/MM/YYYY")
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        date_from, date_to = picked
    else:  # user is mid-selection: keep the full range until both ends are chosen
        date_from, date_to = first, last
    chosen = st.sidebar.multiselect("Estado do cliente (UF)", states, placeholder="Todos os estados")
    st.sidebar.caption(
        "Os filtros valem para **Análise Exploratória** e **Dashboard**. "
        "A seção *Qualidade dos dados* descreve a base inteira, antes de qualquer filtro."
    )
    st.sidebar.caption("Receita = itens + frete de pedidos não cancelados. Nota do pedido = avaliação mais recente.")
    return q.Filters(date_from=date_from, date_to=date_to, states=tuple(chosen))


filters = _sidebar_filters()

tab_dicionario, tab_exploratoria, tab_dashboard = st.tabs(
    ["📖 Dicionário de Dados", "🔬 Análise Exploratória", "📊 Dashboard"]
)

with tab_dicionario:
    dictionary_tab.render(engine)

with tab_exploratoria:
    exploratory_tab.render(engine, filters)

with tab_dashboard:
    insights_tab.render(engine, filters)
