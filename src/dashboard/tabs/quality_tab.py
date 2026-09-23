"""
Sub-aba 'Qualidade dos dados' (dentro da Análise Exploratória).

Mostra, com números reais, o que estava errado na base bruta do Olist e o que o pipeline de limpeza
fez com cada problema: os achados vêm de `staging.dq_issues` e `staging.dq_column_profile`, gravados
pela camada staging a cada execução da DAG.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import Engine

from src.dashboard import queries as q
from src.dashboard import theme

ACTION_ORDER = ["fixed", "imputed", "nullified", "dropped", "flagged", "kept"]
ACTION_SHORT = {"fixed": "Corrigido", "imputed": "Imputado", "nullified": "Anulado", "dropped": "Removido",
                "flagged": "Sinalizado", "kept": "Mantido"}
ACTION_PT = {
    "fixed": "Corrigido",
    "imputed": "Preenchido (imputado)",
    "nullified": "Anulado (virou NULL)",
    "dropped": "Removido",
    "flagged": "Sinalizado (mantido)",
    "kept": "Mantido de propósito",
}
SEVERITY_PT = {"high": "Alta", "medium": "Média", "low": "Baixa", "info": "Informativa"}
SEVERITY_ORDER = ["high", "medium", "low", "info"]

TABLE_PT = {
    "customers": "customers (clientes)",
    "sellers": "sellers (vendedores)",
    "orders": "orders (pedidos)",
    "order_items": "order_items (itens)",
    "order_payments": "order_payments (pagamentos)",
    "order_reviews": "order_reviews (avaliações)",
    "products": "products (produtos)",
    "geolocation": "geolocation",
    "cross_table": "entre tabelas",
}

# Achados de destaque: (tabela, regra, título, por que importa)
HIGHLIGHTS = [
    ("geolocation", "exact_duplicate_rows", "Linhas 100% duplicadas na geolocation",
     "Um mesmo CEP aparece dezenas de vezes. Sem deduplicar, qualquer JOIN por CEP multiplica as linhas."),
    ("geolocation", "coordinates_outside_brazil", "Coordenadas de GPS fora do Brasil",
     "Pontos na Europa e na Argentina distorcem a média de lat/lng do CEP e derrubam mapas."),
    ("customers", "zip_lost_leading_zeros", "CEPs que perderam o zero à esquerda",
     "O CSV foi lido como número: 01046 virou 1046. Sem restaurar, CEPs de SP não cruzam com a geolocation."),
    ("sellers", "invalid_city_to_null", "Cidades inválidas (e-mail, número, só a UF)",
     "Havia e-mails e números digitados no campo cidade. Viram NULL e são reparadas pela cidade do CEP."),
    ("sellers", "city_unknown_in_state_replaced_by_zip", "Cidades trocadas pela do CEP",
     "Siglas (sbc) e erros de digitação (sao pauo, balenario camboriu) são trocados pela cidade do CEP."),
    ("order_reviews", "review_id_shared_by_several_orders", "review_id repetido em outro pedido",
     "O review_id não é único. Deduplicar só por ele descartaria avaliações válidas: a chave certa é (review_id, order_id)."),
    ("orders", "carrier_before_approval", "Postados antes da aprovação",
     "Datas fora de ordem lógica. São sinalizadas (has_date_anomaly), não apagadas."),
    ("order_items", "date_out_of_range", "Data limite de envio em 2020",
     "Pedidos de 2017/2018 com prazo de envio em 2020: data impossível, anulada."),
    ("products", "category_missing_in_translation_table", "Categorias sem tradução (Kaggle)",
     "pc_gamer e portateis_cozinha... viravam 'unknown'. Agora recebem tradução manual."),
    ("products", "missing_category", "Produtos sem categoria",
     "Ficam com o rótulo 'unknown' (sentinela) em vez de sumir dos agrupamentos."),
    ("order_payments", "zero_installments", "Cartão com 0 parcelas",
     "Todo pagamento tem ao menos 1 parcela: o zero é erro de digitação e foi ajustado para 1."),
    ("cross_table", "payment_total_differs_from_items_total", "Pedidos cujo pagamento ≠ itens + frete",
     "Costuma ser juros de parcelamento ou desconto/voucher. Importante ao comparar receita por pagamento e por item."),
]


def render(engine: Engine) -> None:
    try:
        issues = q.dq_issues(engine)
        profile = q.dq_profile(engine)
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Não encontrei as tabelas de auditoria `staging.dq_issues` / `staging.dq_column_profile`. "
            f"Rode a etapa de staging da DAG `olist_etl_pipeline` primeiro.\n\nDetalhe: {exc}"
        )
        return
    if issues.empty:
        st.warning("A tabela de auditoria está vazia. Rode a DAG `olist_etl_pipeline`.")
        return

    st.markdown(
        "Antes de virar análise, os dados brutos do Olist passam por uma limpeza. Esta seção mostra **o que estava "
        "errado, quantas linhas foram afetadas e o que foi feito**. Nenhum registro suspeito some em silêncio: "
        "ou é corrigido, ou é sinalizado por uma coluna de qualidade, e tudo fica registrado."
    )

    relevant = issues[issues["severity"] != "info"]
    with_findings = relevant[relevant["rows_affected"] > 0]
    corrected = with_findings[with_findings["action"].isin(["fixed", "imputed", "nullified"])]["rows_affected"].sum()
    flagged = with_findings[with_findings["action"] == "flagged"]["rows_affected"].sum()
    dropped = with_findings[with_findings["action"] == "dropped"]["rows_affected"].sum()

    cols = st.columns(5)
    cols[0].metric("Regras avaliadas", theme.fmt_int(len(issues)))
    cols[1].metric("Regras com achados", theme.fmt_int(len(with_findings)), help="Regras que encontraram ao menos 1 linha problemática (exceto as informativas).")
    cols[2].metric("Ocorrências corrigidas", theme.fmt_int(corrected), help="Soma de linhas corrigidas, preenchidas ou anuladas. Uma linha com dois problemas conta duas vezes.")
    cols[3].metric("Ocorrências sinalizadas", theme.fmt_int(flagged), help="Mantidas na base com uma flag de qualidade.")
    cols[4].metric("Linhas removidas", theme.fmt_int(dropped), help="Duplicadas ou descartadas (ex.: pontos de GPS fora do Brasil).")
    st.caption(f"Última verificação: {pd.to_datetime(issues['checked_at']).max():%d/%m/%Y %H:%M} (UTC)")

    _render_highlights(issues)
    st.divider()
    _render_overview(with_findings)
    st.divider()
    _render_table_of_findings(issues)
    st.divider()
    _render_before_after(profile)
    st.divider()
    _render_dates(engine)
    st.divider()
    _render_reconciliation(engine, profile)


def _rows(issues: pd.DataFrame, table: str, rule: str):
    match = issues[(issues["table_name"] == table) & (issues["rule"] == rule)]
    return match.iloc[0] if not match.empty else None


def _render_highlights(issues: pd.DataFrame) -> None:
    st.subheader("🔎 Os principais problemas encontrados")
    cards = [(h, _rows(issues, h[0], h[1])) for h in HIGHLIGHTS]
    cards = [(h, r) for h, r in cards if r is not None and r["rows_affected"] > 0]
    for start in range(0, len(cards), 3):
        for col, (highlight, row) in zip(st.columns(3), cards[start:start + 3]):
            with col.container(border=True):
                theme.metric(st, highlight[2], theme.fmt_int(row["rows_affected"]),
                             delta=f"{theme.fmt_pct(row['pct_affected'], 2)} de {TABLE_PT.get(row['table_name'], row['table_name'])}")
                st.caption(highlight[3])
                st.caption(f"**Ação:** {ACTION_PT[row['action']]}" + (f" · ex.: `{row['examples']}`" if row["examples"] else ""))


def _render_overview(with_findings: pd.DataFrame) -> None:
    st.subheader("📊 Onde estão os problemas")
    col1, col2 = st.columns(2)

    by_action = with_findings.groupby(["table_name", "action"]).size().reset_index(name="regras")
    fig = go.Figure()
    order = with_findings.groupby("table_name").size().sort_values().index.tolist()
    for action in ACTION_ORDER:
        part = by_action[by_action["action"] == action].set_index("table_name").reindex(order)
        if part["regras"].fillna(0).sum() == 0:
            continue
        fig.add_bar(y=[TABLE_PT.get(t, t) for t in order], x=part["regras"].fillna(0), name=ACTION_SHORT[action], orientation="h",
                    marker=dict(color="#898781" if action == "kept" else theme.color_for(action, ACTION_ORDER), line=dict(color="rgba(0,0,0,0)", width=0)),
                    hovertemplate="%{y}<br>" + ACTION_PT[action] + ": %{x} regra(s)<extra></extra>")
    fig.update_layout(barmode="stack", xaxis_title="Regras com achados")
    theme.style(fig, "Regras com achados por tabela e ação", height=430, legend=True, legend_rows=2)
    with col1:
        theme.plot(fig, key="dq_by_table")

    top = with_findings.sort_values("pct_affected", ascending=False).head(12).iloc[::-1]
    labels = [f"{TABLE_PT.get(t, t).split(' ')[0]} · {r.replace('_', ' ')}" for t, r in zip(top["table_name"], top["rule"])]
    fig2 = go.Figure(go.Bar(
        y=labels, x=top["pct_affected"], orientation="h", marker_color=theme.colors()[0],
        text=[theme.fmt_pct(v, 1) for v in top["pct_affected"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.2f}% das linhas<extra></extra>",
    ))
    fig2.update_layout(xaxis_title="% das linhas da tabela")
    theme.headroom(fig2, top["pct_affected"].max())
    theme.style(fig2, "Maiores impactos (% das linhas da tabela)", height=430)
    with col2:
        theme.plot(fig2, key="dq_top_pct")

    theme.table_view(with_findings[["table_name", "rule", "action", "rows_affected", "pct_affected"]]
                     .sort_values("pct_affected", ascending=False), "Ver os dados destes gráficos")


def _render_table_of_findings(issues: pd.DataFrame) -> None:
    st.subheader("🗂️ Todas as regras de qualidade")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1.5])
    tables = c1.multiselect("Tabela", sorted(issues["table_name"].unique()), format_func=lambda t: TABLE_PT.get(t, t))
    severities = c2.multiselect("Severidade", SEVERITY_ORDER, format_func=SEVERITY_PT.get)
    actions = c3.multiselect("Ação", ACTION_ORDER, format_func=ACTION_PT.get)
    only_findings = c4.toggle("Só com achados", value=True)

    view = issues.copy()
    if tables:
        view = view[view["table_name"].isin(tables)]
    if severities:
        view = view[view["severity"].isin(severities)]
    if actions:
        view = view[view["action"].isin(actions)]
    if only_findings:
        view = view[view["rows_affected"] > 0]
    view = view.assign(_sev=view["severity"].map({s: i for i, s in enumerate(SEVERITY_ORDER)}))
    view = view.sort_values(["_sev", "pct_affected"], ascending=[True, False])

    shown = pd.DataFrame({
        "Tabela": view["table_name"],
        "Coluna": view["column_name"],
        "Regra": view["rule"],
        "Severidade": view["severity"].map(SEVERITY_PT),
        "Ação": view["action"].map(ACTION_PT),
        "Linhas": view["rows_affected"],
        "% da tabela": view["pct_affected"],
        "O que foi feito": view["description"],
        "Exemplos (antes → depois)": view["examples"],
    })
    st.dataframe(
        shown, width="stretch", hide_index=True, height=420,
        column_config={
            "Linhas": st.column_config.NumberColumn(format="localized"),
            "% da tabela": st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=100),
            "O que foi feito": st.column_config.TextColumn(width="large"),
            "Exemplos (antes → depois)": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(f"{len(view)} regra(s) exibida(s). Severidade *alta* = dado impossível ou que quebraria análises; *informativa* = comportamento esperado, documentado.")


def _render_before_after(profile: pd.DataFrame) -> None:
    st.subheader("🔁 Antes × depois da limpeza")
    tables = sorted(profile["table_name"].unique())
    table = st.selectbox("Tabela", tables, format_func=lambda t: TABLE_PT.get(t, t), key="dq_before_after_table")
    part = profile[profile["table_name"] == table]
    raw = part[part["stage"] == "raw"].set_index("column_name")
    stg = part[part["stage"] == "staging"].set_index("column_name")

    columns = pd.DataFrame({
        "coluna": stg.index,
        "nulos_raw_pct": raw["null_pct"].reindex(stg.index).values,
        "nulos_staging_pct": stg["null_pct"].values,
        "tipo_raw": raw["dtype"].reindex(stg.index).fillna("— (coluna nova)").values,
        "tipo_staging": stg["dtype"].values,
        "distintos_raw": raw["distinct_count"].reindex(stg.index).values,
        "distintos_staging": stg["distinct_count"].values,
    })

    col1, col2 = st.columns([3, 2])
    with col1:
        nulls = columns[(columns["nulos_raw_pct"].fillna(0) > 0) | (columns["nulos_staging_pct"] > 0)]
        if nulls.empty:
            st.success("Nenhuma coluna desta tabela tem valores nulos, antes ou depois.")
        else:
            fig = go.Figure()
            fig.add_bar(y=nulls["coluna"], x=nulls["nulos_raw_pct"], orientation="h", name="Raw (antes)", marker_color=theme.colors()[0],
                        hovertemplate="%{y}: %{x:.2f}% nulos<extra>Raw</extra>")
            fig.add_bar(y=nulls["coluna"], x=nulls["nulos_staging_pct"], orientation="h", name="Staging (depois)", marker_color=theme.colors()[1],
                        hovertemplate="%{y}: %{x:.2f}% nulos<extra>Staging</extra>")
            fig.update_layout(barmode="group", xaxis_title="% de valores nulos", yaxis=dict(autorange="reversed"))
            theme.headroom(fig, max(nulls["nulos_raw_pct"].max(), nulls["nulos_staging_pct"].max()))
            theme.style(fig, "Valores nulos por coluna", height=max(300, 46 * len(nulls) + 110), legend=True)
            theme.plot(fig, key=f"dq_nulls_{table}")
            st.caption("Nulos que **caem** foram preenchidos (imputação/reparo pelo CEP). Nulos que **sobem** são valores inválidos anulados de propósito (NULL honesto em vez de dado errado).")
    with col2:
        changed = columns[columns["tipo_raw"] != columns["tipo_staging"]][["coluna", "tipo_raw", "tipo_staging"]]
        st.markdown("**Colunas cujo tipo mudou (tipagem)**")
        st.dataframe(changed.rename(columns={"coluna": "Coluna", "tipo_raw": "Raw", "tipo_staging": "Staging"}), width="stretch", hide_index=True, height=300)
        st.caption("`object`/`int64` → `string`, `datetime64`, `boolean`: o raw guarda tudo como veio do CSV; a staging aplica o tipo correto.")
    theme.table_view(columns, "Ver o perfil completo das colunas")


def _render_dates(engine: Engine) -> None:
    st.subheader("🕐 Datas fora de ordem lógica")
    try:
        anomalies = q.date_anomalies(engine)
    except Exception:  # noqa: BLE001
        st.info("Tabela staging.stg_orders indisponível.")
        return
    anomalies = anomalies[anomalies["pedidos"] > 0].sort_values("pedidos")
    if anomalies.empty:
        st.success("Nenhuma inconsistência de datas encontrada.")
        return
    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure(go.Bar(
            y=anomalies["anomalia"], x=anomalies["pedidos"], orientation="h", marker_color=theme.colors()[1],
            text=[theme.fmt_int(v) for v in anomalies["pedidos"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>%{x} pedidos<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Pedidos")
        theme.headroom(fig, anomalies["pedidos"].max())
        theme.style(fig, "Pedidos com linha do tempo inconsistente", height=300)
        theme.plot(fig, key="dq_dates")
    with col2:
        total = int(anomalies["pedidos"].sum())
        theme.metric(st, "Ocorrências de datas inconsistentes", theme.fmt_int(total))
        theme.insight(
            "o problema mais comum é a **postagem registrada antes da aprovação do pagamento**: o vendedor "
            "marca o envio antes de o sistema confirmar o pagamento. Não configura erro de cálculo, sim comportamento. "
            "Assim os pedidos ficam na base com `has_date_anomaly = TRUE` e as métricas de entrega só usam "
            "pedidos entregues com a linha do tempo coerente.", "warning")


def _render_reconciliation(engine: Engine, profile: pd.DataFrame) -> None:
    st.subheader("⚖️ Conferência de linhas: raw → staging → data warehouse")
    counts = q.dw_row_counts(engine)
    sizes = profile.groupby(["table_name", "stage"])["total_rows"].max().unstack()
    table = counts.set_index("t").join(sizes[["raw", "staging"]]).reset_index()
    table["diferença"] = table["dw"] - table["raw"]
    table = table.rename(columns={"t": "Tabela", "raw": "Linhas raw", "staging": "Linhas staging", "dw": "Linhas DW"})
    st.dataframe(table[["Tabela", "Linhas raw", "Linhas staging", "Linhas DW", "diferença"]], width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="localized") for c in ["Linhas raw", "Linhas staging", "Linhas DW", "diferença"]})
    lost = table[(table["Tabela"] != "geolocation") & (table["diferença"] != 0)]
    if lost.empty:
        st.success("Nenhuma linha foi perdida, geolocation encolhe pois milhões de pontos viram uma linha por CEP.")
    else:
        st.warning(f"Tabelas com diferença de linhas: {', '.join(lost['Tabela'])}.")
