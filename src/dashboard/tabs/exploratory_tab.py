"""
Aba 'Análise Exploratória' (AED).

Quatro sub-abas: 
    1. A qualidade dos dados, o que estava errado na base e como foi tratado 
    2. Distribuições das variáveis, as correlações entre elas
    3. As comparação entre grupos
    4. As análises do DW, com os dados já limpos, respeitando os filtros da barra lateral
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import Engine

from src.dashboard import queries as q
from src.dashboard import theme
from src.dashboard.tabs import quality_tab

NUMERIC_LABELS = {
    "price": "Preço do item (R$)",
    "freight_value": "Frete do item (R$)",
    "peso_g": "Peso do produto (g)",
    "volume_cm3": "Volume do produto (cm³)",
    "delivery_time_days": "Tempo de entrega (dias)",
    "approval_time_hours": "Tempo até aprovar o pagamento (horas)",
    "items_count": "Itens por pedido",
    "price_total": "Valor dos itens no pedido (R$)",
    "review_score": "Nota da avaliação",
}
ITEM_VARS = ["price", "freight_value", "peso_g", "volume_cm3"]
ORDER_VARS = ["delivery_time_days", "approval_time_hours", "items_count", "price_total", "review_score"]


def render(engine: Engine, filters: q.Filters) -> None:
    sub_quality, sub_dist, sub_corr, sub_cmp = st.tabs(
        ["🧪 Qualidade dos dados", "📈 Distribuições", "🔗 Correlações", "⚖️ Comparações"]
    )
    with sub_quality:
        quality_tab.render(engine)
    try:
        items, orders = q.item_grain(engine, filters), q.order_grain(engine, filters)
    except Exception as exc:  # noqa: BLE001
        for tab in (sub_dist, sub_corr, sub_cmp):
            tab.error(f"Não foi possível consultar o Data Warehouse: {exc}")
        return
    if items.empty or orders.empty:
        for tab in (sub_dist, sub_corr, sub_cmp):
            tab.warning("Nenhum pedido no período/estados selecionados. Ajuste os filtros na barra lateral.")
        return
    items = items.assign(price=items["price"].astype(float), freight_value=items["freight_value"].astype(float))
    orders = orders.astype({c: float for c in ["delivery_time_days", "approval_time_hours", "price_total", "freight_total", "review_score"]})

    with sub_dist:
        _render_distributions(items, orders)
    with sub_corr:
        _render_correlations(orders, items)
    with sub_cmp:
        _render_comparisons(items, orders)


# ----------------------------------------------------------------------


def _series_for(variable: str, items: pd.DataFrame, orders: pd.DataFrame) -> pd.Series:
    frame = items.rename(columns={"price": "price"}) if variable in ITEM_VARS else orders
    return frame[variable].dropna()


def _render_distributions(items: pd.DataFrame, orders: pd.DataFrame) -> None:
    st.markdown("Como cada variável se distribui? Veja o formato, os extremos e o efeito dos **outliers**.")
    col1, col2, col3 = st.columns([3, 2, 2])
    variable = col1.selectbox("Variável", ITEM_VARS + ORDER_VARS, format_func=NUMERIC_LABELS.get, key="aed_dist_var")
    trim = col2.toggle("Ocultar os 1% maiores valores", value=True, help="Corta a cauda para enxergar o miolo da distribuição (as estatísticas ao lado usam todos os dados).")
    log = col3.toggle("Escala logarítmica (x)", value=False)

    values = _series_for(variable, items, orders)
    shown = values[values <= values.quantile(0.99)] if trim else values
    discrete = variable in ("review_score", "items_count")

    fig = go.Figure(go.Histogram(
        x=shown, nbinsx=None if discrete else 50, marker=dict(color=theme.colors()[0], line=dict(color="rgba(0,0,0,0)", width=0)),
        hovertemplate="%{x}<br>%{y} registros<extra></extra>",
    ))
    fig.add_vline(x=float(values.median()), line=dict(color=theme.colors()[1], width=2, dash="dash"),
                  annotation_text=f"mediana {theme.fmt_num(values.median(), 1)}", annotation_position="top")
    fig.add_vline(x=float(values.mean()), line=dict(color=theme.colors()[2], width=2, dash="dot"),
                  annotation_text=f"média {theme.fmt_num(values.mean(), 1)}", annotation_position="top right")
    fig.update_layout(xaxis_title=NUMERIC_LABELS[variable], yaxis_title="Registros", xaxis_type="log" if log and (shown > 0).all() else "linear")
    theme.style(fig, f"Distribuição: {NUMERIC_LABELS[variable]}", height=380)

    stats = values.describe(percentiles=[0.25, 0.5, 0.75, 0.99])
    skew = float(values.skew())
    q1, q3 = values.quantile([0.25, 0.75])
    iqr_fence = q3 + 1.5 * (q3 - q1)
    outliers = int((values > iqr_fence).sum())

    left, right = st.columns([3, 1.4])
    with left:
        theme.plot(fig, key="aed_hist")
    with right:
        theme.metric(st, "Mediana", theme.fmt_num(values.median(), 2))
        theme.metric(st, "Média", theme.fmt_num(values.mean(), 2), delta=f"{(values.mean() / values.median() - 1) * 100:+.0f}% vs mediana" if values.median() else None)
        theme.metric(st, "Outliers (> Q3 + 1,5×IQR)", f"{theme.fmt_int(outliers)}", delta=f"{outliers / len(values) * 100:.1f}% dos registros".replace(".", ","))
    theme.table_view(stats.rename("valor").reset_index().rename(columns={"index": "estatística"}), "Ver estatísticas descritivas")

    if skew > 1:
        theme.insight(f"a distribuição é **assimétrica à direita** (assimetria {skew:.1f}): poucos valores muito altos puxam a média "
                      f"({theme.fmt_num(values.mean(), 1)}) acima da mediana ({theme.fmt_num(values.median(), 1)}). "
                      "Para resumir o caso típico, use a **mediana**; a média superestima. Considere escala logarítmica.")
    elif abs(skew) <= 0.5:
        theme.insight("a distribuição é aproximadamente simétrica: média e mediana contam a mesma história.")
    else:
        theme.insight(f"há assimetria moderada ({skew:.1f}); compare média e mediana antes de tirar conclusões.")

    if variable == "review_score":
        theme.insight("as notas são **bimodais**: a maioria dá 5, uma parcela relevante dá 1, e o meio é raro. Clientes tendem a avaliar "
                      "quando estão muito satisfeitos ou muito insatisfeitos, por isso a média (~4,1) esconde a polarização.", "warning")


def _render_correlations(orders: pd.DataFrame, items: pd.DataFrame) -> None:
    st.markdown("Calculadas **no grão de pedido**: cada pedido conta uma vez (a avaliação mais recente e os totais dos itens são ligados sem multiplicar linhas).")
    method = st.radio("Método", ["pearson", "spearman"], horizontal=True, format_func=lambda m: {"pearson": "Pearson (linear)", "spearman": "Spearman (ordem, resistente a outliers)"}[m])
    cols = ["price_total", "freight_total", "items_count", "delivery_time_days", "delivery_delay_days", "approval_time_hours", "review_score"]
    labels = {**NUMERIC_LABELS, "freight_total": "Frete do pedido (R$)", "delivery_delay_days": "Atraso vs. prazo (dias)"}
    corr = orders[cols].astype(float).corr(method=method)
    short = {"price_total": "Valor itens", "freight_total": "Frete", "items_count": "Itens", "delivery_time_days": "Tempo entrega",
             "delivery_delay_days": "Atraso", "approval_time_hours": "Aprovação (h)", "review_score": "Nota"}
    names = [short[c] for c in cols]

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=names, y=names, zmin=-1, zmax=1, colorscale=[[0, theme.diverging_scale()[0]], [0.5, theme.diverging_scale()[1]], [1, theme.diverging_scale()[2]]],
        text=np.round(corr.values, 2), texttemplate="%{text}", hovertemplate="%{y} × %{x}<br>correlação %{z:.2f}<extra></extra>",
        colorbar=dict(title="r", thickness=12),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=0, side="bottom")
    theme.style(fig, "Matriz de correlação", height=440)

    tri = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    strongest = tri.reindex(tri.abs().sort_values(ascending=False).index).head(3)
    left, right = st.columns([3, 2])
    with left:
        theme.plot(fig, key="aed_corr")
    with right:
        st.markdown("**Pares mais correlacionados**")
        for (a, b), value in strongest.items():
            theme.metric(st, f"{labels[a]} × {labels[b]}", f"{value:+.2f}".replace(".", ","), delta="positiva" if value > 0 else "negativa")
        review_delay = corr.loc["delivery_delay_days", "review_score"]
        theme.insight(f"a nota da avaliação se relaciona com o **atraso** (r = {theme.fmt_num(review_delay, 2)}): quanto mais o pedido passa do prazo, pior a nota. "
                      "Correlação não prova causa, mas é o principal fator operacional que o gestor consegue mexer.")
    theme.table_view(corr.round(3).reset_index().rename(columns={"index": "variável"}), "Ver a matriz em tabela")

    st.markdown("##### Olhando um par de perto")
    c1, c2 = st.columns(2)
    x = c1.selectbox("Eixo X", cols, index=cols.index("delivery_delay_days"), format_func=labels.get, key="aed_x")
    y = c2.selectbox("Eixo Y", cols, index=cols.index("review_score"), format_func=labels.get, key="aed_y")
    sample = orders[[x, y]].dropna()
    sample = sample.sample(min(6000, len(sample)), random_state=7)
    rng = np.random.default_rng(7)
    jitter = lambda col: rng.normal(0, 0.07, len(sample)) if sample[col].nunique() <= 6 else 0  # noqa: E731 - spreads discrete values so points don't stack
    scatter = go.Figure(go.Scattergl(x=sample[x] + jitter(x), y=sample[y] + jitter(y), mode="markers", marker=dict(size=5, color=theme.colors()[0], opacity=0.35),
                                     hovertemplate=f"{labels[x]}: %{{x}}<br>{labels[y]}: %{{y}}<extra></extra>", name="Pedidos"))
    if len(sample) > 2 and sample[x].nunique() > 1:
        slope, intercept = np.polyfit(sample[x], sample[y], 1)
        xs = np.array([sample[x].quantile(0.01), sample[x].quantile(0.99)])
        scatter.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", line=dict(color=theme.colors()[1], width=3), name="Tendência linear"))
    scatter.update_layout(xaxis_title=labels[x], yaxis_title=labels[y])
    theme.style(scatter, f"{labels[y]} × {labels[x]} (amostra de {theme.fmt_int(len(sample))} pedidos)", height=420, legend=True)
    theme.plot(scatter, key="aed_scatter")


def _render_comparisons(items: pd.DataFrame, orders: pd.DataFrame) -> None:
    st.markdown("Comparações entre grupos: onde os números **mudam de verdade** de um grupo para outro.")
    col1, col2 = st.columns(2)

    top_groups = items["grupo"].value_counts().head(8).index.tolist()
    fig = go.Figure()
    for group in reversed(top_groups):
        fig.add_trace(go.Box(x=items.loc[items["grupo"] == group, "price"], name=group, boxpoints=False, marker_color=theme.colors()[0],
                             line=dict(width=1.5), showlegend=False, orientation="h", hovertemplate="R$ %{x:.2f}<extra>" + group + "</extra>"))
    fig.update_layout(xaxis_title="Preço do item (R$, escala log)", xaxis_type="log")
    theme.style(fig, "Preço por macro-categoria (8 maiores em volume)", height=400)
    with col1:
        theme.plot(fig, key="aed_box_price")

    regions = [r for r in theme.REGIONS if r in orders["regiao"].dropna().unique()]
    fig2 = go.Figure()
    for region in regions:
        fig2.add_trace(go.Box(y=orders.loc[orders["regiao"] == region, "delivery_time_days"].dropna(), name=region, boxpoints=False,
                              marker_color=theme.region_colors()[region], line=dict(width=1.5), showlegend=False))
    fig2.update_layout(yaxis_title="Dias entre compra e entrega", yaxis_range=[0, float(orders["delivery_time_days"].quantile(0.99))])
    theme.style(fig2, "Tempo de entrega por região do cliente (até o percentil 99)", height=400)
    with col2:
        theme.plot(fig2, key="aed_box_delivery")

    med_price = items.groupby("grupo")["price"].median().sort_values(ascending=False)
    region_med = orders.groupby("regiao")["delivery_time_days"].median().sort_values()
    if len(region_med) > 1:
        theme.insight(f"a mediana de entrega vai de **{region_med.iloc[0]:.0f} dias** ({region_med.index[0]}) a "
                      f"**{region_med.iloc[-1]:.0f} dias** ({region_med.index[-1]}). Já o preço mediano varia de "
                      f"{theme.fmt_brl(med_price.iloc[-1])} a {theme.fmt_brl(med_price.iloc[0])} entre macro-categorias.")

    st.markdown("##### Frete pesa mais onde o produto é barato")
    ratio = items.assign(razao=(items["freight_value"] / items["price"]).clip(upper=3)).groupby("grupo").agg(
        razao_media=("razao", "median"), itens=("price", "size")).sort_values("razao_media")
    fig3 = go.Figure(go.Bar(y=ratio.index, x=ratio["razao_media"] * 100, orientation="h", marker_color=theme.colors()[0],
                            text=[theme.fmt_pct(v * 100, 0) for v in ratio["razao_media"]], textposition="outside", cliponaxis=False,
                            hovertemplate="%{y}<br>frete = %{x:.0f}% do preço (mediana)<extra></extra>"))
    fig3.update_layout(xaxis_title="Frete como % do preço do item (mediana)")
    theme.headroom(fig3, ratio["razao_media"].max() * 100)
    theme.style(fig3, "Frete ÷ preço por macro-categoria", height=440)
    theme.plot(fig3, key="aed_freight_ratio")
    theme.table_view(ratio.reset_index(), "Ver os dados deste gráfico")
    flagged = int(items["is_price_outlier"].sum())
    theme.insight(f"{theme.fmt_int(flagged)} itens ({flagged / len(items) * 100:.1f}%) têm preço extremo (acima de Q3 + 3×IQR). Foram **mantidos** — podem ser "
                  "produtos caros legítimos — mas sinalizados em `is_price_outlier` para que médias de preço não sejam puxadas por eles.")
