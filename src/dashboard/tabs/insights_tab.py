"""
Aba 'Dashboard': análises para gestores e diretoria.

Cada seção traz os números-chave, um gráfico e uma **leitura em linguagem de negócio** calculada a partir
dos próprios dados (não um texto fixo). Todas respeitam os filtros de período e estado da barra lateral.

Definições: pedido válido = não cancelado/indisponível; receita = itens + frete dos pedidos válidos;
atraso = entrega depois da data estimada; nota do pedido = sua avaliação mais recente.
"""

from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import Engine

from src.dashboard import queries as q
from src.dashboard import theme
from src.etl.cleaning import tokenize_pt

STATUS_PT = {
    "delivered": "Entregue", "shipped": "Enviado", "canceled": "Cancelado", "unavailable": "Indisponível",
    "invoiced": "Faturado", "processing": "Processando", "created": "Criado", "approved": "Aprovado",
}
PAYMENT_PT = {"credit_card": "Cartão de crédito", "boleto": "Boleto", "voucher": "Voucher", "debit_card": "Cartão de débito"}


def render(engine: Engine, filters: q.Filters) -> None:
    try:
        kpi = q.kpis(engine, filters)
        if not kpi["pedidos"]:
            st.warning("Nenhum pedido no período/estados selecionados. Ajuste os filtros na barra lateral.")
            return
        _headline(engine, filters, kpi)
        for section in (_revenue, _seasonality, _categories, _payments, _delivery, _reviews, _geography, _sellers, _customers):
            st.divider()
            section(engine, filters)
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Não foi possível consultar o Data Warehouse. Verifique se a DAG `olist_etl_pipeline` já foi executada.\n\n"
            f"Detalhe: {exc}"
        )


def _hbar(df: pd.DataFrame, x: str, y: str, title: str, key: str, text=None, x_title: str = "", height: int = 400, color=None) -> None:
    """Single-series horizontal bar, largest on top, value labels at the bar end."""
    ordered = df.sort_values(x)
    fig = go.Figure(go.Bar(
        y=ordered[y], x=ordered[x], orientation="h", marker_color=color or theme.colors()[0],
        text=ordered[text] if text else None, textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:,.2f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title=x_title, yaxis_title=None)
    theme.headroom(fig, ordered[x].max() if len(ordered) else 1)
    theme.style(fig, title, height=height)
    theme.plot(fig, key=key)


# ----------------------------------------------------------------------
# Headline
# ----------------------------------------------------------------------


def _headline(engine: Engine, f: q.Filters, kpi: pd.Series) -> None:
    rk = q.review_kpis(engine, f)
    ticket = kpi["receita"] / kpi["pedidos_validos"] if kpi["pedidos_validos"] else np.nan
    row1 = st.columns(4)
    theme.metric(row1[0], "Receita", theme.fmt_brl_short(kpi["receita"]), help="Preço + frete dos itens de pedidos válidos (sem cancelados/indisponíveis).")
    theme.metric(row1[1], "Pedidos", theme.fmt_int(kpi["pedidos_validos"]), delta=f"{theme.fmt_pct(kpi['cancelados'] / kpi['pedidos'] * 100)} cancelados")
    theme.metric(row1[2], "Ticket médio", theme.fmt_brl(ticket, 2), help="Receita ÷ pedidos válidos.")
    theme.metric(row1[3], "Clientes únicos", theme.fmt_int(kpi["clientes"]))
    row2 = st.columns(4)
    theme.metric(row2[0], "Nota média", f"{theme.fmt_num(rk['nota_media'], 2)} / 5", delta=f"{theme.fmt_pct(rk['pct_positivas'], 0)} com nota 4-5")
    row2[1].metric("Avaliações negativas", theme.fmt_pct(rk["pct_negativas"]), help="Notas 1 ou 2.")
    row2[2].metric("Entrega média", f"{theme.fmt_num(kpi['entrega_media'], 1)} dias", help="Da compra à entrega, só pedidos entregues.")
    row2[3].metric("Entregas com atraso", theme.fmt_pct(kpi["pct_atraso"]), help="Entregues depois da data estimada.")
    st.caption(f"Frete representa {theme.fmt_pct(kpi['frete'] / kpi['receita'] * 100 if kpi['receita'] else np.nan)} da receita · "
               f"{theme.fmt_num(kpi['itens_por_pedido'], 2)} itens por pedido em média.")


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------


def _revenue(engine: Engine, f: q.Filters) -> None:
    st.subheader("💰 Receita ao longo do tempo")
    df = q.monthly(engine, f)
    if df.empty:
        st.info("Sem dados de receita no filtro atual.")
        return
    df = df.assign(mes=pd.to_datetime(df["mes"]))
    df["ticket"] = df["receita"] / df["pedidos"]
    df["label"] = df["mes"].map(lambda d: f"{theme.MONTH_ABBR_PT[d.month - 1]}/{str(d.year)[2:]}")
    incomplete = df["pedidos"] < 0.2 * df["pedidos"].median()

    muted = "#898781"
    fig = go.Figure(go.Bar(
        x=df["label"], y=df["receita"], marker_color=[muted if inc else theme.colors()[0] for inc in incomplete],
        customdata=np.stack([df["pedidos"], incomplete.map({True: " (mês incompleto)", False: ""})], axis=-1),
        hovertemplate="%{x}%{customdata[1]}<br>Receita R$ %{y:,.0f}<br>%{customdata[0]} pedidos<extra></extra>",
    ))
    fig.update_layout(yaxis_title="Receita (R$)")
    fig.update_xaxes(tickangle=45, tickmode="array", tickvals=df["label"].iloc[::2])
    theme.style(fig, "Receita mensal (cinza = mês incompleto)", height=380)
    ok_months = df[~incomplete]
    fig2 = go.Figure(go.Scatter(x=ok_months["label"], y=ok_months["ticket"], mode="lines+markers", line=dict(color=theme.colors()[1], width=2), marker=dict(size=7),
                                hovertemplate="%{x}<br>Ticket médio R$ %{y:,.2f}<extra></extra>"))
    fig2.update_layout(yaxis_title="R$ por pedido", yaxis_rangemode="tozero")
    fig2.update_xaxes(tickangle=45, tickmode="array", tickvals=ok_months["label"].iloc[::2])
    theme.style(fig2, "Ticket médio mensal (meses completos)", height=380)

    col1, col2 = st.columns([3, 2])
    with col1:
        theme.plot(fig, key="rev_month")
    with col2:
        theme.plot(fig2, key="rev_ticket")

    complete = df[~incomplete]
    best = complete.loc[complete["receita"].idxmax()] if not complete.empty else df.loc[df["receita"].idxmax()]
    text = f"o melhor mês foi **{best['label']}** ({theme.fmt_brl_short(best['receita'])}, {theme.fmt_int(best['pedidos'])} pedidos)."
    d = complete.assign(ano=complete["mes"].dt.year, m=complete["mes"].dt.month)
    common = sorted(set(d.loc[d["ano"] == 2017, "m"]) & set(d.loc[d["ano"] == 2018, "m"]))
    if common:
        r17 = d[(d["ano"] == 2017) & d["m"].isin(common)]["receita"].sum()
        r18 = d[(d["ano"] == 2018) & d["m"].isin(common)]["receita"].sum()
        text += f" Nos mesmos meses, 2018 rendeu **{(r18 / r17 - 1) * 100:+.0f}%** vs. 2017."
    growth_orders = complete["pedidos"].iloc[-1] / complete["pedidos"].iloc[0] if len(complete) > 1 and complete["pedidos"].iloc[0] else None
    if growth_orders and growth_orders > 2:
        text += f" O volume mensal de pedidos é {growth_orders:.0f}x o do primeiro mês completo: marketplace em forte expansão."
    theme.insight(text)
    theme.table_view(df[["label", "receita", "pedidos", "clientes", "ticket"]])


def _seasonality(engine: Engine, f: q.Filters) -> None:
    st.subheader("📅 Quando os clientes compram")
    df = q.weekday_hour(engine, f)
    if df.empty:
        st.info("Sem dados de sazonalidade no filtro atual.")
        return
    grid = df.pivot_table(index="day_of_week", columns="hora", values="pedidos", fill_value=0).reindex(index=range(7), columns=range(24), fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[f"{h}h" for h in grid.columns], y=theme.DAY_ABBR_PT, colorscale=theme.sequential_scale(),
        hovertemplate="%{y} às %{x}<br>%{z} pedidos<extra></extra>", colorbar=dict(title="Pedidos", thickness=12),
    ))
    fig.update_yaxes(autorange="reversed")
    theme.style(fig, "Pedidos por dia da semana e hora da compra", height=340)
    by_day = df.groupby("day_of_week")["pedidos"].sum().reindex(range(7), fill_value=0)
    by_hour = df.groupby("hora")["pedidos"].sum()
    peak_day, peak_hour = int(by_day.idxmax()), int(by_hour.idxmax())
    weekend_share = by_day.iloc[5:].sum() / by_day.sum() * 100
    evening = by_hour[(by_hour.index >= 10) & (by_hour.index <= 22)].sum() / by_hour.sum() * 100

    col1, col2 = st.columns([3, 1.4])
    with col1:
        theme.plot(fig, key="season_heat")
    with col2:
        theme.metric(st, "Dia mais forte", theme.DAY_FULL_PT[peak_day])
        theme.metric(st, "Hora mais forte", f"{peak_hour}h")
        theme.metric(st, "Compras no fim de semana", theme.fmt_pct(weekend_share, 0))
    theme.insight(f"as compras se concentram nos **dias úteis**, com pico de **{theme.DAY_FULL_PT[peak_day].lower()}** por volta das **{peak_hour}h**; {evening:.0f}% dos pedidos "
                  f"acontecem entre 10h e 22h e o fim de semana pesa só {weekend_share:.0f}%. Campanhas e disparos de e-mail tendem a render mais nesse horário; "
                  "manutenções e trocas de estoque cabem na madrugada.")
    theme.table_view(by_day.rename("pedidos").rename(index=dict(enumerate(theme.DAY_ABBR_PT))).reset_index().rename(columns={"day_of_week": "dia"}))


def _categories(engine: Engine, f: q.Filters) -> None:
    st.subheader("🏷️ Categorias")
    df = q.categories(engine, f)
    if df.empty:
        st.info("Sem dados de categorias no filtro atual.")
        return
    groups = df.groupby("grupo").agg(receita=("receita", "sum"), itens=("itens", "sum"), frete=("frete", "sum")).reset_index()
    groups["share"] = groups["receita"] / groups["receita"].sum() * 100
    groups["frete_pct"] = groups["frete"] / groups["receita"] * 100
    groups["label"] = groups["share"].map(lambda v: theme.fmt_pct(v, 1))

    cats = df.sort_values("receita", ascending=False).reset_index(drop=True)
    cats["categoria"] = cats["categoria_pt"].str.replace("_", " ")
    cats["acumulado"] = cats["receita"].cumsum() / cats["receita"].sum() * 100
    n80 = int((cats["acumulado"] < 80).sum() + 1)

    col1, col2 = st.columns(2)
    with col1:
        _hbar(groups, "receita", "grupo", "Receita por macro-categoria", "cat_groups", text="label", x_title="Receita dos itens (R$)", height=430)
    with col2:
        fig = go.Figure(go.Scatter(x=np.arange(1, len(cats) + 1), y=cats["acumulado"], mode="lines", line=dict(color=theme.colors()[0], width=3),
                                   customdata=cats["categoria"], hovertemplate="%{customdata}<br>%{x}ª categoria: %{y:.1f}% da receita acumulada<extra></extra>"))
        fig.add_hline(y=80, line=dict(color=theme.colors()[1], width=1.5, dash="dash"), annotation_text="80% da receita")
        fig.add_vline(x=n80, line=dict(color=theme.colors()[1], width=1.5, dash="dash"), annotation_text=f"{n80} categorias", annotation_position="bottom right")
        fig.update_layout(xaxis_title="Categorias, da maior para a menor", yaxis_title="% da receita acumulada", yaxis_range=[0, 102])
        theme.style(fig, "Curva de Pareto: concentração da receita", height=430)
        theme.plot(fig, key="cat_pareto")

    top = groups.sort_values("receita", ascending=False).iloc[0]
    priciest = cats[cats["itens"] >= 100].assign(ticket=lambda d: d["receita"] / d["itens"]).sort_values("ticket", ascending=False).iloc[0]
    freight_heavy = groups[groups["itens"] >= 200].sort_values("frete_pct", ascending=False).iloc[0]
    theme.insight(f"**{top['grupo']}** lidera com {top['share']:.0f}% da receita, e apenas **{n80} das {len(cats)} categorias** somam 80% do faturamento: "
                  f"o resto é cauda longa. A categoria de maior ticket é *{priciest['categoria']}* ({theme.fmt_brl(priciest['ticket'])} por item). "
                  f"O frete pesa mais em **{freight_heavy['grupo']}** ({freight_heavy['frete_pct']:.0f}% do valor dos itens), onde faz sentido negociar frete ou subsídio.")
    view = cats.assign(ticket=cats["receita"] / cats["itens"])[["grupo", "categoria", "receita", "itens", "ticket", "acumulado"]]
    theme.table_view(view.round(2), "Ver todas as categorias")


def _payments(engine: Engine, f: q.Filters) -> None:
    st.subheader("💳 Pagamentos")
    df = q.payments(engine, f)
    if df.empty:
        st.info("Sem dados de pagamento no filtro atual.")
        return
    df = df.assign(valor=df["valor"].astype(float))
    by_type = df.groupby("tipo").agg(pagamentos=("valor", "size"), valor=("valor", "sum"), ticket=("valor", "mean")).reset_index()
    by_type["share"] = by_type["pagamentos"] / by_type["pagamentos"].sum() * 100
    by_type["nome"] = by_type["tipo"].map(PAYMENT_PT).fillna(by_type["tipo"])
    by_type["label"] = by_type["share"].map(lambda v: theme.fmt_pct(v, 1))
    card = df[df["tipo"] == "credit_card"]
    installments = card["parcelas"].clip(upper=12).astype(int).value_counts().sort_index()
    inst_labels = [f"{i}x" if i < 12 else "12x+" for i in installments.index]

    col1, col2 = st.columns(2)
    with col1:
        _hbar(by_type, "share", "nome", "Forma de pagamento (% dos pagamentos)", "pay_type", text="label", x_title="% dos pagamentos", height=320)
    with col2:
        fig = go.Figure(go.Bar(x=inst_labels, y=installments.values, marker_color=theme.colors()[0],
                               hovertemplate="%{x}<br>%{y} pagamentos<extra></extra>"))
        fig.update_layout(xaxis_title="Parcelas no cartão de crédito", yaxis_title="Pagamentos")
        theme.style(fig, "Parcelamento no cartão", height=320)
        theme.plot(fig, key="pay_inst")

    cc_share = by_type.loc[by_type["tipo"] == "credit_card", "share"].sum()
    boleto_share = by_type.loc[by_type["tipo"] == "boleto", "share"].sum()
    avg_inst = card["parcelas"].mean() if not card.empty else np.nan
    one_x = card.loc[card["parcelas"] == 1, "valor"].mean()
    multi = card.loc[card["parcelas"] >= 6, "valor"].mean()
    text = (f"o **cartão de crédito** responde por {cc_share:.0f}% dos pagamentos e o **boleto** por {boleto_share:.0f}%. "
            f"Parcelamento médio de {theme.fmt_num(avg_inst, 1)}x.")
    if pd.notna(one_x) and pd.notna(multi):
        text += (f" Quem parcela em 6x ou mais gasta em média {theme.fmt_brl(multi)} contra {theme.fmt_brl(one_x)} à vista: "
                 "o parcelamento é o que viabiliza tickets altos, então limitar parcelas reduz vendas de maior valor.")
    theme.insight(text)
    theme.table_view(by_type[["nome", "pagamentos", "share", "valor", "ticket"]].round(2))


def _delivery(engine: Engine, f: q.Filters) -> None:
    st.subheader("🚚 Entregas e satisfação")
    states = q.delivery_by_state(engine, f)
    hist = q.delivery_time_hist(engine, f)
    bucket = q.delay_vs_review(engine, f)
    status = q.status_counts(engine, f)
    if states.empty or hist.empty:
        st.info("Sem entregas no filtro atual.")
        return
    states = states.astype({"entrega_media": float, "prazo_prometido": float, "pct_atraso": float, "frete_pct_preco": float})

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(x=hist["dias"], y=hist["pedidos"], marker_color=theme.colors()[0], hovertemplate="%{x} dias<br>%{y} pedidos<extra></extra>"))
        fig.update_layout(xaxis_title="Dias entre compra e entrega (60 = 60 ou mais)", yaxis_title="Pedidos", bargap=0.05)
        theme.style(fig, "Distribuição do tempo de entrega", height=360)
        theme.plot(fig, key="del_hist")
    with col2:
        fig = go.Figure(go.Bar(
            x=[b.split(". ")[1] for b in bucket["faixa"]], y=bucket["nota_media"].astype(float), marker_color=theme.colors()[0],
            text=[theme.fmt_num(v, 2) for v in bucket["nota_media"].astype(float)], textposition="outside", cliponaxis=False,
            customdata=bucket["pedidos"], hovertemplate="%{x}<br>nota média %{y:.2f}<br>%{customdata} pedidos<extra></extra>",
        ))
        fig.update_layout(yaxis_title="Nota média (1-5)", yaxis_range=[0, 5.4])
        theme.style(fig, "O atraso derruba a nota da avaliação", height=360)
        theme.plot(fig, key="del_bucket")

    late = states[states["pedidos_entregues"] >= 50].sort_values("pct_atraso", ascending=False)
    col3, col4 = st.columns(2)
    with col3:
        top = late.head(12).sort_values("pct_atraso")
        fig = go.Figure()
        for region in theme.REGIONS:
            part = top[top["regiao"] == region]
            if part.empty:
                continue
            fig.add_bar(y=part["estado"], x=part["pct_atraso"], orientation="h", name=region, marker_color=theme.region_colors()[region],
                        text=[theme.fmt_pct(v, 1) for v in part["pct_atraso"]], textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: %{x:.1f}% de atraso<extra>" + region + "</extra>")
        fig.update_layout(barmode="stack", xaxis_title="% de pedidos entregues com atraso")
        fig.update_yaxes(categoryorder="array", categoryarray=top["estado"].tolist())
        theme.headroom(fig, top["pct_atraso"].max())
        theme.style(fig, "Estados com mais atraso (12 piores)", height=430, legend=True)
        theme.plot(fig, key="del_states")
    with col4:
        s = status.assign(nome=status["status"].map(STATUS_PT).fillna(status["status"]))
        s["pct"] = s["pedidos"] / s["pedidos"].sum() * 100
        s["label"] = s["pct"].map(lambda v: theme.fmt_pct(v, 1))
        _hbar(s, "pedidos", "nome", "Pedidos por status", "del_status", text="label", x_title="Pedidos", height=400)

    on_time = bucket.loc[bucket["faixa"].str.startswith("1."), "nota_media"].astype(float)
    severe = bucket.loc[bucket["faixa"].str.startswith("4."), "nota_media"].astype(float)
    pct_late = states["pct_atraso"].mul(states["pedidos_entregues"]).sum() / states["pedidos_entregues"].sum()
    margin = (states["prazo_prometido"] * states["pedidos_entregues"]).sum() / states["pedidos_entregues"].sum() - \
             (states["entrega_media"] * states["pedidos_entregues"]).sum() / states["pedidos_entregues"].sum()
    text = (f"apenas **{pct_late:.1f}%** dos pedidos chegam atrasados, e o prazo prometido tem em média **{margin:.0f} dias de folga** sobre a entrega real. "
            "O problema é a cauda: ")
    if len(on_time) and len(severe):
        text += f"pedidos no prazo recebem nota **{on_time.iloc[0]:.1f}**, mas com 8+ dias de atraso a nota cai para **{severe.iloc[0]:.1f}**. "
    if not late.empty:
        w = late.iloc[0]
        text += (f"O pior estado é **{w['estado']}** ({w['pct_atraso']:.0f}% de atraso, entrega média de {w['entrega_media']:.0f} dias). "
                 "Atacar a cauda de atrasos graves (revisar transportadoras nas regiões críticas) melhora a satisfação mais do que encurtar o prazo médio.")
    theme.insight(text, "warning")
    theme.table_view(states.sort_values("pct_atraso", ascending=False).round(2), "Ver entrega por estado")


def _reviews(engine: Engine, f: q.Filters) -> None:
    st.subheader("⭐ Avaliações dos clientes")
    scores = q.review_scores(engine, f)
    by_group = q.review_by_group(engine, f)
    if scores.empty:
        st.info("Sem avaliações no filtro atual.")
        return
    scores = scores.assign(nota=scores["nota"].astype(int))
    total = scores["avaliacoes"].sum()
    scores["pct"] = scores["avaliacoes"] / total * 100
    promoters = scores.loc[scores["nota"] == 5, "pct"].sum()
    detractors = scores.loc[scores["nota"] <= 3, "pct"].sum()

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(x=scores["nota"], y=scores["avaliacoes"], marker_color=theme.colors()[0],
                               text=[theme.fmt_pct(v, 0) for v in scores["pct"]], textposition="outside", cliponaxis=False,
                               hovertemplate="Nota %{x}<br>%{y} avaliações<extra></extra>"))
        fig.update_layout(xaxis=dict(title="Nota", dtick=1), yaxis_title="Avaliações")
        theme.style(fig, "Distribuição das notas", height=360)
        theme.plot(fig, key="rev_scores")
    with col2:
        g = by_group[by_group["pedidos"] >= 200].copy()
        g["label"] = g["pct_negativas"].astype(float).map(lambda v: theme.fmt_pct(v, 1))
        _hbar(g.assign(pct_negativas=g["pct_negativas"].astype(float)), "pct_negativas", "grupo", "% de avaliações negativas (1-2) por macro-categoria", "rev_group",
              text="label", x_title="% de notas 1 ou 2", height=360)

    st.markdown("##### O que os clientes dizem")
    words = _word_shares(engine, f)
    logistics_neg = logistics_pos = None
    if words:
        (n_neg, neg, log_neg), (n_pos, pos, log_pos) = words["neg"], words["pos"]
        logistics_neg, logistics_pos = log_neg / n_neg * 100, log_pos / n_pos * 100
        c1, c2 = st.columns(2)
        for col, title, n, items, color, key in ((c1, "Palavras mais citadas nas avaliações NEGATIVAS (1-2)", n_neg, neg, theme.colors()[7], "w_neg"),
                                                (c2, "Palavras mais citadas nas avaliações POSITIVAS (4-5)", n_pos, pos, theme.colors()[2], "w_pos")):
            with col:
                d = pd.DataFrame(items, columns=["palavra", "n"]).assign(pct=lambda x: x["n"] / n * 100)
                d["label"] = d["pct"].map(lambda v: theme.fmt_pct(v, 1))
                _hbar(d, "pct", "palavra", title, key, text="label", x_title=f"% dos {theme.fmt_int(n)} comentários", height=420, color=color)
        neg_only = [w for w, _ in neg if w not in {p for p, _ in pos}][:5]
        text = ("os comentários passaram por limpeza de texto (quebras de linha, espaços, codificação e comentários vazios como *ok* ou *.* foram tratados) "
                f"antes desta contagem. Termos que aparecem **só** nas reclamações: {', '.join(f'*{w}*' for w in neg_only) or '—'}. "
                f"**{logistics_neg:.0f}%** das reclamações falam de entrega ou recebimento (prazo, atraso, *não recebi*), contra {logistics_pos:.0f}% dos elogios")
        if logistics_neg - logistics_pos >= 15:
            text += ": a insatisfação é sobretudo **logística**, e não do produto."
        else:
            text += ": a entrega é o assunto dominante dos dois lados; quando falha vira reclamação, quando funciona vira elogio."
        theme.insight(text)
    tail = "converter os insatisfeitos"
    if logistics_neg is not None and logistics_pos is not None and logistics_neg - logistics_pos >= 15:
        tail += ", cuja causa mais frequente é a entrega"
    theme.insight(f"{promoters:.0f}% dão nota máxima, mas {detractors:.0f}% dão 3 ou menos. A média não mostra essa polarização; o foco deve ser {tail}.", "info")
    theme.table_view(scores)


LOGISTICS_TERMS = frozenset({
    "entrega", "entregue", "entregar", "entregaram", "prazo", "recebi", "receber", "recebido", "chegou", "chegar", "atraso", "atrasou",
    "atrasado", "demora", "demorou", "demorando", "correios", "transportadora", "postado", "rastreio", "rastreamento",
})


@st.cache_data(ttl=600, show_spinner="Analisando os comentários...")
def _word_shares(_engine: Engine, f: q.Filters):
    df = q.review_comments(_engine, f)
    if df.empty:
        return None
    result = {}
    for label, mask in (("neg", df["nota"] <= 2), ("pos", df["nota"] >= 4)):
        texts = df.loc[mask, "mensagem"]
        counter, logistics = Counter(), 0
        for message in texts:
            tokens = set(tokenize_pt(message))
            counter.update(tokens)
            logistics += bool(tokens & LOGISTICS_TERMS)
        result[label] = (len(texts), counter.most_common(12), logistics)
    return result if result["neg"][0] and result["pos"][0] else None


def _geography(engine: Engine, f: q.Filters) -> None:
    st.subheader("🗺️ Geografia")
    df = q.revenue_by_state(engine, f)
    if df.empty:
        st.info("Sem dados geográficos no filtro atual.")
        return
    df = df.astype({"receita": float, "frete_pct_preco": float})
    df["share"] = df["receita"] / df["receita"].sum() * 100
    top = df.sort_values("receita", ascending=False).head(12).sort_values("receita")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for region in theme.REGIONS:
            part = top[top["regiao"] == region]
            if part.empty:
                continue
            fig.add_bar(y=part["estado"], x=part["receita"], orientation="h", name=region, marker_color=theme.region_colors()[region],
                        text=[theme.fmt_pct(v, 1) for v in part["share"]], textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}<br>R$ %{x:,.0f}<extra>" + region + "</extra>")
        fig.update_layout(barmode="stack", xaxis_title="Receita (R$)")
        fig.update_yaxes(categoryorder="array", categoryarray=top["estado"].tolist())
        theme.headroom(fig, top["receita"].max())
        theme.style(fig, "Receita por estado do cliente (12 maiores)", height=470, legend=True)
        theme.plot(fig, key="geo_states")
    with col2:
        points = q.map_points(engine, f)
        if points.empty:
            st.info("Sem CEPs georreferenciados no filtro atual.")
        else:
            points = points.astype({"lat": float, "lng": float, "receita": float})
            fig_map = px.scatter_map(points, lat="lat", lon="lng", size="pedidos", size_max=24, opacity=0.55, zoom=2.6, center=dict(lat=-14, lon=-50),
                                     hover_name="cidade", hover_data={"estado": True, "pedidos": ":,", "receita": ":,.0f", "lat": False, "lng": False},
                                     color_discrete_sequence=[theme.colors()[0]], map_style="carto-darkmatter" if theme.mode() == "dark" else "carto-positron")
            theme.style(fig_map, "Onde estão os clientes (3.000 CEPs com mais pedidos)", height=470)
            theme.plot(fig_map, key="geo_map")

    regions = df.groupby("regiao").agg(receita=("receita", "sum"), pedidos=("pedidos", "sum")).reset_index()
    regions["share"] = regions["receita"] / regions["receita"].sum() * 100
    regions["ticket"] = regions["receita"] / regions["pedidos"]
    regions = regions.sort_values("receita", ascending=False)
    lead = df.sort_values("receita", ascending=False).iloc[0]
    sudeste = regions.loc[regions["regiao"] == "Sudeste", "share"].sum()
    freight_high = df[df["pedidos"] >= 200].sort_values("frete_pct_preco", ascending=False).head(3)["estado"].tolist()
    theme.insight(f"**{lead['estado']}** concentra {lead['share']:.0f}% da receita e o **Sudeste** {sudeste:.0f}%: a operação é fortemente dependente de uma região. "
                  f"O frete é proporcionalmente mais caro em {', '.join(freight_high)}, onde um centro de distribuição próximo reduziria custo e prazo e abriria mercado.")
    st.dataframe(regions.rename(columns={"regiao": "Região", "receita": "Receita (R$)", "pedidos": "Pedidos", "share": "% da receita", "ticket": "Ticket médio (R$)"}),
                 width="stretch", hide_index=True,
                 column_config={"Receita (R$)": st.column_config.NumberColumn(format="localized"), "Pedidos": st.column_config.NumberColumn(format="localized"),
                                "% da receita": st.column_config.NumberColumn(format="%.1f%%"), "Ticket médio (R$)": st.column_config.NumberColumn(format="%.2f")})


def _sellers(engine: Engine, f: q.Filters) -> None:
    st.subheader("🏪 Vendedores")
    df = q.sellers(engine, f)
    if df.empty:
        st.info("Sem dados de vendedores no filtro atual.")
        return
    df = df.astype({"receita": float, "nota_media": float, "pct_atraso": float}).sort_values("receita", ascending=False).reset_index(drop=True)
    total = df["receita"].sum()
    cum = df["receita"].cumsum() / total * 100
    n80 = int((cum < 80).sum() + 1)
    top10pct = df.head(max(1, int(len(df) * 0.1)))["receita"].sum() / total * 100
    top5 = df.head(5)["receita"].sum() / total * 100
    active = df[df["pedidos"] >= 30].dropna(subset=["nota_media", "pct_atraso"])

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure(go.Scatter(
            x=active["pct_atraso"], y=active["nota_media"], mode="markers", customdata=np.stack([active["seller_id"].str[:8], active["pedidos"], active["receita"]], axis=-1),
            marker=dict(size=np.clip(np.sqrt(active["receita"]) / 18, 5, 26), color=theme.colors()[0], opacity=0.45, line=dict(color="rgba(0,0,0,0)", width=0)),
            hovertemplate="Vendedor %{customdata[0]}…<br>atraso %{x:.1f}% · nota %{y:.2f}<br>%{customdata[1]} pedidos · R$ %{customdata[2]:,.0f}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="% de pedidos entregues com atraso", yaxis_title="Nota média", yaxis_range=[1, 5.2])
        theme.style(fig, f"Vendedores com 30+ pedidos ({theme.fmt_int(len(active))}): atraso × nota (tamanho = receita)", height=420)
        theme.plot(fig, key="sell_scatter")
    with col2:
        theme.metric(st, "Vendedores ativos", theme.fmt_int(len(df)))
        theme.metric(st, "Receita nos 5 maiores", theme.fmt_pct(top5))
        theme.metric(st, "Receita nos 10% maiores", theme.fmt_pct(top10pct))
        theme.metric(st, "Vendedores para 80% da receita", theme.fmt_int(n80), delta=f"{n80 / len(df) * 100:.0f}% da base")

    corr = active["pct_atraso"].corr(active["nota_media"]) if len(active) > 5 else np.nan
    risk = "risco de dependência" if top10pct > 60 else "base relativamente distribuída"
    theme.insight(f"os 10% maiores vendedores respondem por **{top10pct:.0f}%** da receita ({risk}); {n80} vendedores ({n80 / len(df) * 100:.0f}% da base) fazem 80% do faturamento. "
                  f"Entre os vendedores com volume, atraso e nota andam em sentido oposto (correlação {theme.fmt_num(corr, 2)}): **quem atrasa é mal avaliado**. "
                  "Vale definir metas de prazo (SLA) para os grandes e acompanhar a cauda de vendedores com atraso alto.")
    top = df.head(10).assign(seller=df["seller_id"].str[:8] + "…")[["seller", "estado", "receita", "pedidos", "nota_media", "pct_atraso"]]
    st.dataframe(top.rename(columns={"seller": "Vendedor", "estado": "UF", "receita": "Receita (R$)", "pedidos": "Pedidos", "nota_media": "Nota média", "pct_atraso": "% atraso"}),
                 width="stretch", hide_index=True,
                 column_config={"Receita (R$)": st.column_config.NumberColumn(format="localized"), "Pedidos": st.column_config.NumberColumn(format="localized"),
                                "Nota média": st.column_config.NumberColumn(format="%.2f"), "% atraso": st.column_config.NumberColumn(format="%.1f%%")})


def _customers(engine: Engine, f: q.Filters) -> None:
    st.subheader("🔁 Recompra e fidelidade")
    df = q.repeat_customers(engine, f)
    if df.empty:
        st.info("Sem dados de clientes no filtro atual.")
        return
    df = df.astype({"receita": float})
    repeat = df[df["pedidos"] >= 2]
    pct_repeat = len(repeat) / len(df) * 100
    rev_share = repeat["receita"].sum() / df["receita"].sum() * 100
    dist = df["pedidos"].clip(upper=5).value_counts().sort_index()
    recurrent = dist[dist.index >= 2]

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure(go.Bar(x=[f"{i} pedidos" if i < 5 else "5+ pedidos" for i in recurrent.index], y=recurrent.values, marker_color=theme.colors()[0],
                               text=[theme.fmt_int(v) for v in recurrent.values], textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}<br>%{y} clientes<extra></extra>"))
        fig.update_layout(yaxis_title="Clientes")
        theme.style(fig, "Clientes recorrentes por número de pedidos", height=340)
        theme.plot(fig, key="cust_repeat")
    with col2:
        theme.metric(st, "Clientes únicos", theme.fmt_int(len(df)))
        theme.metric(st, "Compraram mais de uma vez", theme.fmt_pct(pct_repeat), delta=f"{theme.fmt_int(len(repeat))} clientes")
        theme.metric(st, "Receita de recorrentes", theme.fmt_pct(rev_share))
    theme.insight(f"só **{pct_repeat:.1f}%** dos clientes voltaram a comprar. Quase toda a receita vem de primeira compra, o que torna o marketplace muito dependente de "
                  "aquisição paga. Aumentar a recompra em poucos pontos percentuais (e-mail pós-venda, cupom no 2º pedido, entrega no prazo) "
                  "tem retorno alto e custo baixo.", "warning")
    st.caption("Clientes identificados por `customer_unique_id`: o Olist gera um `customer_id` novo a cada pedido, então contar por `customer_id` esconderia a recompra.")
