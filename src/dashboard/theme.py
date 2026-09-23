"""
Dashboard visual theme: one validated palette, Plotly layout defaults and pt-BR number formatting.

Palette rules (dataviz method): categorical hues are assigned in a fixed order and follow the
entity (a region is always the same color), magnitude uses a single blue ramp, polarity uses
blue <-> red around a neutral gray, and status colors are reserved for good/warning/critical
cues that always come with a text label. Values are validated for both light and dark surfaces.
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Categorical slots (fixed order): blue, orange, aqua, yellow, magenta, green, violet, red.
_CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
_INK = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}
# Sequential blue ramp (light -> dark) and diverging pair with a neutral gray midpoint.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = {
    "light": ["#e34948", "#f0efec", "#2a78d6"],
    "dark": ["#e66767", "#383835", "#3987e5"],
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

REGIONS = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]

MONTH_ABBR_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
DAY_ABBR_PT = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
DAY_FULL_PT = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]


def mode() -> str:
    """'dark' or 'light', following the Streamlit theme (light when it cannot be detected)."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001 - older Streamlit / no script context
        return "light"


def colors() -> list:
    return _CATEGORICAL[mode()]


def color_for(entity: str, order: list) -> str:
    """Color that follows the entity (not its rank): index in a fixed, filter-independent list."""
    palette = colors()
    return palette[order.index(entity) % len(palette)] if entity in order else _INK[mode()]["muted"]


def region_colors() -> dict:
    return {region: color_for(region, REGIONS) for region in REGIONS}


def sequential_scale() -> list:
    """Blue ramp where the lightest step means 'near zero': reversed on dark so low values recede into the surface."""
    return SEQUENTIAL_BLUE[::-1] if mode() == "dark" else SEQUENTIAL_BLUE


def diverging_scale() -> list:
    return DIVERGING[mode()]


def status_color(level: str) -> str:
    return STATUS[level]


def md(text: str) -> str:
    """Escape '$' so Streamlit markdown does not read 'R$ 30 a R$ 100' as a LaTeX formula."""
    return text.replace("$", "\\$")


def metric(container, label: str, value, delta: Optional[str] = None, help: Optional[str] = None) -> None:
    """st.metric with a neutral note under the value (no misleading up/down arrow) and '$'-safe label."""
    container.metric(md(label), value, delta=delta, delta_color="off", delta_arrow="off", help=help)


def headroom(fig: go.Figure, max_value: float, axis: str = "x", pad: float = 0.22) -> None:
    """Extend a value axis past the longest bar so outside data labels are not clipped."""
    limit = [0, float(max_value) * (1 + pad)]
    (fig.update_xaxes if axis == "x" else fig.update_yaxes)(range=limit)


def style(fig: go.Figure, title: Optional[str] = None, height: int = 380, legend: bool = False, legend_rows: int = 1) -> go.Figure:
    """Apply the shared look: transparent surface, recessive grid, thin marks, ink-colored text."""
    ink = _INK[mode()]
    top = (48 if title else 16) + (26 * legend_rows if legend else 0)
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=ink["primary"])) if title else None,
        height=height,
        margin=dict(l=8, r=8, t=top, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color=ink["secondary"]),
        colorway=colors(),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(font_size=12),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=ink["grid"], linecolor=ink["axis"], zerolinecolor=ink["axis"], tickfont=dict(color=ink["muted"]),
                     title_font=dict(color=ink["secondary"]), showgrid=False)
    fig.update_yaxes(gridcolor=ink["grid"], linecolor="rgba(0,0,0,0)", zerolinecolor=ink["axis"], tickfont=dict(color=ink["muted"]),
                     title_font=dict(color=ink["secondary"]))
    return fig


def plot(fig: go.Figure, key: Optional[str] = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def table_view(df: pd.DataFrame, label: str = "Ver os dados deste gráfico") -> None:
    """Accessible table alternative to a chart (required when some colors have low contrast)."""
    with st.expander(label):
        st.dataframe(df, width="stretch", hide_index=True)


def insight(text: str, level: str = "info") -> None:
    """Business reading of a chart. `level` picks the Streamlit callout; text always states the meaning."""
    {"info": st.info, "warning": st.warning, "success": st.success, "error": st.error}[level](md(f"**Leitura:** {text}"))


# ----------------------------------------------------------------------
# pt-BR formatting
# ----------------------------------------------------------------------


def fmt_int(value) -> str:
    return f"{int(round(value)):,}".replace(",", ".") if pd.notna(value) else "—"


def fmt_brl(value, decimals: int = 0) -> str:
    if pd.isna(value):
        return "—"
    text = f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {text}"


def fmt_brl_short(value) -> str:
    """R$ 1,2 mi / R$ 350 mil / R$ 980."""
    if pd.isna(value):
        return "—"
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"R$ {value / 1_000_000:.1f} mi".replace(".", ",")
    if absolute >= 10_000:
        return f"R$ {value / 1_000:.0f} mil"
    return fmt_brl(value)


def fmt_pct(value, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%".replace(".", ",") if pd.notna(value) else "—"


def fmt_num(value, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(value) else "—"
