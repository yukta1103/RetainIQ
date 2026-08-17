"""Plotly chart builders.

Conventions applied throughout, per the data-viz method:
  * one y-axis per chart -- two measures of different scale become two charts
  * magnitude -> single hue; identity -> fixed categorical slots, never cycled
  * sequential heatmaps use one hue, light->dark
  * 2px lines, >=8px markers, 4px rounded bar ends, hairline horizontal grid
  * a legend whenever there are >=2 series; direct labels when there are <=4
  * hover on every mark
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app import theme as T

LINE_W = 2
MARKER = 8


def _fig(title: str, height: int = 340, ylab: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, height=height)
    if ylab:
        fig.update_yaxes(title=dict(text=ylab, font=dict(size=12, color=T.INK_MUTED)))
    return fig


def trend_line(
    df: pd.DataFrame, x: str, y: str, title: str, ylab: str,
    color: str = T.BLUE, money: bool = False, height: int = 320,
) -> go.Figure:
    """Single-measure time series. One series, so no legend box -- title names it."""
    fig = _fig(title, height, ylab)
    hover = "%{x|%b %Y}<br>" + (
        "R$ %{y:,.0f}<extra></extra>" if money else "%{y:,.0f}<extra></extra>"
    )
    fig.add_trace(
        go.Scatter(
            x=df[x], y=df[y], mode="lines+markers",
            line=dict(color=color, width=LINE_W),
            marker=dict(size=MARKER, color=color, line=dict(width=2, color=T.SURFACE)),
            hovertemplate=hover, name=ylab,
        )
    )
    fig.update_layout(hovermode="x unified", showlegend=False)
    return fig


def new_vs_returning(df: pd.DataFrame, height: int = 340) -> go.Figure:
    """Two-series stacked bars: a 2px surface gap separates the segments."""
    fig = _fig("Revenue from new vs returning customers", height, "Revenue (R$)")
    for name, col, color in [
        ("New customers", "new_customer_revenue", T.BLUE),
        ("Returning", "returning_revenue", T.ORANGE),
    ]:
        fig.add_trace(
            go.Bar(
                x=df["order_month"], y=df[col], name=name,
                marker=dict(color=color, line=dict(width=2, color=T.SURFACE)),
                hovertemplate="%{x|%b %Y}<br>" + name + ": R$ %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack", hovermode="x unified", bargap=0.25)
    return fig


def segment_bars(summary: pd.DataFrame, height: int = 420) -> go.Figure:
    """Segment sizes -- magnitude, so one hue, sorted descending."""
    d = summary.sort_values("customers")
    fig = _fig("Customers per segment", height, "")
    fig.add_trace(
        go.Bar(
            x=d["customers"], y=d["segment"], orientation="h",
            marker=dict(color=T.BLUE, line=dict(width=2, color=T.SURFACE)),
            text=[f"{v:,.0f}" for v in d["customers"]],
            textposition="outside",
            textfont=dict(size=11, color=T.INK_SECONDARY),
            hovertemplate="<b>%{y}</b><br>%{x:,.0f} customers<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor=T.GRIDLINE, showline=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(showlegend=False, margin=dict(l=8, r=64, t=48, b=8))
    return fig


def segment_value_scatter(summary: pd.DataFrame, height: int = 420) -> go.Figure:
    """Share of customers vs share of revenue. The diagonal is 'fair share'."""
    d = summary[summary["customers"] > 0]
    fig = _fig("Segment share of customers vs share of revenue", height, "% of revenue")

    lim = max(d["pct_customers"].max(), d["pct_revenue"].max()) * 1.15
    fig.add_trace(
        go.Scatter(
            x=[0, lim], y=[0, lim], mode="lines",
            line=dict(color=T.BASELINE, width=1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_annotation(
        x=lim * 0.72, y=lim * 0.78, text="equal share", showarrow=False,
        font=dict(size=11, color=T.INK_MUTED),
    )
    fig.add_trace(
        go.Scatter(
            x=d["pct_customers"], y=d["pct_revenue"],
            mode="markers+text",
            text=d["segment"], textposition="top center",
            textfont=dict(size=11, color=T.INK_SECONDARY),
            marker=dict(
                size=np.clip(np.sqrt(d["revenue"]) / 40, 10, 34),
                color=[T.SEGMENT_COLORS.get(s, T.BLUE) for s in d["segment"]],
                line=dict(width=2, color=T.SURFACE),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>%{x:.2f}% of customers<br>"
                "%{y:.2f}% of revenue<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_xaxes(title=dict(text="% of customers", font=dict(size=12, color=T.INK_MUTED)),
                     showgrid=True, gridcolor=T.GRIDLINE, showline=False)
    fig.update_layout(hovermode="closest")
    return fig


def cohort_heatmap(
    mat: pd.DataFrame, sizes: pd.Series, title: str, zmax: float | None = None,
    height: int = 520, unit: str = "%",
) -> go.Figure:
    """Sequential single-hue heatmap. NaN cells (unobservable) stay blank."""
    z = mat.to_numpy(dtype=float)
    cohorts = [c.strftime("%Y-%m") for c in mat.index]
    months = [f"M{c}" for c in mat.columns]

    text = np.where(np.isnan(z), "", np.round(z, 1).astype(str))
    fig = _fig(title, height)
    fig.add_trace(
        go.Heatmap(
            z=z, x=months, y=cohorts,
            colorscale=T.SEQUENTIAL,
            zmin=0, zmax=zmax if zmax is not None else np.nanmax(z),
            text=text, texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate=(
                "cohort %{y}<br>month %{x}<br>" + unit + ": %{z:.2f}<extra></extra>"
            ),
            xgap=2, ygap=2,
            colorbar=dict(
                title=dict(text=unit, font=dict(size=11, color=T.INK_MUTED)),
                thickness=12, len=0.7, outlinewidth=0,
                tickfont=dict(size=11, color=T.INK_MUTED),
            ),
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, title=dict(
        text="first-purchase cohort", font=dict(size=12, color=T.INK_MUTED)))
    fig.update_xaxes(side="top", showgrid=False, showline=False)
    fig.update_layout(margin=dict(l=8, r=8, t=64, b=8))
    return fig


def cohort_curves(mat: pd.DataFrame, height: int = 380) -> go.Figure:
    """Cumulative repeat curves. >4 cohorts, so legend carries identity."""
    fig = _fig("Cumulative repeat rate by cohort age", height, "% of cohort returned")
    n = len(mat.index)
    for i, cohort in enumerate(mat.index):
        row = mat.loc[cohort].dropna()
        if row.empty:
            continue
        # Single-hue ordinal ramp: older cohorts darker. Identity is secondary
        # here -- the shape of the family is the message.
        shade = 0.25 + 0.75 * (i / max(n - 1, 1))
        fig.add_trace(
            go.Scatter(
                x=[int(c) for c in row.index], y=row.to_numpy(),
                mode="lines", name=cohort.strftime("%Y-%m"),
                line=dict(color=_ramp(shade), width=LINE_W),
                hovertemplate=f"{cohort:%Y-%m}<br>month %{{x}}: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_xaxes(title=dict(text="months since first purchase",
                                font=dict(size=12, color=T.INK_MUTED)))
    fig.update_layout(hovermode="closest", legend=dict(font=dict(size=10)))
    return fig


def _ramp(t: float) -> str:
    """Interpolate the sequential blue ramp at position t in [0, 1]."""
    stops = [(s[0], s[1]) for s in T.SEQUENTIAL]
    t = float(np.clip(t, 0, 1))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if p0 <= t <= p1:
            f = 0 if p1 == p0 else (t - p0) / (p1 - p0)
            a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
            b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
            return "#%02x%02x%02x" % tuple(int(a[i] + f * (b[i] - a[i])) for i in range(3))
    return stops[-1][1]


def decile_calibration(cal: pd.DataFrame, title: str, height: int = 380) -> go.Figure:
    """Predicted vs actual per decile -- 2 series, both direct-labelled."""
    fig = _fig(title, height, "Mean 90-day spend (R$)")
    for name, col, color in [
        ("Predicted", "mean_predicted", T.BLUE),
        ("Actual", "mean_actual", T.ORANGE),
    ]:
        fig.add_trace(
            go.Bar(
                x=cal["decile"].astype(str), y=cal[col], name=name,
                marker=dict(color=color, line=dict(width=2, color=T.SURFACE)),
                hovertemplate="%{x} " + name + ": R$ %{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08,
                      hovermode="x unified")
    fig.update_xaxes(title=dict(text="predicted-value decile (D1 = highest)",
                                font=dict(size=12, color=T.INK_MUTED)))
    return fig


def revenue_capture(cal: pd.DataFrame, height: int = 340) -> go.Figure:
    """Share of actual holdout revenue per predicted decile -- magnitude, one hue."""
    fig = _fig("Where the actual revenue landed, by predicted decile", height,
               "% of holdout revenue")
    fig.add_trace(
        go.Bar(
            x=cal["decile"].astype(str), y=cal["pct_of_actual_revenue"],
            marker=dict(color=T.BLUE, line=dict(width=2, color=T.SURFACE)),
            text=[f"{v:.1f}%" for v in cal["pct_of_actual_revenue"]],
            textposition="outside", textfont=dict(size=11, color=T.INK_SECONDARY),
            hovertemplate="%{x}: %{y:.1f}% of revenue<extra></extra>",
        )
    )
    fig.add_hline(y=10, line=dict(color=T.BASELINE, width=1, dash="dot"))
    fig.add_annotation(x=cal["decile"].astype(str).iloc[-1], y=10,
                       text="random targeting = 10%", showarrow=False, yshift=12,
                       font=dict(size=11, color=T.INK_MUTED))
    fig.update_layout(showlegend=False, bargap=0.3)
    return fig


def clv_by_segment(df: pd.DataFrame, value_col: str, height: int = 400) -> go.Figure:
    """Predicted-CLV distribution per segment. Box plots; identity via position."""
    order = (
        df.groupby("segment", observed=True)[value_col].median()
        .sort_values(ascending=False).index.tolist()
    )
    fig = _fig("Predicted 90-day CLV distribution by segment", height,
               "Predicted spend (R$)")
    for seg in order:
        vals = df.loc[df["segment"] == seg, value_col]
        if vals.empty:
            continue
        fig.add_trace(
            go.Box(
                y=vals, name=str(seg),
                marker=dict(color=T.SEGMENT_COLORS.get(seg, T.BLUE), size=4),
                line=dict(width=1.5),
                fillcolor="rgba(0,0,0,0)",
                boxpoints=False,
                hovertemplate=(
                    f"<b>{seg}</b><br>median R$ %{{median:.2f}}<extra></extra>"
                ),
            )
        )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-30)
    return fig


def model_comparison(evals: pd.DataFrame, metric: str, title: str,
                     good_high: bool = True, height: int = 320) -> go.Figure:
    """Model scores on one metric. Magnitude -> one hue; the winner is annotated."""
    d = evals.sort_values(metric, ascending=not good_high)
    best = d[metric].iloc[0]
    colors = [T.BLUE if v == best else "#9ec5f4" for v in d[metric]]
    fig = _fig(title, height, metric)
    fig.add_trace(
        go.Bar(
            x=d["model"], y=d[metric],
            marker=dict(color=colors, line=dict(width=2, color=T.SURFACE)),
            text=[f"{v:.3f}" for v in d[metric]],
            textposition="outside", textfont=dict(size=11, color=T.INK_SECONDARY),
            hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False, bargap=0.35)
    fig.update_xaxes(tickangle=-20)
    return fig


def category_bars(cat: pd.DataFrame, n: int = 12, height: int = 420) -> go.Figure:
    d = cat.head(n).sort_values("revenue")
    fig = _fig(f"Top {n} categories by revenue", height, "")
    fig.add_trace(
        go.Bar(
            x=d["revenue"], y=d["product_category"], orientation="h",
            marker=dict(color=T.BLUE, line=dict(width=2, color=T.SURFACE)),
            text=[f"R$ {v/1000:,.0f}k" for v in d["revenue"]],
            textposition="outside", textfont=dict(size=11, color=T.INK_SECONDARY),
            hovertemplate="<b>%{y}</b><br>R$ %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor=T.GRIDLINE, showline=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(showlegend=False, margin=dict(l=8, r=72, t=48, b=8))
    return fig


def histogram(values: pd.Series, title: str, xlab: str, color: str = T.BLUE,
              nbins: int = 50, height: int = 320, logy: bool = False) -> go.Figure:
    fig = _fig(title, height, "Customers")
    fig.add_trace(
        go.Histogram(
            x=values, nbinsx=nbins,
            marker=dict(color=color, line=dict(width=1, color=T.SURFACE)),
            hovertemplate=xlab + ": %{x}<br>%{y:,.0f} customers<extra></extra>",
        )
    )
    fig.update_xaxes(title=dict(text=xlab, font=dict(size=12, color=T.INK_MUTED)))
    if logy:
        fig.update_yaxes(type="log")
    fig.update_layout(showlegend=False, hovermode="closest", bargap=0.02)
    return fig
