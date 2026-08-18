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


def lorenz_curve(monetary: pd.Series, height: int = 400) -> go.Figure:
    """Cumulative revenue share vs cumulative customer share.

    Computed straight from the spend distribution, with no reference to the
    segmentation -- which is the point. A segment defined by high spend will
    always look revenue-dense; this curve cannot be circular because nothing
    about it depends on how customers were grouped.
    """
    v = np.sort(monetary.to_numpy())[::-1]
    cum = np.cumsum(v) / v.sum() * 100
    share = np.arange(1, len(v) + 1) / len(v) * 100

    fig = _fig("Revenue concentration across the customer base", height,
               "% of cumulative revenue")
    fig.add_trace(
        go.Scatter(
            x=[0, 100], y=[0, 100], mode="lines",
            line=dict(color=T.BASELINE, width=1, dash="dot"),
            name="perfect equality", hoverinfo="skip",
        )
    )
    step = max(len(v) // 2000, 1)
    fig.add_trace(
        go.Scatter(
            x=share[::step], y=cum[::step], mode="lines",
            line=dict(color=T.BLUE, width=LINE_W), name="observed",
            hovertemplate="top %{x:.1f}% of customers<br>"
                          "hold %{y:.1f}% of revenue<extra></extra>",
        )
    )
    for pct, color in [(5, T.ORANGE), (20, T.ORANGE)]:
        y = cum[int(len(v) * pct / 100) - 1]
        fig.add_trace(
            go.Scatter(
                x=[pct], y=[y], mode="markers+text",
                text=[f"  top {pct}% → {y:.1f}%"], textposition="middle right",
                textfont=dict(size=11, color=T.INK_SECONDARY),
                marker=dict(size=MARKER + 2, color=color,
                            line=dict(width=2, color=T.SURFACE)),
                hovertemplate=f"top {pct}% of customers hold {y:.1f}% of revenue<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_xaxes(title=dict(text="% of customers, ranked by spend",
                                font=dict(size=12, color=T.INK_MUTED)),
                     showgrid=True, gridcolor=T.GRIDLINE, showline=False)
    fig.update_layout(hovermode="closest")
    return fig


def quintile_calibration(cal: pd.DataFrame, title: str, height: int = 400) -> go.Figure:
    """Predicted vs actual per quintile, with a bootstrap band on actual.

    Quintiles rather than deciles: at 418 holdout positives a decile holds ~41
    (binomial sd ~6.5) and its wobble is noise. The error bars are the point --
    they show which bins are actually distinguishable and which are not.
    """
    fig = _fig(title, height, "Mean 90-day spend (R$)")
    x = cal["bin"].astype(str)

    fig.add_trace(
        go.Bar(
            x=x, y=cal["mean_predicted"], name="Predicted",
            marker=dict(color=T.BLUE, line=dict(width=2, color=T.SURFACE)),
            hovertemplate="%{x} predicted: R$ %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x, y=cal["mean_actual"], name="Actual (95% CI)",
            marker=dict(color=T.ORANGE, line=dict(width=2, color=T.SURFACE)),
            error_y=dict(
                type="data", symmetric=False,
                array=(cal["actual_hi"] - cal["mean_actual"]).to_numpy(),
                arrayminus=(cal["mean_actual"] - cal["actual_lo"]).to_numpy(),
                color=T.INK_SECONDARY, thickness=1.5, width=6,
            ),
            hovertemplate=("%{x} actual: R$ %{y:,.2f}<extra></extra>"),
        )
    )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08,
                      hovermode="x unified")
    fig.update_xaxes(title=dict(text="predicted-score quintile (Q1 = highest)",
                                font=dict(size=12, color=T.INK_MUTED)))
    return fig


def revenue_capture(cal: pd.DataFrame, height: int = 360) -> go.Figure:
    """Share of actual holdout revenue per predicted quintile."""
    n = len(cal)
    even = 100.0 / n
    fig = _fig("Where the actual revenue landed, by predicted score", height,
               "% of holdout revenue")
    fig.add_trace(
        go.Bar(
            x=cal["bin"].astype(str), y=cal["pct_of_actual_revenue"],
            marker=dict(color=T.BLUE, line=dict(width=2, color=T.SURFACE)),
            text=[f"{v:.1f}%" for v in cal["pct_of_actual_revenue"]],
            textposition="outside", textfont=dict(size=11, color=T.INK_SECONDARY),
            hovertemplate="%{x}: %{y:.1f}% of revenue<extra></extra>",
        )
    )
    fig.add_hline(y=even, line=dict(color=T.BASELINE, width=1, dash="dot"))
    fig.add_annotation(
        x=cal["bin"].astype(str).iloc[-1], y=even,
        text=f"random targeting = {even:.0f}%", showarrow=False, yshift=12,
        font=dict(size=11, color=T.INK_MUTED),
    )
    fig.update_layout(showlegend=False, bargap=0.3)
    return fig


def auc_intervals(ev: pd.DataFrame, height: int = 360) -> go.Figure:
    """Model AUCs as point + 95% interval against the 0.5 random line.

    A bar chart of point estimates would imply separation that the intervals
    do not support, so the interval is the mark, not decoration.
    """
    d = ev.sort_values("AUC")
    fig = _fig("Ranking quality: AUC with 95% bootstrap intervals", height, "")
    fig.add_vline(x=0.5, line=dict(color=T.CRITICAL, width=1, dash="dot"))
    fig.add_annotation(x=0.5, y=-0.6, text="random", showarrow=False,
                       font=dict(size=11, color=T.CRITICAL), yshift=-6)
    for _, r in d.iterrows():
        beats = r["AUC_lo"] > 0.5
        color = T.BLUE if beats else T.INK_MUTED
        fig.add_trace(
            go.Scatter(
                x=[r["AUC_lo"], r["AUC_hi"]], y=[r["model"], r["model"]],
                mode="lines", line=dict(color=color, width=3),
                hoverinfo="skip", showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[r["AUC"]], y=[r["model"]], mode="markers",
                marker=dict(size=MARKER + 2, color=color,
                            line=dict(width=2, color=T.SURFACE)),
                hovertemplate=(f"<b>{r['model']}</b><br>AUC {r['AUC']:.3f} "
                               f"[{r['AUC_lo']:.3f}, {r['AUC_hi']:.3f}]<extra></extra>"),
                showlegend=False,
            )
        )
    fig.update_xaxes(showgrid=True, gridcolor=T.GRIDLINE, showline=False,
                     title=dict(text="AUC", font=dict(size=12, color=T.INK_MUTED)))
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=8, r=24, t=48, b=32))
    return fig


def score_by_segment(df: pd.DataFrame, value_col: str, height: int = 400) -> go.Figure:
    """Propensity-score distribution per segment. Box plots; identity via position.

    Axis says 'score', not 'predicted CLV': the values order customers by
    repurchase likelihood and are not calibrated per-customer spend forecasts.
    """
    order = (
        df.groupby("segment", observed=True)[value_col].median()
        .sort_values(ascending=False).index.tolist()
    )
    fig = _fig("Repeat-propensity score distribution by segment", height,
               "Propensity score (R$-denominated, ordering only)")
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
