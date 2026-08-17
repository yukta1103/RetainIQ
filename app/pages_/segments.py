"""Page 2 — Customer Segments: RFM breakdown with drill-down."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, data
from app import theme as T


def render(f: data.Filters) -> None:
    st.title("Customer Segments")
    st.caption(
        "RFM segmentation on 93,104 customers. Thresholds are behavioural, "
        "not quintiles — see the methodology note below."
    )

    seg = data.filter_customers(data.load("segments"), f)
    if seg.empty:
        st.warning("No customers match the current filters.")
        return

    summary = _summarise(seg)

    total_rev = seg["monetary"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers in view", f"{len(seg):,}")
    c2.metric("Revenue in view", data.fmt_brl(total_rev))
    c3.metric("Repeat customers", f"{int(seg['is_repeat'].sum()):,}",
              help="Customers with more than one distinct purchase day.")
    c4.metric("Segments present", f"{(summary['customers'] > 0).sum()}")

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(charts.segment_bars(summary), use_container_width=True)
    with right:
        st.plotly_chart(charts.segment_value_scatter(summary), use_container_width=True)

    st.markdown("#### Segment detail")
    show = summary[summary["customers"] > 0].copy()
    st.dataframe(
        show[["segment", "customers", "pct_customers", "revenue", "pct_revenue",
              "revenue_index", "avg_monetary", "avg_recency", "avg_frequency",
              "repeat_rate", "action"]].rename(columns={
            "segment": "Segment", "customers": "Customers",
            "pct_customers": "% of customers", "revenue": "Revenue",
            "pct_revenue": "% of revenue", "revenue_index": "Revenue index",
            "avg_monetary": "Avg spend", "avg_recency": "Avg recency (d)",
            "avg_frequency": "Avg frequency", "repeat_rate": "Repeat %",
            "action": "What it means"}),
        hide_index=True, use_container_width=True,
        column_config={
            "Revenue": st.column_config.NumberColumn(format="R$ %.0f"),
            "Avg spend": st.column_config.NumberColumn(format="R$ %.0f"),
            "% of customers": st.column_config.NumberColumn(format="%.2f%%"),
            "% of revenue": st.column_config.NumberColumn(format="%.2f%%"),
            "Repeat %": st.column_config.NumberColumn(format="%.1f%%"),
            "Revenue index": st.column_config.NumberColumn(
                format="%.2f", help=">1 means the segment earns more revenue than its size implies."),
            "Avg recency (d)": st.column_config.NumberColumn(format="%.0f"),
            "Avg frequency": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.markdown("#### Drill into a segment")
    names = [s for s in summary.loc[summary["customers"] > 0, "segment"]]
    picked = st.multiselect("Segments", names, default=names[:1] or None)
    sub = seg[seg["segment"].isin(picked)] if picked else seg

    if sub.empty:
        st.info("Pick at least one segment.")
        return

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Customers", f"{len(sub):,}")
    d2.metric("Revenue", data.fmt_brl(sub["monetary"].sum()))
    d3.metric("Median spend", data.fmt_brl(sub["monetary"].median(), 2))
    d4.metric("Median recency", f"{sub['recency'].median():.0f} d")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.histogram(sub["monetary"].clip(upper=sub["monetary"].quantile(0.99)),
                             "Spend distribution (99th pct clipped)", "Total spend (R$)"),
            use_container_width=True)
    with right:
        st.plotly_chart(
            charts.histogram(sub["recency"], "Recency distribution",
                             "Days since last purchase", color=T.ORANGE),
            use_container_width=True)

    left, right = st.columns(2)
    with left:
        top_state = (sub.groupby("customer_state", observed=True)
                     .size().reset_index(name="customers")
                     .sort_values("customers", ascending=False).head(10))
        st.markdown("**Top states**")
        st.dataframe(top_state, hide_index=True, use_container_width=True)
    with right:
        top_cat = (sub.groupby("top_category", observed=True)
                   .size().reset_index(name="customers")
                   .sort_values("customers", ascending=False).head(10))
        st.markdown("**Top categories**")
        st.dataframe(top_cat, hide_index=True, use_container_width=True)

    with st.expander("Why these thresholds, and not quintiles?"):
        meta = data.meta()
        st.markdown(f"""
Quintiles are **undefined** on this data: 97.85% of customers have frequency 1,
so four of five frequency quintiles would contain identical customers and the
split would be arbitrary. Each cut is behavioural instead.

**Recency — 90 / 180 / 365 days.** Taken from the observed repurchase curve,
excluding same-day basket splits. Of genuine repurchases, 56.0% happen within
90 days, 76.9% within 180, and 96.4% within 365 — so the cuts sit on real
inflection points. Past 365 days only 3.6% of repurchases ever occur, which is
what makes "Lost" a defensible label rather than a guess.

**Frequency — distinct purchase days.** Olist splits one basket into an order
per seller; 29.6% of consecutive order pairs are under 24 hours apart. Counting
orders would inflate the repeat rate from 2.15% to 3.00%.

**Monetary — {data.fmt_brl(float(meta['monetary_cut_low']), 2)} and
{data.fmt_brl(float(meta['monetary_cut_high']), 2)}** (1x and 3x AOV). AOV
multiples are business-interpretable, and the 3x cut lands almost exactly on
the 95th percentile — where revenue concentration bites.

Snapshot date for recency: **{meta['snapshot_date']}**.
        """)


@st.cache_data(show_spinner=False)
def _summarise(seg: pd.DataFrame) -> pd.DataFrame:
    """Recompute segment summary over the filtered slice."""
    from retainiq.analytics.rfm import SEGMENT_ACTIONS, SEGMENT_ORDER

    total_rev = seg["monetary"].sum()
    total_cust = len(seg)
    s = (
        seg.groupby("segment", observed=False)
        .agg(customers=(data.CUSTOMER_KEY, "size"),
             revenue=("monetary", "sum"),
             avg_monetary=("monetary", "mean"),
             avg_recency=("recency", "mean"),
             avg_frequency=("frequency", "mean"),
             repeat_rate=("is_repeat", "mean"))
        .reset_index()
    )
    s["pct_customers"] = 100.0 * s["customers"] / max(total_cust, 1)
    s["pct_revenue"] = 100.0 * s["revenue"] / max(total_rev, 1e-9)
    s["revenue_index"] = s["pct_revenue"] / s["pct_customers"].replace(0, pd.NA)
    s["repeat_rate"] *= 100.0
    s["segment"] = s["segment"].astype(str)
    s["action"] = s["segment"].map(SEGMENT_ACTIONS)
    s["_ord"] = s["segment"].map({n: i for i, n in enumerate(SEGMENT_ORDER)})
    return s.sort_values("revenue", ascending=False).drop(columns="_ord").reset_index(drop=True)
