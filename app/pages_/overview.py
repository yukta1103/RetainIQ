"""Page 1 — Overview: headline KPIs and trends over the filtered slice."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, data
from app import theme as T
from retainiq.analytics import metrics


def render(f: data.Filters) -> None:
    st.title("Overview")
    st.caption(
        "Brazilian e-commerce marketplace (Olist), Jan 2017 – Aug 2018. "
        "All figures respond to the sidebar filters."
    )

    orders = data.apply_filters(data.load("orders"), f)
    if orders.empty:
        st.warning("No orders match the current filters. Widen the selection.")
        return

    m = metrics.headline_metrics(orders)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total revenue", data.fmt_brl(m["total_revenue"]))
    c2.metric("Orders", f"{m['n_orders']:,}")
    c3.metric("Customers", f"{m['n_customers']:,}")
    c4.metric("Average order value", data.fmt_brl(m["aov"], 2))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Repeat purchase rate", data.fmt_pct(m["repeat_purchase_rate"], 2))
    c2.metric("Revenue per customer", data.fmt_brl(m["revenue_per_customer"], 2))
    c3.metric("Avg review score", f"{m['avg_review_score']:.2f} / 5")
    c4.metric("Late deliveries", data.fmt_pct(m["late_delivery_rate"], 2))

    st.markdown(
        "<div class='caveat'><b>On the repeat purchase rate.</b> This counts "
        "distinct purchase <i>days</i>, not orders. Olist splits one basket into "
        "a separate order per seller, so 29.6% of consecutive order pairs are "
        "under 24h apart. Counting those as repeat purchases would report 3.00% "
        "instead of the true 2.15%.</div>",
        unsafe_allow_html=True,
    )

    trend = metrics.monthly_trend(orders)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.trend_line(trend, "order_month", "revenue",
                              "Monthly revenue", "Revenue (R$)", money=True),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            charts.trend_line(trend, "order_month", "orders",
                              "Monthly orders", "Orders", color=T.ORANGE),
            use_container_width=True,
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.trend_line(trend, "order_month", "aov",
                              "Average order value", "AOV (R$)", money=True),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(charts.new_vs_returning(trend), use_container_width=True)

    st.markdown("#### Where the revenue comes from")
    left, right = st.columns(2)
    with left:
        cat = data.load("category_monthly")
        if f.categories:
            cat = cat[cat["product_category"].isin(f.categories)]
        cat_agg = (
            cat.groupby("product_category", observed=True)["revenue"].sum()
            .reset_index().sort_values("revenue", ascending=False)
        )
        st.plotly_chart(charts.category_bars(cat_agg), use_container_width=True)
    with right:
        st_rev = (
            orders.groupby("customer_state", observed=True)
            .agg(revenue=("order_revenue", "sum"),
                 customers=(data.CUSTOMER_KEY, "nunique"),
                 orders=("order_id", "size"),
                 aov=("order_revenue", "mean"))
            .reset_index().sort_values("revenue", ascending=False)
        )
        st_rev["pct_revenue"] = 100 * st_rev["revenue"] / st_rev["revenue"].sum()
        st.markdown("**Revenue by state**")
        st.dataframe(
            st_rev.head(12).rename(columns={
                "customer_state": "State", "revenue": "Revenue",
                "customers": "Customers", "orders": "Orders",
                "aov": "AOV", "pct_revenue": "% of revenue"}),
            hide_index=True, use_container_width=True, height=420,
            column_config={
                "Revenue": st.column_config.NumberColumn(format="R$ %.0f"),
                "AOV": st.column_config.NumberColumn(format="R$ %.2f"),
                "% of revenue": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    with st.expander("Show the underlying monthly table"):
        st.dataframe(
            trend[["order_month", "revenue", "orders", "customers", "aov",
                   "avg_review", "late_rate"]],
            hide_index=True, use_container_width=True,
        )
