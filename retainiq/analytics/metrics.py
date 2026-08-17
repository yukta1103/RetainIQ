"""Headline business metrics for the dashboard Overview page."""

from __future__ import annotations

import pandas as pd

from retainiq.config import CUSTOMER_KEY


def headline_metrics(orders: pd.DataFrame) -> dict[str, float]:
    """KPIs over whatever slice of orders is passed in (respects filters)."""
    if orders.empty:
        return {k: 0.0 for k in (
            "total_revenue", "n_orders", "n_customers", "aov",
            "repeat_purchase_rate", "revenue_per_customer", "avg_items_per_order",
            "avg_review_score", "late_delivery_rate",
        )}

    o = orders.copy()
    o["purchase_day"] = o["order_purchase_timestamp"].dt.normalize()
    occasions = o.groupby(CUSTOMER_KEY)["purchase_day"].nunique()
    n_customers = len(occasions)

    return {
        "total_revenue": float(o["order_revenue"].sum()),
        "n_orders": int(len(o)),
        "n_customers": int(n_customers),
        "aov": float(o["order_revenue"].mean()),
        # Occasion-based, consistent with rfm.py — see that module for why.
        "repeat_purchase_rate": float(100.0 * (occasions > 1).mean()),
        "revenue_per_customer": float(o["order_revenue"].sum() / n_customers),
        "avg_items_per_order": float(o["n_items"].mean()),
        "avg_review_score": float(o["review_score"].mean()),
        "late_delivery_rate": float(100.0 * o["delivered_late"].mean()),
    }


def monthly_trend(orders: pd.DataFrame) -> pd.DataFrame:
    """Month-by-month series for the Overview trend charts."""
    o = orders.copy()
    o["purchase_day"] = o["order_purchase_timestamp"].dt.normalize()

    trend = o.groupby("order_month").agg(
        revenue=("order_revenue", "sum"),
        orders=("order_id", "size"),
        customers=(CUSTOMER_KEY, "nunique"),
        aov=("order_revenue", "mean"),
        avg_review=("review_score", "mean"),
        late_rate=("delivered_late", "mean"),
    ).reset_index()

    trend["late_rate"] *= 100.0
    trend["revenue_per_customer"] = trend["revenue"] / trend["customers"]

    # New vs returning split, by first-purchase month.
    first = o.groupby(CUSTOMER_KEY)["order_month"].transform("min")
    o["is_new_customer"] = o["order_month"] == first
    split = o.groupby("order_month").agg(
        new_customer_orders=("is_new_customer", "sum"),
        new_customer_revenue=("order_revenue", lambda s: s[o.loc[s.index, "is_new_customer"]].sum()),
    ).reset_index()

    trend = trend.merge(split, on="order_month", how="left")
    trend["returning_orders"] = trend["orders"] - trend["new_customer_orders"]
    trend["returning_revenue"] = trend["revenue"] - trend["new_customer_revenue"]
    return trend


def category_breakdown(transactions: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Revenue by product category, for segment drill-downs."""
    cat = (
        transactions.groupby("product_category")
        .agg(
            revenue=("item_revenue", "sum"),
            items=("order_item_id", "size"),
            orders=("order_id", "nunique"),
            avg_price=("price", "mean"),
            avg_review=("review_score", "mean"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    cat["pct_revenue"] = 100.0 * cat["revenue"] / cat["revenue"].sum()
    return cat.head(top_n).reset_index(drop=True)


def state_breakdown(orders: pd.DataFrame) -> pd.DataFrame:
    """Revenue and customers by Brazilian state."""
    st = (
        orders.groupby("customer_state")
        .agg(
            revenue=("order_revenue", "sum"),
            orders=("order_id", "size"),
            customers=(CUSTOMER_KEY, "nunique"),
            aov=("order_revenue", "mean"),
            avg_review=("review_score", "mean"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    st["pct_revenue"] = 100.0 * st["revenue"] / st["revenue"].sum()
    return st.reset_index(drop=True)
