"""Cohort retention: group customers by first-purchase month, track them forward.

A caveat that matters for reading the output: Olist's repeat rate is ~2%, so
classic month-by-month retention is near-zero everywhere and a heatmap of it
conveys almost nothing. `cumulative_repeat_matrix` is the more informative view
for a business like this — it asks "what share of this cohort has EVER come
back by month N", which actually accumulates signal instead of flatlining.

Both are computed. The dashboard shows both, because the contrast is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retainiq.config import CUSTOMER_KEY


def _month_index(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Whole months between two month-start timestamps."""
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


def build_cohort_table(orders: pd.DataFrame) -> pd.DataFrame:
    """Attach each order's cohort month and its offset from that cohort."""
    o = orders.copy()
    first = o.groupby(CUSTOMER_KEY)["order_purchase_timestamp"].transform("min")
    o["cohort_month"] = first.dt.to_period("M").dt.to_timestamp()
    o["month_index"] = _month_index(o["order_month"], o["cohort_month"])
    return o


def retention_matrix(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Classic retention: % of cohort placing an order in month N.

    Returns (percentage matrix, cohort sizes). Cells beyond the observable
    window are NaN rather than 0 — a cohort that hasn't had 12 months yet must
    not be read as 0% retention at month 12.
    """
    o = build_cohort_table(orders)
    sizes = o.groupby("cohort_month")[CUSTOMER_KEY].nunique()

    counts = (
        o.groupby(["cohort_month", "month_index"])[CUSTOMER_KEY]
        .nunique()
        .unstack("month_index")
    )
    pct = counts.div(sizes, axis=0) * 100.0

    # Mask the unobservable triangle.
    last_month = o["order_month"].max()
    for cohort in pct.index:
        observable = _month_index(
            pd.Series([last_month]), pd.Series([cohort])
        ).iloc[0]
        pct.loc[cohort, [c for c in pct.columns if c > observable]] = np.nan

    return pct, sizes


def cumulative_repeat_matrix(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """% of cohort that has made a repeat purchase by month N (cumulative).

    The honest retention metric for a low-frequency marketplace: it answers
    "has this customer ever come back yet", which is the question the business
    actually cares about.
    """
    o = build_cohort_table(orders)
    sizes = o.groupby("cohort_month")[CUSTOMER_KEY].nunique()

    # A customer's repeat is their first order at month_index > 0.
    repeats = o[o["month_index"] > 0]
    first_repeat = repeats.groupby([CUSTOMER_KEY]).agg(
        cohort_month=("cohort_month", "first"), month_index=("month_index", "min")
    )

    counts = (
        first_repeat.groupby(["cohort_month", "month_index"])
        .size()
        .unstack("month_index")
        .reindex(index=sizes.index)
        .fillna(0)
    )
    # Cumulative across month_index, then as % of cohort.
    cum = counts.cumsum(axis=1).div(sizes, axis=0) * 100.0

    last_month = o["order_month"].max()
    for cohort in cum.index:
        observable = _month_index(
            pd.Series([last_month]), pd.Series([cohort])
        ).iloc[0]
        cum.loc[cohort, [c for c in cum.columns if c > observable]] = np.nan

    return cum, sizes


def cohort_revenue_matrix(orders: pd.DataFrame) -> pd.DataFrame:
    """Cumulative revenue per customer by cohort age — the CLV curve by cohort."""
    o = build_cohort_table(orders)
    sizes = o.groupby("cohort_month")[CUSTOMER_KEY].nunique()

    rev = (
        o.groupby(["cohort_month", "month_index"])["order_revenue"]
        .sum()
        .unstack("month_index")
        .fillna(0)
    )
    cum = rev.cumsum(axis=1).div(sizes, axis=0)

    last_month = o["order_month"].max()
    for cohort in cum.index:
        observable = _month_index(
            pd.Series([last_month]), pd.Series([cohort])
        ).iloc[0]
        cum.loc[cohort, [c for c in cum.columns if c > observable]] = np.nan

    return cum


def cohort_summary(orders: pd.DataFrame) -> pd.DataFrame:
    """One row per cohort: size, revenue, and repeat rate at fixed horizons."""
    o = build_cohort_table(orders)
    sizes = o.groupby("cohort_month")[CUSTOMER_KEY].nunique()
    cum, _ = cumulative_repeat_matrix(o)

    rows = []
    for cohort in sizes.index:
        sub = o[o["cohort_month"] == cohort]
        row = {
            "cohort_month": cohort,
            "customers": int(sizes[cohort]),
            "revenue": float(sub["order_revenue"].sum()),
            "revenue_per_customer": float(sub["order_revenue"].sum() / sizes[cohort]),
        }
        for h in (3, 6, 12):
            row[f"repeat_by_m{h}"] = (
                float(cum.loc[cohort, h]) if h in cum.columns and pd.notna(cum.loc[cohort, h]) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
