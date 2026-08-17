"""Export slim, dashboard-ready tables into data/dashboard/.

Why this exists: the full processed parquets total ~44 MB and live under a
gitignored directory, so a Streamlit Cloud deploy would boot with no data.
Re-running the ETL on the cloud instance is not an option either -- it would
mean a 126 MB download on every cold start inside a 1 GB RAM container.

So we commit a trimmed export instead: same rows, only the columns the app
actually reads, with categoricals and downcast numerics. That lands around
10 MB, which is comfortable in git and instant to load.
"""

from __future__ import annotations

import pandas as pd

from retainiq.analytics import cohorts, rfm
from retainiq.analytics.run_analytics import RFM_PARQUET
from retainiq.config import (
    CUSTOMER_KEY,
    DATA_DIR,
    ORDERS_PARQUET,
    TRANSACTIONS_PARQUET,
)
from retainiq.models.train_clv import PREDICTIONS_PARQUET

DASHBOARD_DIR = DATA_DIR / "dashboard"

ORDER_COLS = [
    "order_id", CUSTOMER_KEY, "order_purchase_timestamp", "order_month",
    "order_revenue", "n_items", "review_score", "delivered_late",
    "delivery_days", "customer_state", "top_category", "payment_type",
]
RFM_COLS = [
    CUSTOMER_KEY, "segment", "R", "F", "M", "recency", "frequency", "monetary",
    "avg_order_value", "n_orders", "is_repeat", "customer_state",
    "top_category", "tenure", "avg_review",
]
CLV_COLS = [
    CUSTOMER_KEY, "actual_90d_spend", "pred_gbm", "pred_bgnbd",
    "pred_bgnbd_empirical", "recency", "frequency", "monetary", "tenure",
    "aov", "customer_state", "top_category",
]

CATEGORICAL_COLS = ["customer_state", "top_category", "payment_type", "segment"]


def _shrink(df: pd.DataFrame) -> pd.DataFrame:
    """Categoricals for repeated strings, float32 for measures."""
    out = df.copy()
    for c in out.columns:
        if c in CATEGORICAL_COLS:
            out[c] = out[c].astype("category")
        elif out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
        elif out[c].dtype == "int64":
            out[c] = pd.to_numeric(out[c], downcast="integer")
    return out


def run() -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting dashboard tables -> {DASHBOARD_DIR}\n")

    orders = pd.read_parquet(ORDERS_PARQUET)
    transactions = pd.read_parquet(TRANSACTIONS_PARQUET)
    segments = pd.read_parquet(RFM_PARQUET)
    clv = pd.read_parquet(PREDICTIONS_PARQUET)

    exports: dict[str, pd.DataFrame] = {}

    exports["orders"] = _shrink(orders[ORDER_COLS])
    exports["segments"] = _shrink(segments[RFM_COLS])
    exports["clv"] = _shrink(clv[[c for c in CLV_COLS if c in clv.columns]])

    # Category revenue at line-item grain, pre-aggregated: the full 110k-row
    # transactions table is only ever used for this breakdown.
    cat = (
        transactions.groupby(["order_month", "product_category"], observed=True)
        .agg(
            revenue=("item_revenue", "sum"),
            items=("order_item_id", "size"),
            orders=("order_id", "nunique"),
            avg_review=("review_score", "mean"),
        )
        .reset_index()
    )
    cat["product_category"] = cat["product_category"].astype("category")
    exports["category_monthly"] = _shrink(cat)

    # Cohort matrices are tiny; ship them precomputed.
    ret, sizes = cohorts.retention_matrix(orders)
    cum, _ = cohorts.cumulative_repeat_matrix(orders)
    rev = cohorts.cohort_revenue_matrix(orders)
    for name, mat in [
        ("cohort_retention", ret),
        ("cohort_cumulative", cum),
        ("cohort_revenue", rev),
    ]:
        m = mat.copy()
        m.columns = [str(c) for c in m.columns]
        exports[name] = m.reset_index()

    exports["cohort_sizes"] = sizes.rename("customers").reset_index()

    # Segment summary and thresholds, so the app never recomputes them.
    _, summary, th = rfm.run(orders)
    exports["segment_summary"] = summary.assign(segment=summary["segment"].astype(str))

    # Mixed-type values would break the parquet write, so keep everything as
    # text and let the app coerce what it needs.
    exports["meta"] = pd.DataFrame(
        [
            {"key": "aov", "value": f"{th.aov:.6f}"},
            {"key": "monetary_cut_low", "value": f"{th.monetary[0]:.6f}"},
            {"key": "monetary_cut_high", "value": f"{th.monetary[1]:.6f}"},
            {"key": "recency_cuts", "value": ",".join(str(c) for c in th.recency)},
            {"key": "snapshot_date", "value": str(rfm.snapshot_date(orders).date())},
            {"key": "total_revenue", "value": f"{orders['order_revenue'].sum():.6f}"},
            {"key": "n_customers", "value": str(orders[CUSTOMER_KEY].nunique())},
            {"key": "n_orders", "value": str(len(orders))},
        ]
    )

    total = 0
    for name, df in exports.items():
        path = DASHBOARD_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        size = path.stat().st_size
        total += size
        print(f"  {name:<22} {len(df):>8,} rows  {size / 1e6:>7.2f} MB")

    print(f"\n  TOTAL {total / 1e6:.2f} MB (committed to git for Streamlit Cloud)")


if __name__ == "__main__":  # pragma: no cover
    run()
