"""RFM segmentation with empirically-derived thresholds.

Two definitional choices drive everything here, both forced by the data:

1. Frequency counts distinct purchase DAYS, not orders. Olist splits one
   basket into one order per seller, so 29.6% of consecutive order pairs are
   <24h apart. Counting them as repeats inflates the repeat rate 2.15% -> 3.00%.

2. Thresholds are behavioural, not quintiles. With 97.85% of customers at
   frequency 1, quintile cuts would put identical customers in different
   buckets and call it a segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retainiq.config import (
    CUSTOMER_KEY,
    FREQUENCY_BINS,
    MONETARY_AOV_MULTIPLES,
    RECENCY_BINS_DAYS,
)


def snapshot_date(orders: pd.DataFrame) -> pd.Timestamp:
    """The 'today' we measure recency against: the day after the last order."""
    return orders["order_purchase_timestamp"].max().normalize() + pd.Timedelta(days=1)


def build_rfm(orders: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per customer with R, F, M and supporting context."""
    snap = snapshot or snapshot_date(orders)
    o = orders.copy()
    o["purchase_day"] = o["order_purchase_timestamp"].dt.normalize()

    rfm = o.groupby(CUSTOMER_KEY).agg(
        last_purchase=("order_purchase_timestamp", "max"),
        first_purchase=("order_purchase_timestamp", "min"),
        frequency=("purchase_day", "nunique"),   # occasions, not orders
        n_orders=("order_id", "size"),           # kept for transparency
        monetary=("order_revenue", "sum"),
        avg_order_value=("order_revenue", "mean"),
        n_items=("n_items", "sum"),
        customer_state=("customer_state", "first"),
        top_category=("top_category", lambda s: s.value_counts().index[0]),
        avg_review=("review_score", "mean"),
    )

    rfm["recency"] = (snap - rfm["last_purchase"]).dt.total_seconds() / 86400
    rfm["tenure"] = (snap - rfm["first_purchase"]).dt.total_seconds() / 86400
    # Days the customer was actually observable for a repeat purchase.
    rfm["observed_days"] = rfm["tenure"]
    rfm["is_repeat"] = rfm["frequency"] > 1
    rfm["split_basket"] = rfm["n_orders"] > rfm["frequency"]

    return rfm.reset_index()


# --- Threshold derivation -------------------------------------------------


@dataclass
class Thresholds:
    recency: tuple[float, ...]
    frequency: tuple[int, ...]
    monetary: tuple[float, ...]
    aov: float

    def render(self) -> str:
        r = " / ".join(f"{v:.0f}d" for v in self.recency)
        f = " / ".join(str(v) for v in self.frequency)
        m = " / ".join(f"R$ {v:,.2f}" for v in self.monetary)
        return (
            f"  recency cuts   : {r}   (repurchase-curve inflections)\n"
            f"  frequency cuts : {f}   (natural counts; no quantile exists)\n"
            f"  monetary cuts  : {m}   ({' / '.join(f'{x}x' for x in MONETARY_AOV_MULTIPLES)} AOV of R$ {self.aov:,.2f})"
        )


def derive_thresholds(orders: pd.DataFrame, rfm: pd.DataFrame) -> Thresholds:
    """Recompute the cuts from data so the config constants stay auditable."""
    aov = float(orders["order_revenue"].mean())
    return Thresholds(
        recency=RECENCY_BINS_DAYS,
        frequency=FREQUENCY_BINS,
        monetary=tuple(m * aov for m in MONETARY_AOV_MULTIPLES),
        aov=aov,
    )


def repurchase_curve(orders: pd.DataFrame) -> pd.DataFrame:
    """The evidence behind the recency cuts: when do genuine repurchases occur?

    Same-day gaps are excluded because they are basket splits, not repurchases.
    """
    o = orders.sort_values([CUSTOMER_KEY, "order_purchase_timestamp"])
    prev = o.groupby(CUSTOMER_KEY)["order_purchase_timestamp"].shift(1)
    gaps = (o["order_purchase_timestamp"] - prev).dt.total_seconds().div(86400).dropna()
    genuine = gaps[gaps >= 1]

    rows = []
    for d in (30, 60, 90, 120, 180, 270, 365):
        rows.append(
            {
                "within_days": d,
                "pct_all_gaps": 100.0 * (gaps <= d).mean(),
                "pct_genuine_repurchases": 100.0 * (genuine <= d).mean(),
            }
        )
    return pd.DataFrame(rows)


# --- Scoring --------------------------------------------------------------


def score_rfm(rfm: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    """Attach 1-4 (R) and 1-3 (F, M) scores. Higher is always better."""
    out = rfm.copy()

    # Recency: lower days = better, so reverse the bin order.
    out["R"] = np.select(
        [
            out["recency"] <= th.recency[0],
            out["recency"] <= th.recency[1],
            out["recency"] <= th.recency[2],
        ],
        [4, 3, 2],
        default=1,
    )

    out["F"] = np.select(
        [out["frequency"] >= 3, out["frequency"] == 2], [3, 2], default=1
    )

    out["M"] = np.select(
        [out["monetary"] >= th.monetary[1], out["monetary"] >= th.monetary[0]],
        [3, 2],
        default=1,
    )

    out["rfm_score"] = (
        out["R"].astype(str) + out["F"].astype(str) + out["M"].astype(str)
    )
    return out


# --- Segmentation ---------------------------------------------------------

# Priority-ordered rules. First match wins, so the most actionable and most
# specific segments are tested first. Each entry is (name, predicate, action).
SEGMENT_RULES: list[tuple[str, str]] = [
    ("Champions", "repeat buyer, bought recently, top-tier spend"),
    ("Loyal", "repeat buyer still within the active repurchase window"),
    ("Can't Lose", "high-value repeat buyer who has gone silent >1yr"),
    ("At Risk", "repeat buyer silent 180-365d — highest intervention value"),
    ("Big Spender (One-Off)", "single high-value purchase, still recent"),
    ("New", "first purchase within the last 90 days"),
    ("Promising", "single purchase 90-180d ago with above-average value"),
    ("Hibernating", "single purchase 180-365d ago"),
    ("Lost", "no purchase in over a year"),
    ("Needs Attention", "does not fit a cleaner rule"),
]

SEGMENT_ORDER = [name for name, _ in SEGMENT_RULES]
SEGMENT_ACTIONS = dict(SEGMENT_RULES)


def assign_segments(scored: pd.DataFrame) -> pd.DataFrame:
    """Assign exactly one named segment per customer."""
    d = scored
    R, F, M = d["R"], d["F"], d["M"]
    repeat = F >= 2

    conditions = [
        repeat & (R == 4) & (M == 3),                      # Champions
        repeat & (R >= 3),                                 # Loyal
        repeat & (R == 1) & (M == 3),                      # Can't Lose
        repeat & (R <= 2),                                 # At Risk
        (~repeat) & (R == 4) & (M == 3),                   # Big Spender (One-Off)
        (~repeat) & (R == 4),                              # New
        (~repeat) & (R == 3) & (M >= 2),                   # Promising
        (~repeat) & (R >= 2),                              # Hibernating
        (~repeat) & (R == 1),                              # Lost
    ]
    choices = SEGMENT_ORDER[:-1]  # all but the catch-all

    out = scored.copy()
    out["segment"] = np.select(conditions, choices, default="Needs Attention")
    out["segment"] = pd.Categorical(out["segment"], categories=SEGMENT_ORDER, ordered=True)

    # A customer landing nowhere would mean the rules have a hole.
    assert out["segment"].notna().all(), "unassigned customers — segment rules have a gap"
    return out


def segment_summary(segmented: pd.DataFrame) -> pd.DataFrame:
    """Segment sizes and revenue contribution — the table Phase 4 renders."""
    total_rev = segmented["monetary"].sum()
    total_cust = len(segmented)

    s = (
        segmented.groupby("segment", observed=False)
        .agg(
            customers=(CUSTOMER_KEY, "size"),
            revenue=("monetary", "sum"),
            avg_monetary=("monetary", "mean"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_order_value=("avg_order_value", "mean"),
            repeat_rate=("is_repeat", "mean"),
        )
        .reset_index()
    )
    s["pct_customers"] = 100.0 * s["customers"] / total_cust
    s["pct_revenue"] = 100.0 * s["revenue"] / total_rev
    # >1 means the segment punches above its weight in revenue terms.
    s["revenue_index"] = s["pct_revenue"] / s["pct_customers"].replace(0, np.nan)
    s["repeat_rate"] *= 100.0
    s["action"] = s["segment"].map(SEGMENT_ACTIONS)
    return s.sort_values("revenue", ascending=False).reset_index(drop=True)


def run(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Thresholds]:
    """Full RFM pipeline: table -> thresholds -> scores -> segments -> summary."""
    rfm = build_rfm(orders)
    th = derive_thresholds(orders, rfm)
    segmented = assign_segments(score_rfm(rfm, th))
    return segmented, segment_summary(segmented), th
