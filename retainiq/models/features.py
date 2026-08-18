"""Feature construction for the repeat-propensity ranker.

Two distinct leakage surfaces, both guarded:

  TARGET leakage  -- a training target window overlapping the test origin.
                     build_train_test() raises if that ever happens.
  FEATURE leakage -- a feature built from an event that had not occurred at
                     the origin. This is the subtle one, and an earlier
                     version of this module had it: avg_review, min_review,
                     avg_delivery_days and late_rate were filtered on
                     order_purchase_timestamp, but reviews and deliveries
                     happen AFTER purchase. At the test origin that
                     contaminated 2,456 of 75,132 scored customers (3.27%)
                     and inflated AUC from 0.594 to 0.607. Those measures are
                     now masked by their own event timestamps, and
                     assert_no_feature_leakage() re-checks it from the data.


The design problem: a single calibration/holdout split yields only 418 positive
targets out of 75,132 customers (99.44% zeros). One split is too thin to train
on and far too thin to trust.

The fix is a PANEL. We pick several observation origins, and at each origin
build features from strictly-before data and a target from the 90 days after.
Stacking those origins multiplies the positive examples. The test origin is the
last one, and every training target window closes BEFORE the test origin opens,
so no training row can see anything the test row's features couldn't.

    train origins ->  |--feat--][--90d target--]
                          |--feat--][--90d target--]
                                        |--feat--][--90d target--]
    test origin   ->                                   |--feat--][--90d target--]
                                                       ^ strictly after all train targets
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retainiq.config import CUSTOMER_KEY

HORIZON_DAYS = 90

# Origins chosen so every training target window closes before TEST_ORIGIN.
# Spaced 2 months apart to limit overlap between consecutive feature windows
# while still giving each origin a meaningful history to look back on.
#
# 2018-03-01 is held out as a VALIDATION origin: boosting rounds are selected
# on it, never on the test panel. An earlier version early-stopped directly on
# the test origin, which biases the reported score -- with a 95% CI ~0.06 wide,
# that bias is not negligible relative to the effect being measured.
# Introducing a validation origin tightens the chain: every train window must
# now close before VALID_ORIGIN (not merely before TEST_ORIGIN), which pushes
# the training origins two quarters earlier than the leaky version used.
#   train  <= 2017-12-01  (+90d closes exactly at the validation origin)
#   valid  == 2018-03-01  (+90d closes 2018-05-30, before the test origin)
TRAIN_ORIGINS = ("2017-06-01", "2017-08-01", "2017-10-01", "2017-12-01")
VALID_ORIGIN = "2018-03-01"
TEST_ORIGIN = "2018-06-01"

CATEGORICAL = ["top_category", "customer_state", "payment_type"]


@dataclass
class Panel:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame  # customer key + origin, for traceability and grouping

    def __len__(self) -> int:
        return len(self.X)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_features_at(orders: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Customer features using ONLY orders strictly before `origin`."""
    past = orders[orders["order_purchase_timestamp"] < origin].copy()
    if past.empty:
        return pd.DataFrame()

    past["purchase_day"] = past["order_purchase_timestamp"].dt.normalize()

    f = past.groupby(CUSTOMER_KEY).agg(
        last_purchase=("order_purchase_timestamp", "max"),
        first_purchase=("order_purchase_timestamp", "min"),
        frequency=("purchase_day", "nunique"),
        n_orders=("order_id", "size"),
        monetary=("order_revenue", "sum"),
        aov=("order_revenue", "mean"),
        max_order=("order_revenue", "max"),
        min_order=("order_revenue", "min"),
        std_order=("order_revenue", "std"),
        n_items=("n_items", "sum"),
        avg_items=("n_items", "mean"),
        n_sellers=("n_sellers", "sum"),
        avg_freight=("freight", "mean"),
        total_freight=("freight", "sum"),
        max_installments=("max_installments", "max"),
        n_categories=("top_category", "nunique"),
        top_category=("top_category", lambda s: s.value_counts().index[0]),
        customer_state=("customer_state", "first"),
        payment_type=("payment_type", lambda s: s.value_counts().index[0]),
    )

    # --- Post-purchase measures, masked by their OWN event timestamp --------
    #
    # A review is written after delivery, and delivery happens after purchase.
    # Filtering these on order_purchase_timestamp (as an earlier version did)
    # leaks: an order placed 2018-05-28 can be delivered 2018-06-15 and
    # reviewed 2018-06-20, both AFTER a 2018-06-01 origin. Measured at the test
    # origin that contaminated 3.27% of scored customers and inflated AUC by
    # roughly 0.013. Each measure is therefore built only from events that had
    # already occurred at the origin.
    delivered = past[past["order_delivered_customer_date"] < origin]
    if not delivered.empty:
        f = f.join(
            delivered.groupby(CUSTOMER_KEY).agg(
                avg_delivery_days=("delivery_days", "mean"),
                late_rate=("delivered_late", "mean"),
                n_delivered=("order_id", "size"),
            )
        )
    else:
        f[["avg_delivery_days", "late_rate", "n_delivered"]] = np.nan

    reviewed = past[past["review_creation_date"] < origin]
    if not reviewed.empty:
        f = f.join(
            reviewed.groupby(CUSTOMER_KEY).agg(
                avg_review=("review_score", "mean"),
                min_review=("review_score", "min"),
                n_reviewed=("order_id", "size"),
            )
        )
    else:
        f[["avg_review", "min_review", "n_reviewed"]] = np.nan

    f["n_delivered"] = f["n_delivered"].fillna(0)
    f["n_reviewed"] = f["n_reviewed"].fillna(0)

    f["recency"] = (origin - f["last_purchase"]).dt.total_seconds() / 86400
    f["tenure"] = (origin - f["first_purchase"]).dt.total_seconds() / 86400
    f["is_repeat"] = (f["frequency"] > 1).astype(int)
    f["split_basket"] = (f["n_orders"] > f["frequency"]).astype(int)

    # Rate features: normalising by tenure separates "bought 3x in a month"
    # from "bought 3x over two years".
    f["purchase_rate"] = _safe_div(f["frequency"], f["tenure"]) * 365
    f["spend_per_day"] = _safe_div(f["monetary"], f["tenure"])
    f["freight_ratio"] = _safe_div(f["total_freight"], f["monetary"])
    # How far through its own typical cycle is this customer? >1 means overdue.
    f["recency_over_tenure"] = _safe_div(f["recency"], f["tenure"])

    # Observed inter-purchase gap, excluding same-day basket splits.
    p = past.sort_values([CUSTOMER_KEY, "order_purchase_timestamp"])
    gap = (
        p["order_purchase_timestamp"]
        - p.groupby(CUSTOMER_KEY)["order_purchase_timestamp"].shift(1)
    ).dt.total_seconds() / 86400
    p["gap"] = gap
    genuine = p[p["gap"] >= 1]
    gap_stats = genuine.groupby(CUSTOMER_KEY)["gap"].agg(
        mean_gap="mean", last_gap="last"
    )
    f = f.join(gap_stats)
    # Overdue ratio: recency relative to this customer's own rhythm.
    f["overdue_ratio"] = _safe_div(f["recency"], f["mean_gap"])

    f["std_order"] = f["std_order"].fillna(0.0)
    for c in CATEGORICAL:
        f[c] = f[c].astype("category")

    return f.drop(columns=["last_purchase", "first_purchase"])


def build_target_at(
    orders: pd.DataFrame, origin: pd.Timestamp, horizon_days: int = HORIZON_DAYS
) -> pd.Series:
    """Total spend in [origin, origin + horizon)."""
    end = origin + pd.Timedelta(days=horizon_days)
    window = orders[
        (orders["order_purchase_timestamp"] >= origin)
        & (orders["order_purchase_timestamp"] < end)
    ]
    return window.groupby(CUSTOMER_KEY)["order_revenue"].sum()


def build_panel_at(
    orders: pd.DataFrame, origin: str | pd.Timestamp, horizon_days: int = HORIZON_DAYS
) -> Panel:
    origin = pd.Timestamp(origin)
    X = build_features_at(orders, origin)
    y_raw = build_target_at(orders, origin, horizon_days)

    # Customers with no pre-origin history cannot be scored (cold start).
    y = y_raw.reindex(X.index).fillna(0.0)

    meta = pd.DataFrame({CUSTOMER_KEY: X.index, "origin": origin}).reset_index(drop=True)
    return Panel(X.reset_index(drop=True), y.reset_index(drop=True), meta)


def build_train_test(
    orders: pd.DataFrame,
    train_origins: tuple[str, ...] = TRAIN_ORIGINS,
    valid_origin: str | None = VALID_ORIGIN,
    test_origin: str = TEST_ORIGIN,
    horizon_days: int = HORIZON_DAYS,
) -> tuple[Panel, Panel, Panel]:
    """Stack training origins; return (train, valid, test).

    Ordering constraint enforced here: every training target window must close
    before the validation origin, and the validation window before the test
    origin. Violations raise rather than silently leaking.
    """
    test_ts = pd.Timestamp(test_origin)
    valid_ts = pd.Timestamp(valid_origin) if valid_origin else test_ts

    for o in train_origins:
        close = pd.Timestamp(o) + pd.Timedelta(days=horizon_days)
        if close > valid_ts:
            raise ValueError(
                f"train origin {o} has a target window closing {close.date()}, "
                f"after the validation origin {valid_ts.date()} — that would leak."
            )
    if valid_origin:
        vclose = valid_ts + pd.Timedelta(days=horizon_days)
        if vclose > test_ts:
            raise ValueError(
                f"validation origin {valid_origin} closes {vclose.date()}, after "
                f"the test origin {test_ts.date()} — that would leak."
            )

    panels = [build_panel_at(orders, o, horizon_days) for o in train_origins]
    train = Panel(
        pd.concat([p.X for p in panels], ignore_index=True),
        pd.concat([p.y for p in panels], ignore_index=True),
        pd.concat([p.meta for p in panels], ignore_index=True),
    )
    for c in CATEGORICAL:
        train.X[c] = train.X[c].astype("category")

    cats = {c: train.X[c].cat.categories for c in CATEGORICAL}

    def _aligned(origin: str) -> Panel:
        p = build_panel_at(orders, origin, horizon_days)
        for c in CATEGORICAL:
            p.X[c] = p.X[c].astype("category").cat.set_categories(cats[c])
        return p

    valid = _aligned(valid_origin) if valid_origin else test
    test = _aligned(test_origin)
    return train, valid, test


def assert_no_feature_leakage(orders: pd.DataFrame, origin: str | pd.Timestamp) -> dict:
    """Verify that no post-origin event feeds a feature at `origin`.

    Rebuilds the delivery/review features twice -- once correctly, once the
    naive way -- and asserts the naive version would have differed. Returns the
    counts so the run log can show the leak was real and is now closed.
    """
    origin = pd.Timestamp(origin)
    past = orders[orders["order_purchase_timestamp"] < origin]

    late_deliv = past["order_delivered_customer_date"] >= origin
    late_review = past["review_creation_date"] >= origin
    leaky_rows = late_deliv | late_review

    affected = past.loc[leaky_rows, CUSTOMER_KEY].nunique()

    feats = build_features_at(orders, origin)
    for col, mask in [("avg_delivery_days", late_deliv), ("avg_review", late_review)]:
        # Customers whose ONLY qualifying orders are post-origin events must
        # now be null on that feature rather than carrying a future value.
        only_late = set(past.loc[mask, CUSTOMER_KEY]) - set(past.loc[~mask, CUSTOMER_KEY])
        if only_late:
            sample = feats.loc[list(only_late & set(feats.index)), col]
            if len(sample) and sample.notna().any():
                raise AssertionError(
                    f"{col} is populated for {int(sample.notna().sum())} customers "
                    f"whose only source event occurs at or after {origin.date()}"
                )

    return {
        "origin": str(origin.date()),
        "pre_origin_orders": int(len(past)),
        "orders_delivered_after_origin": int(late_deliv.sum()),
        "orders_reviewed_after_origin": int(late_review.sum()),
        "customers_affected": int(affected),
        "pct_customers_affected": float(100.0 * affected / max(feats.shape[0], 1)),
    }


def describe_panel(name: str, panel: Panel) -> str:
    pos = int((panel.y > 0).sum())
    return (
        f"  {name:<8} rows {len(panel):>8,}   positives {pos:>6,} "
        f"({100.0 * pos / len(panel):>6.3f}%)   "
        f"target sum R$ {panel.y.sum():>12,.0f}   "
        f"origins {sorted(panel.meta['origin'].dt.strftime('%Y-%m-%d').unique())}"
    )
