"""Stage 2 — cleaning, with every dropped row accounted for.

The design rule here: no transform silently loses rows. Every filter goes
through `CleaningLog.apply`, which records before/after counts and a reason,
so the end of the run can print a ledger that reconciles raw -> clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from retainiq.config import ANALYSIS_END, ANALYSIS_START, REVENUE_ORDER_STATUSES
from retainiq.data.schema import TABLES_BY_NAME


@dataclass
class DropRecord:
    table: str
    reason: str
    rows_before: int
    rows_after: int

    @property
    def dropped(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def pct(self) -> float:
        return 100.0 * self.dropped / self.rows_before if self.rows_before else 0.0


@dataclass
class CleaningLog:
    records: list[DropRecord] = field(default_factory=list)

    def apply(
        self, df: pd.DataFrame, table: str, reason: str, fn: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> pd.DataFrame:
        before = len(df)
        out = fn(df)
        self.records.append(DropRecord(table, reason, before, len(out)))
        return out

    def note(self, table: str, reason: str, count: int) -> None:
        """Record something that changed values but not row count (e.g. imputation)."""
        self.records.append(DropRecord(table, reason, count, count))

    def render(self) -> str:
        if not self.records:
            return "(no cleaning operations recorded)"
        w = max(len(r.reason) for r in self.records)
        lines = [
            f"{'table':<20} {'reason':<{w}} {'before':>10} {'after':>10} {'dropped':>9} {'%':>7}",
            "-" * (20 + w + 40),
        ]
        for r in self.records:
            marker = "" if r.dropped else "  (no rows lost)"
            lines.append(
                f"{r.table:<20} {r.reason:<{w}} {r.rows_before:>10,} "
                f"{r.rows_after:>10,} {r.dropped:>9,} {r.pct:>6.2f}%{marker}"
            )
        return "\n".join(lines)


# --- Per-table cleaners ---------------------------------------------------


def clean_customers(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = log.apply(
        df, "customers", "exact duplicate rows", lambda d: d.drop_duplicates()
    )
    df = log.apply(
        df,
        "customers",
        "null customer_unique_id",
        lambda d: d.dropna(subset=["customer_id", "customer_unique_id"]),
    )
    return df


def clean_orders(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = log.apply(df, "orders", "exact duplicate rows", lambda d: d.drop_duplicates())
    df = log.apply(
        df,
        "orders",
        "duplicate order_id",
        lambda d: d.drop_duplicates(subset=["order_id"], keep="first"),
    )
    df = log.apply(
        df,
        "orders",
        f"status not in {list(REVENUE_ORDER_STATUSES)}",
        lambda d: d[d["order_status"].isin(REVENUE_ORDER_STATUSES)],
    )
    df = log.apply(
        df,
        "orders",
        "null purchase timestamp",
        lambda d: d.dropna(subset=["order_purchase_timestamp"]),
    )
    df = log.apply(
        df,
        "orders",
        f"outside analysis window {ANALYSIS_START}..{ANALYSIS_END}",
        lambda d: d[
            (d["order_purchase_timestamp"] >= pd.Timestamp(ANALYSIS_START))
            & (d["order_purchase_timestamp"] <= pd.Timestamp(ANALYSIS_END) + pd.Timedelta(days=1))
        ],
    )
    # Impossible chronology would corrupt any delivery-time feature.
    df = log.apply(
        df,
        "orders",
        "delivered before purchased",
        lambda d: d[
            d["order_delivered_customer_date"].isna()
            | (d["order_delivered_customer_date"] >= d["order_purchase_timestamp"])
        ],
    )
    return df


def clean_order_items(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = log.apply(df, "order_items", "exact duplicate rows", lambda d: d.drop_duplicates())
    df = log.apply(
        df,
        "order_items",
        "duplicate (order_id, order_item_id)",
        lambda d: d.drop_duplicates(subset=["order_id", "order_item_id"], keep="first"),
    )
    df = log.apply(
        df,
        "order_items",
        "null price or product_id",
        lambda d: d.dropna(subset=["price", "product_id", "order_id"]),
    )
    # A zero/negative-price line is not a sale; keeping it would deflate AOV.
    df = log.apply(
        df, "order_items", "price <= 0", lambda d: d[d["price"] > 0]
    )
    n_neg_freight = int((df["freight_value"] < 0).sum())
    if n_neg_freight:
        df = df.copy()
        df.loc[df["freight_value"] < 0, "freight_value"] = 0.0
        log.note("order_items", "negative freight clipped to 0", n_neg_freight)
    df["freight_value"] = df["freight_value"].fillna(0.0)
    return df


def clean_order_payments(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = log.apply(df, "order_payments", "exact duplicate rows", lambda d: d.drop_duplicates())
    df = log.apply(
        df,
        "order_payments",
        "duplicate (order_id, payment_sequential)",
        lambda d: d.drop_duplicates(subset=["order_id", "payment_sequential"], keep="first"),
    )
    df = log.apply(
        df, "order_payments", "null payment_value", lambda d: d.dropna(subset=["payment_value"])
    )
    return df


def clean_order_reviews(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = log.apply(df, "order_reviews", "exact duplicate rows", lambda d: d.drop_duplicates())
    df = log.apply(
        df, "order_reviews", "null review_score", lambda d: d.dropna(subset=["review_score", "order_id"])
    )
    # review_id is not unique; an order can also collect more than one review.
    # Collapse to one mean score per order so the join stays 1:1 against orders.
    df = log.apply(
        df,
        "order_reviews",
        "collapsed to mean score per order",
        lambda d: (
            d.groupby("order_id", as_index=False)
            .agg(review_score=("review_score", "mean"), n_reviews=("review_score", "size"))
        ),
    )
    return df


def clean_products(df: pd.DataFrame, log: CleaningLog, translation: pd.DataFrame) -> pd.DataFrame:
    df = log.apply(df, "products", "exact duplicate rows", lambda d: d.drop_duplicates())
    df = log.apply(
        df,
        "products",
        "duplicate product_id",
        lambda d: d.drop_duplicates(subset=["product_id"], keep="first"),
    )

    # Category is descriptive, not a join key — a null category is not a reason
    # to throw away a real sale. Impute to an explicit 'unknown' bucket instead.
    n_missing = int(df["product_category_name"].isna().sum())
    df = df.copy()
    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    if n_missing:
        log.note("products", "null category -> 'unknown'", n_missing)

    df = df.merge(translation, on="product_category_name", how="left")
    n_untranslated = int(df["product_category_name_english"].isna().sum())
    df["product_category"] = df["product_category_name_english"].fillna(
        df["product_category_name"]
    )
    if n_untranslated:
        log.note("products", "no EN translation -> kept PT name", n_untranslated)

    return df.drop(columns=["product_category_name_english"])


def clean_sellers(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = log.apply(df, "sellers", "exact duplicate rows", lambda d: d.drop_duplicates())
    return log.apply(
        df,
        "sellers",
        "duplicate seller_id",
        lambda d: d.drop_duplicates(subset=["seller_id"], keep="first"),
    )


def clean_all(raw: dict[str, pd.DataFrame], log: CleaningLog) -> dict[str, pd.DataFrame]:
    """Run every per-table cleaner. `raw` is left untouched."""
    translation = raw["category_translation"].drop_duplicates(
        subset=["product_category_name"]
    )
    return {
        "customers": clean_customers(raw["customers"], log),
        "orders": clean_orders(raw["orders"], log),
        "order_items": clean_order_items(raw["order_items"], log),
        "order_payments": clean_order_payments(raw["order_payments"], log),
        "order_reviews": clean_order_reviews(raw["order_reviews"], log),
        "products": clean_products(raw["products"], log, translation),
        "sellers": clean_sellers(raw["sellers"], log),
        "category_translation": translation,
    }


__all__ = ["CleaningLog", "DropRecord", "clean_all", "TABLES_BY_NAME"]
