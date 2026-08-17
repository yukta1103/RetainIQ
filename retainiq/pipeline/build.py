"""Stage 3 — join the cleaned tables into analysis-ready outputs.

Two grains are produced, because Phases 2-4 need both:

  transactions.parquet — one row per ORDER LINE ITEM. Keeps product/seller/
                         category detail for category-level analysis.
  orders.parquet       — one row per ORDER, the natural grain for RFM, cohort
                         retention and CLV. Derived from transactions so the
                         two always reconcile.
"""

from __future__ import annotations

import pandas as pd

from retainiq.config import CUSTOMER_KEY
from retainiq.pipeline.clean import CleaningLog


def build_transactions(clean: dict[str, pd.DataFrame], log: CleaningLog) -> pd.DataFrame:
    items = clean["order_items"]
    orders = clean["orders"]
    customers = clean["customers"]
    products = clean["products"]
    sellers = clean["sellers"]
    reviews = clean["order_reviews"]
    payments = clean["order_payments"]

    # Payments are 1:many against orders (card + voucher splits). Collapse to
    # one row per order first, or the join would duplicate line items.
    #
    # We keep payment_total only as a CROSS-CHECK, not as the revenue measure.
    # Revenue = price + freight from order_items, because:
    #   1. it decomposes to product and seller, which payments cannot;
    #   2. voucher-funded orders understate payment_value vs goods delivered;
    #   3. installment rows would otherwise need careful de-duplication.
    # validation/checks.py reports how closely the two agree.
    pay_agg = (
        payments.groupby("order_id")
        .agg(
            payment_total=("payment_value", "sum"),
            n_payment_methods=("payment_sequential", "size"),
            max_installments=("payment_installments", "max"),
            payment_type=("payment_type", lambda s: s.value_counts().index[0]),
        )
        .reset_index()
    )

    # INNER against orders: items whose order was filtered out (not delivered,
    # or outside the window) must not survive into the analysis table.
    tx = log.apply(
        items,
        "transactions",
        "items whose order was filtered out",
        lambda d: d.merge(
            orders[
                [
                    "order_id",
                    "customer_id",
                    "order_status",
                    "order_purchase_timestamp",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                ]
            ],
            on="order_id",
            how="inner",
        ),
    )

    tx = log.apply(
        tx,
        "transactions",
        "orders with no matching customer",
        lambda d: d.merge(
            customers[
                ["customer_id", CUSTOMER_KEY, "customer_city", "customer_state"]
            ],
            on="customer_id",
            how="inner",
        ),
    )

    # LEFT for descriptive dimensions: a missing product row should not delete
    # a real sale, so unmatched products fall back to 'unknown'.
    before = len(tx)
    tx = tx.merge(
        products[["product_id", "product_category"]], on="product_id", how="left"
    )
    tx = tx.merge(
        sellers[["seller_id", "seller_state"]], on="seller_id", how="left"
    )
    tx = tx.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")
    tx = tx.merge(pay_agg, on="order_id", how="left")
    assert len(tx) == before, "dimension join changed the transaction grain"

    tx["product_category"] = tx["product_category"].fillna("unknown")
    tx["item_revenue"] = tx["price"] + tx["freight_value"]

    tx["order_month"] = tx["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    tx["order_date"] = tx["order_purchase_timestamp"].dt.normalize()
    tx["delivery_days"] = (
        tx["order_delivered_customer_date"] - tx["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    tx["delivered_late"] = (
        tx["order_delivered_customer_date"] > tx["order_estimated_delivery_date"]
    )

    cols = [
        "order_id", "order_item_id", CUSTOMER_KEY, "customer_id",
        "order_purchase_timestamp", "order_date", "order_month", "order_status",
        "product_id", "product_category", "seller_id", "seller_state",
        "price", "freight_value", "item_revenue",
        "customer_city", "customer_state",
        "review_score", "payment_type", "max_installments", "n_payment_methods",
        "payment_total", "delivery_days", "delivered_late",
    ]
    return tx[cols].sort_values(["order_purchase_timestamp", "order_id", "order_item_id"]).reset_index(drop=True)


def build_orders(tx: pd.DataFrame) -> pd.DataFrame:
    """Collapse the line-item table to one row per order."""
    grouped = tx.groupby("order_id", as_index=False).agg(
        **{CUSTOMER_KEY: (CUSTOMER_KEY, "first")},
        customer_id=("customer_id", "first"),
        order_purchase_timestamp=("order_purchase_timestamp", "first"),
        order_date=("order_date", "first"),
        order_month=("order_month", "first"),
        n_items=("order_item_id", "size"),
        n_distinct_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        product_revenue=("price", "sum"),
        freight=("freight_value", "sum"),
        order_revenue=("item_revenue", "sum"),
        payment_total=("payment_total", "first"),
        payment_type=("payment_type", "first"),
        max_installments=("max_installments", "first"),
        review_score=("review_score", "first"),
        delivery_days=("delivery_days", "first"),
        delivered_late=("delivered_late", "first"),
        customer_state=("customer_state", "first"),
        customer_city=("customer_city", "first"),
        top_category=("product_category", lambda s: s.value_counts().index[0]),
    )
    return grouped.sort_values("order_purchase_timestamp").reset_index(drop=True)
