"""ETL orchestrator: load -> clean -> build -> validate -> write parquet."""

from __future__ import annotations

import sys
import time

import pandas as pd

from retainiq.config import ORDERS_PARQUET, TRANSACTIONS_PARQUET
from retainiq.pipeline import build, clean, load
from retainiq.validation import checks


def _banner(title: str) -> None:
    print(f"\n\n{'#' * 78}\n#  {title}\n{'#' * 78}")


def run(*, describe_raw: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, checks.ValidationReport]:
    t0 = time.time()

    _banner("STAGE 1 — LOAD RAW TABLES")
    raw = load.load_all()
    if describe_raw:
        load.describe(raw)
    raw_total = sum(len(d) for d in raw.values())
    print(f"\n  Loaded {len(raw)} tables, {raw_total:,} raw rows total.")

    _banner("STAGE 2 — CLEAN (every dropped row logged)")
    log = clean.CleaningLog()
    cleaned = clean.clean_all(raw, log)

    _banner("STAGE 3 — JOIN INTO ANALYSIS TABLES")
    tx = build.build_transactions(cleaned, log)
    orders = build.build_orders(tx)
    print(f"  transactions : {len(tx):,} rows x {tx.shape[1]} cols (order-line grain)")
    print(f"  orders       : {len(orders):,} rows x {orders.shape[1]} cols (order grain)")

    _banner("CLEANING LEDGER")
    print(log.render())

    _banner("STAGE 4 — VALIDATION")
    report = checks.run_all(cleaned, tx, orders)
    print(report.render())

    _banner("STAGE 5 — WRITE PARQUET")
    tx.to_parquet(TRANSACTIONS_PARQUET, index=False)
    orders.to_parquet(ORDERS_PARQUET, index=False)
    for p in (TRANSACTIONS_PARQUET, ORDERS_PARQUET):
        print(f"  wrote {p}  ({p.stat().st_size / 1e6:,.1f} MB)")

    _banner("SUMMARY")
    print(f"  analysis window : {orders['order_purchase_timestamp'].min():%Y-%m-%d} "
          f".. {orders['order_purchase_timestamp'].max():%Y-%m-%d}")
    print(f"  orders          : {len(orders):,}")
    print(f"  line items      : {len(tx):,}")
    print(f"  customers       : {orders['customer_unique_id'].nunique():,}")
    print(f"  revenue         : R$ {orders['order_revenue'].sum():,.2f}")
    print(f"  AOV             : R$ {orders['order_revenue'].mean():,.2f}")
    print(f"  elapsed         : {time.time() - t0:.1f}s")

    return tx, orders, report


def main() -> int:
    _, _, report = run(describe_raw="--quiet" not in sys.argv)
    if report.failed:
        print("\nETL FAILED validation — parquet written but must not be trusted.")
        return 1
    print("\nETL completed and passed validation.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
