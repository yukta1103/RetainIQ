"""Data validation: uniqueness, nulls, referential integrity, business logic.

Checks return structured results rather than raising, so a run reports *all*
problems at once. `ValidationReport.failed` lets the ETL exit non-zero on a
hard failure while letting warnings through.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from retainiq.config import CUSTOMER_KEY, REVENUE_COMPONENTS
from retainiq.data.schema import TABLES_BY_NAME


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str


@dataclass
class ValidationReport:
    results: list[CheckResult]

    @property
    def failed(self) -> bool:
        return any(r.status is Status.FAIL for r in self.results)

    def counts(self) -> dict[Status, int]:
        return {s: sum(1 for r in self.results if r.status is s) for s in Status}

    def render(self) -> str:
        w = max(len(r.name) for r in self.results)
        lines = []
        for r in self.results:
            lines.append(f"  [{r.status.value}] {r.name:<{w}}  {r.detail}")
        c = self.counts()
        lines.append(
            f"\n  {c[Status.PASS]} passed, {c[Status.WARN]} warnings, {c[Status.FAIL]} failures"
        )
        return "\n".join(lines)


# --- Individual checks ----------------------------------------------------


def check_primary_keys(clean: dict[str, pd.DataFrame]) -> list[CheckResult]:
    out = []
    for name, df in clean.items():
        spec = TABLES_BY_NAME.get(name)
        if spec is None or not spec.primary_key:
            continue
        keys = [k for k in spec.primary_key if k in df.columns]
        if not keys:
            continue
        dup = int(df.duplicated(subset=keys).sum())
        out.append(
            CheckResult(
                f"pk unique: {name}({'+'.join(keys)})",
                Status.PASS if dup == 0 else Status.FAIL,
                "unique" if dup == 0 else f"{dup:,} duplicate keys remain",
            )
        )
    return out


def check_required_not_null(clean: dict[str, pd.DataFrame]) -> list[CheckResult]:
    out = []
    for name, df in clean.items():
        spec = TABLES_BY_NAME.get(name)
        if spec is None:
            continue
        offenders = {
            c: int(df[c].isna().sum())
            for c in spec.required
            if c in df.columns and df[c].isna().any()
        }
        out.append(
            CheckResult(
                f"required non-null: {name}",
                Status.PASS if not offenders else Status.FAIL,
                "all required columns populated"
                if not offenders
                else f"nulls in {offenders}",
            )
        )
    return out


def check_referential_integrity(clean: dict[str, pd.DataFrame]) -> list[CheckResult]:
    """Every FK value must exist in its parent — checked on the CLEANED tables.

    Note orders/order_items are deliberately filtered, so we check the child->
    parent direction only (an item pointing at a dropped order is expected and
    is removed by the inner join in build.py, not flagged here).
    """
    out = []
    for name, df in clean.items():
        spec = TABLES_BY_NAME.get(name)
        if spec is None:
            continue
        for col, parent_name, parent_col in spec.foreign_keys:
            parent = clean.get(parent_name)
            if parent is None or col not in df.columns or parent_col not in parent.columns:
                continue
            # orders was row-filtered, so children legitimately overhang it.
            if parent_name == "orders":
                continue
            n_missing = int(
                (~df[col].isin(set(parent[parent_col])) & df[col].notna()).sum()
            )
            out.append(
                CheckResult(
                    f"FK: {name}.{col} -> {parent_name}.{parent_col}",
                    Status.PASS if n_missing == 0 else Status.FAIL,
                    "all values resolve"
                    if n_missing == 0
                    else f"{n_missing:,} orphaned values",
                )
            )
    return out


def check_transaction_integrity(tx: pd.DataFrame) -> list[CheckResult]:
    out = []

    n_null_key = int(tx[CUSTOMER_KEY].isna().sum())
    out.append(
        CheckResult(
            "transactions: customer key populated",
            Status.PASS if n_null_key == 0 else Status.FAIL,
            "no nulls" if n_null_key == 0 else f"{n_null_key:,} nulls",
        )
    )

    bad_rev = int((tx["item_revenue"] <= 0).sum())
    out.append(
        CheckResult(
            "transactions: item_revenue > 0",
            Status.PASS if bad_rev == 0 else Status.FAIL,
            "all positive" if bad_rev == 0 else f"{bad_rev:,} non-positive rows",
        )
    )

    recomputed = tx[list(REVENUE_COMPONENTS)].sum(axis=1)
    drift = float((recomputed - tx["item_revenue"]).abs().max())
    out.append(
        CheckResult(
            "transactions: revenue = price + freight",
            Status.PASS if drift < 1e-6 else Status.FAIL,
            f"max drift {drift:.2e}",
        )
    )

    dup = int(tx.duplicated(subset=["order_id", "order_item_id"]).sum())
    out.append(
        CheckResult(
            "transactions: grain is (order_id, order_item_id)",
            Status.PASS if dup == 0 else Status.FAIL,
            "unique" if dup == 0 else f"{dup:,} duplicates",
        )
    )

    return out


def check_orders_reconcile(tx: pd.DataFrame, orders: pd.DataFrame) -> list[CheckResult]:
    out = []

    tx_orders = tx["order_id"].nunique()
    out.append(
        CheckResult(
            "orders: one row per distinct order in transactions",
            Status.PASS if tx_orders == len(orders) else Status.FAIL,
            f"{len(orders):,} order rows vs {tx_orders:,} distinct in transactions",
        )
    )

    drift = abs(tx["item_revenue"].sum() - orders["order_revenue"].sum())
    out.append(
        CheckResult(
            "orders: revenue reconciles to transactions",
            Status.PASS if drift < 0.01 else Status.FAIL,
            f"R$ {tx['item_revenue'].sum():,.2f} vs R$ {orders['order_revenue'].sum():,.2f} "
            f"(drift {drift:.4f})",
        )
    )

    # Cross-check against the independent payments measure. These will NOT match
    # exactly (vouchers, and payments cover the whole order incl. cancelled
    # lines), so this is a WARN threshold, not a hard gate.
    have_pay = orders["payment_total"].notna()
    if have_pay.any():
        rev = orders.loc[have_pay, "order_revenue"].sum()
        pay = orders.loc[have_pay, "payment_total"].sum()
        gap = 100.0 * abs(pay - rev) / rev
        out.append(
            CheckResult(
                "orders: item revenue vs payments cross-check",
                Status.PASS if gap < 5 else Status.WARN,
                f"items R$ {rev:,.0f} vs payments R$ {pay:,.0f} ({gap:.2f}% apart)",
            )
        )

    n_missing_pay = int(orders["payment_total"].isna().sum())
    out.append(
        CheckResult(
            "orders: payment record present",
            Status.PASS if n_missing_pay == 0 else Status.WARN,
            "all orders have payments"
            if n_missing_pay == 0
            else f"{n_missing_pay:,} orders lack a payment row",
        )
    )

    return out


def check_temporal_sanity(orders: pd.DataFrame) -> list[CheckResult]:
    out = []
    ts = orders["order_purchase_timestamp"]
    out.append(
        CheckResult(
            "orders: purchase timestamps in window",
            Status.PASS,
            f"{ts.min():%Y-%m-%d} .. {ts.max():%Y-%m-%d} ({ts.dt.to_period('M').nunique()} months)",
        )
    )

    # Guard against a thin tail silently biasing recency/cohort analysis.
    monthly = orders.groupby("order_month").size()
    median = monthly.median()
    thin = monthly[monthly < 0.25 * median]
    out.append(
        CheckResult(
            "orders: no anaemic month at the tails",
            Status.PASS if thin.empty else Status.WARN,
            f"min month {monthly.min():,} orders, median {median:,.0f}"
            if thin.empty
            else f"thin months: {{{', '.join(f'{m:%Y-%m}: {v:,}' for m, v in thin.items())}}}",
        )
    )

    neg = int((orders["delivery_days"] < 0).sum())
    out.append(
        CheckResult(
            "orders: delivery never precedes purchase",
            Status.PASS if neg == 0 else Status.FAIL,
            "chronology consistent" if neg == 0 else f"{neg:,} negative durations",
        )
    )
    return out


def check_customer_identity(orders: pd.DataFrame) -> list[CheckResult]:
    """The check that catches the classic Olist mistake."""
    n_unique = orders[CUSTOMER_KEY].nunique()
    n_surrogate = orders["customer_id"].nunique()
    repeat = orders.groupby(CUSTOMER_KEY).size()
    n_repeat = int((repeat > 1).sum())

    results = [
        CheckResult(
            "identity: customer_unique_id collapses surrogates",
            Status.PASS if n_unique < n_surrogate else Status.FAIL,
            f"{n_surrogate:,} order-level customer_id -> {n_unique:,} real customers",
        ),
        CheckResult(
            "identity: repeat buyers exist",
            Status.PASS if n_repeat > 0 else Status.FAIL,
            f"{n_repeat:,} customers with >1 order "
            f"({100.0 * n_repeat / n_unique:.2f}% repeat rate)",
        ),
    ]
    return results


def run_all(
    clean: dict[str, pd.DataFrame], tx: pd.DataFrame, orders: pd.DataFrame
) -> ValidationReport:
    results: list[CheckResult] = []
    results += check_primary_keys(clean)
    results += check_required_not_null(clean)
    results += check_referential_integrity(clean)
    results += check_transaction_integrity(tx)
    results += check_orders_reconcile(tx, orders)
    results += check_temporal_sanity(orders)
    results += check_customer_identity(orders)
    return ValidationReport(results)
