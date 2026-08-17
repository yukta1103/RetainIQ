"""Run Phase 2 analytics and persist the outputs the dashboard will read."""

from __future__ import annotations

import pandas as pd

from retainiq.analytics import cohorts, metrics, rfm
from retainiq.config import ORDERS_PARQUET, PROCESSED_DIR, TRANSACTIONS_PARQUET

RFM_PARQUET = PROCESSED_DIR / "rfm_segments.parquet"
COHORT_PARQUET = PROCESSED_DIR / "cohort_retention.parquet"
COHORT_CUM_PARQUET = PROCESSED_DIR / "cohort_cumulative_repeat.parquet"


def _banner(title: str) -> None:
    print(f"\n\n{'#' * 78}\n#  {title}\n{'#' * 78}")


def run() -> None:
    orders = pd.read_parquet(ORDERS_PARQUET)
    transactions = pd.read_parquet(TRANSACTIONS_PARQUET)

    _banner("HEADLINE METRICS")
    for k, v in metrics.headline_metrics(orders).items():
        if k in ("total_revenue", "aov", "revenue_per_customer"):
            print(f"  {k:<24} R$ {v:>14,.2f}")
        elif k.endswith("_rate"):
            print(f"  {k:<24} {v:>14.2f} %")
        elif k.startswith("n_"):
            print(f"  {k:<24} {v:>14,.0f}")
        else:
            print(f"  {k:<24} {v:>14.2f}")

    _banner("RFM THRESHOLD DERIVATION")
    curve = rfm.repurchase_curve(orders)
    print("  Evidence for the recency cuts — when do repurchases actually happen?")
    print("  (same-day basket splits excluded from the 'genuine' column)\n")
    print("    within   all gaps   genuine repurchases")
    for _, r in curve.iterrows():
        print(
            f"    {int(r['within_days']):>4}d    {r['pct_all_gaps']:>6.1f}%   "
            f"{r['pct_genuine_repurchases']:>16.1f}%"
        )

    segmented, summary, th = rfm.run(orders)
    print("\n  Resulting thresholds:")
    print(th.render())

    _banner("FREQUENCY DEFINITION IMPACT")
    n = len(segmented)
    by_orders = int((segmented["n_orders"] > 1).sum())
    by_occasion = int(segmented["is_repeat"].sum())
    print(f"  customers                                : {n:,}")
    print(f"  'repeat' counting raw orders             : {by_orders:,}  ({100*by_orders/n:.2f}%)")
    print(f"  'repeat' counting purchase occasions     : {by_occasion:,}  ({100*by_occasion/n:.2f}%)")
    print(f"  customers with a split basket (inflation): {int(segmented['split_basket'].sum()):,}")

    _banner("RFM SEGMENTS")
    cols = ["segment", "customers", "pct_customers", "revenue", "pct_revenue",
            "revenue_index", "avg_monetary", "avg_recency", "repeat_rate"]
    disp = summary[cols].copy()
    print(
        f"  {'segment':<24}{'custs':>8}{'cust%':>8}{'revenue':>14}{'rev%':>8}"
        f"{'rev idx':>9}{'avg R$':>10}{'avg rec':>9}{'rep%':>7}"
    )
    print("  " + "-" * 97)
    for _, r in disp.iterrows():
        if r["customers"] == 0:
            # The catch-all being empty is the desired outcome: it proves the
            # rule set covers every customer without a fallback.
            print(f"  {str(r['segment']):<24}{0:>8,}   (empty — rules are exhaustive)")
            continue
        print(
            f"  {str(r['segment']):<24}{r['customers']:>8,}{r['pct_customers']:>7.2f}%"
            f"{r['revenue']:>14,.0f}{r['pct_revenue']:>7.2f}%{r['revenue_index']:>9.2f}"
            f"{r['avg_monetary']:>10.0f}{r['avg_recency']:>9.0f}{r['repeat_rate']:>6.1f}%"
        )
    print(f"\n  {'TOTAL':<24}{summary['customers'].sum():>8,}{100.0:>7.2f}%"
          f"{summary['revenue'].sum():>14,.0f}{100.0:>7.2f}%")

    _banner("COHORT RETENTION (classic: % of cohort ordering in month N)")
    ret, sizes = cohorts.retention_matrix(orders)
    _print_matrix(ret, sizes, max_cols=13, fmt="{:5.1f}")

    _banner("COHORT CUMULATIVE REPEAT (% of cohort that has EVER returned by month N)")
    cum, sizes = cohorts.cumulative_repeat_matrix(orders)
    _print_matrix(cum, sizes, max_cols=13, fmt="{:5.1f}")

    _banner("WRITE ANALYTICS OUTPUTS")
    segmented.to_parquet(RFM_PARQUET, index=False)
    ret.to_parquet(COHORT_PARQUET)
    cum.to_parquet(COHORT_CUM_PARQUET)
    for p in (RFM_PARQUET, COHORT_PARQUET, COHORT_CUM_PARQUET):
        print(f"  wrote {p.name}  ({p.stat().st_size / 1e6:,.2f} MB)")

    top_cat = metrics.category_breakdown(transactions, top_n=5)
    print("\n  top categories by revenue:")
    for _, r in top_cat.iterrows():
        print(f"    {r['product_category']:<32} R$ {r['revenue']:>12,.0f}  ({r['pct_revenue']:>5.2f}%)")


def _print_matrix(mat: pd.DataFrame, sizes: pd.Series, max_cols: int, fmt: str) -> None:
    cols = [c for c in mat.columns if c <= max_cols]
    header = "".join(f"{('m' + str(c)):>7}" for c in cols)
    print(f"  {'cohort':<12}{'size':>8}  {header}")
    print("  " + "-" * (22 + 7 * len(cols)))
    for cohort in mat.index:
        cells = ""
        for c in cols:
            v = mat.loc[cohort, c]
            cells += f"{'':>7}" if pd.isna(v) else f"{fmt.format(v):>7}"
        print(f"  {cohort:%Y-%m}     {int(sizes[cohort]):>8,}  {cells}")


if __name__ == "__main__":  # pragma: no cover
    run()
