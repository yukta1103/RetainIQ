"""Does the ranker support a positive-ROI intervention?

A ranking metric answers "can we order customers better than chance". It does
not answer "should anyone act on this", and those come apart badly when the
base rate is 0.556%. This module does the arithmetic that decides it.

The model earns money only on the INCREMENTAL revenue a campaign causes. Most
of the revenue in the top decile would have arrived anyway -- those customers
were going to buy. So:

    net = (revenue_in_targeted_group x incremental_uplift)
          - (customers_targeted x cost_per_contact)

Break-even cost per contact therefore falls out directly, and can be compared
against what channels actually cost. The comparison against targeting by
historical spend matters just as much: if a one-line sort does as well, the
model is not earning its complexity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Indicative Brazilian marketing costs per contact, 2018 (R$).
CHANNEL_COSTS = {
    "Email (own list)": 0.02,
    "Push notification": 0.05,
    "SMS": 0.30,
    "Paid social retargeting": 1.50,
    "Direct mail": 4.00,
}


@dataclass
class TargetingResult:
    fraction: float
    n_targeted: int
    revenue_in_group: float
    revenue_total: float
    n_returners: int

    @property
    def capture_pct(self) -> float:
        return 100.0 * self.revenue_in_group / max(self.revenue_total, 1e-9)

    @property
    def lift(self) -> float:
        return self.capture_pct / (100.0 * self.fraction)

    def breakeven_cost(self, uplift: float) -> float:
        """Cost per contact at which the campaign exactly washes its face."""
        return self.revenue_in_group * uplift / max(self.n_targeted, 1)

    def net(self, uplift: float, cost_per_contact: float) -> float:
        return self.revenue_in_group * uplift - self.n_targeted * cost_per_contact


def target_top(y_true: np.ndarray, y_pred: np.ndarray, fraction: float) -> TargetingResult:
    n = len(y_true)
    k = max(int(n * fraction), 1)
    top = np.argsort(-y_pred, kind="stable")[:k]
    return TargetingResult(
        fraction=fraction,
        n_targeted=k,
        revenue_in_group=float(y_true[top].sum()),
        revenue_total=float(y_true.sum()),
        n_returners=int((y_true[top] > 0).sum()),
    )


def breakeven_table(
    res: TargetingResult, uplifts: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
) -> pd.DataFrame:
    rows = []
    for u in uplifts:
        be = res.breakeven_cost(u)
        rows.append({
            "incremental_uplift": f"{100 * u:.0f}%",
            "incremental_revenue": res.revenue_in_group * u,
            "breakeven_cost_per_contact": be,
            "viable_channels": ", ".join(
                name for name, c in CHANNEL_COSTS.items() if c <= be
            ) or "NONE",
        })
    return pd.DataFrame(rows)


def channel_grid(
    res: TargetingResult, uplifts: tuple[float, ...] = (0.05, 0.10, 0.20)
) -> pd.DataFrame:
    """Net R$ for each (channel, uplift) pair. Negative means value destroyed."""
    rows = []
    for name, cost in CHANNEL_COSTS.items():
        row = {"channel": name, "cost_per_contact": cost,
               "campaign_cost": res.n_targeted * cost}
        for u in uplifts:
            row[f"net @ {100 * u:.0f}% uplift"] = res.net(u, cost)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_strategies(
    y_true: np.ndarray,
    strategies: dict[str, np.ndarray],
    fraction: float = 0.10,
    n_boot: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Revenue captured by each ranking, with bootstrap CIs.

    Includes a random-targeting row as the do-nothing floor. If the model's
    interval overlaps the historical-spend interval, the model is not adding
    decision value over a sort.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = np.arange(n)
    rows = []

    for name, pred in strategies.items():
        point = target_top(y_true, pred, fraction)
        caps = []
        for _ in range(n_boot):
            s = rng.choice(idx, n, replace=True)
            caps.append(target_top(y_true[s], pred[s], fraction).capture_pct)
        lo, hi = np.percentile(caps, [2.5, 97.5])
        rows.append({
            "strategy": name,
            "customers_targeted": point.n_targeted,
            "returners_reached": point.n_returners,
            "revenue_captured": point.revenue_in_group,
            "capture_pct": point.capture_pct,
            "capture_lo": float(lo),
            "capture_hi": float(hi),
            "lift": point.lift,
        })

    rows.append({
        "strategy": "Random targeting",
        "customers_targeted": int(n * fraction),
        "returners_reached": int(round((y_true > 0).sum() * fraction)),
        "revenue_captured": float(y_true.sum() * fraction),
        "capture_pct": 100.0 * fraction,
        "capture_lo": np.nan, "capture_hi": np.nan, "lift": 1.0,
    })
    return pd.DataFrame(rows)


def render(res: TargetingResult, be: pd.DataFrame, grid: pd.DataFrame,
           strategies: pd.DataFrame) -> str:
    L = []
    L.append(f"  Targeting the top {100 * res.fraction:.0f}% by predicted score:")
    L.append(f"    customers targeted     {res.n_targeted:,}")
    L.append(f"    of whom actually returned  {res.n_returners:,} "
             f"({100 * res.n_returners / res.n_targeted:.2f}%)")
    L.append(f"    revenue in that group  R$ {res.revenue_in_group:,.0f} "
             f"({res.capture_pct:.1f}% of R$ {res.revenue_total:,.0f}, "
             f"{res.lift:.2f}x random)")

    L.append("\n  Break-even cost per contact (campaign exactly pays for itself):")
    L.append(f"    {'uplift':<10}{'incremental R$':>18}{'break-even/contact':>22}"
             f"   viable channels")
    L.append("    " + "-" * 86)
    for _, r in be.iterrows():
        L.append(f"    {r['incremental_uplift']:<10}"
                 f"{r['incremental_revenue']:>18,.0f}"
                 f"{r['breakeven_cost_per_contact']:>22,.3f}"
                 f"   {r['viable_channels']}")

    L.append("\n  Net R$ by channel (negative = value destroyed):")
    cols = [c for c in grid.columns if c.startswith("net @")]
    L.append(f"    {'channel':<26}{'cost/contact':>13}{'campaign cost':>15}"
             + "".join(f"{c:>20}" for c in cols))
    L.append("    " + "-" * (26 + 13 + 15 + 20 * len(cols)))
    for _, r in grid.iterrows():
        L.append(f"    {r['channel']:<26}{r['cost_per_contact']:>13,.2f}"
                 f"{r['campaign_cost']:>15,.0f}"
                 + "".join(f"{r[c]:>20,.0f}" for c in cols))

    L.append("\n  Does the model beat a simple sort?")
    L.append(f"    {'strategy':<30}{'reached':>9}{'captured R$':>14}"
             f"{'capture % [95% CI]':>28}")
    L.append("    " + "-" * 82)
    for _, r in strategies.iterrows():
        ci = ("      (exact)" if np.isnan(r["capture_lo"])
              else f"[{r['capture_lo']:.1f}, {r['capture_hi']:.1f}]")
        capture = f"{r['capture_pct']:.1f}% {ci}"
        L.append(f"    {r['strategy']:<30}{int(r['returners_reached']):>9,}"
                 f"{r['revenue_captured']:>14,.0f}{capture:>28}")
    return "\n".join(L)
