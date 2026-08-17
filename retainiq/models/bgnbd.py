"""BG/NBD + Gamma-Gamma probabilistic CLV.

The textbook approach: BG/NBD models *how many* purchases a customer will make
(a Pareto-style buy-till-you-die process), Gamma-Gamma models *how much* each
purchase is worth, and CLV is the product.

It is included here mainly so its failure on this data can be demonstrated
rather than asserted. Two structural problems:

  * BG/NBD learns the repeat-purchase process from customers who have repeated.
    Only 1,493 of 75,132 calibration customers (1.99%) have frequency > 0, so
    the likelihood is dominated by 73,639 customers contributing only "never
    came back" information.
  * Gamma-Gamma trains exclusively on frequency > 0 customers, and its core
    assumption -- that spend per transaction is independent of frequency -- is
    checked in `gamma_gamma_assumption_check` rather than assumed.

`simulate_identifiability` tests the first concern by recovering known
parameters from synthetic data at descending repeat rates. Its verdict is more
interesting than expected and worth recording honestly: recovery DEGRADES but
does not collapse (mean parameter error 6.6% at a 45% repeat rate, 14.1% at
2%). So BG/NBD is not hopeless at this repeat rate *when the data genuinely
follows a BG/NBD process*.

That reframes the Olist failure. It is not primarily an identifiability
problem -- it is (a) Gamma-Gamma's q < 1 breakdown, which is structural, and
(b) misspecification: Olist's customers are not mostly-alive buyers with low
purchase rates, they are genuinely one-time buyers. The fitted b = 0.017 is
the model correctly reporting near-immediate dropout for everyone.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from retainiq.config import CUSTOMER_KEY

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class BGNBDResult:
    bgf: object
    ggf: object | None
    summary: pd.DataFrame          # per-customer calibration RFM in lifetimes format
    predictions: pd.Series         # expected 90-day spend, Gamma-Gamma as fitted
    predictions_empirical: pd.Series  # same purchase counts x observed AOV
    expected_purchases: pd.Series
    expected_value: pd.Series
    params: dict
    gg_fitted: bool
    gg_valid: bool
    gg_note: str
    diagnostics: dict


def gamma_gamma_validity(params: dict) -> tuple[bool, str]:
    """Gamma-Gamma's expected value is p*v/(q-1): finite and positive only if q > 1.

    When it is fitted on too few repeat customers, q drifts below 1 and the
    model starts emitting NEGATIVE expected spend. That is a hard structural
    failure, not a tuning problem, so it is checked explicitly rather than
    discovered downstream as a strange-looking number.
    """
    q = params.get("gg_q")
    if q is None:
        return False, "Gamma-Gamma not fitted"
    if q <= 1:
        return False, (
            f"q = {q:.4f} <= 1 -> E[spend] = p*v/(q-1) is negative. "
            f"The fitted Gamma-Gamma is structurally invalid."
        )
    return True, f"q = {q:.4f} > 1 -> expected value is well defined."


def rebuild_bgf(params: dict):
    """Reconstruct a BetaGeoFitter from saved parameters.

    lifetimes' fitters close over a lambda during fit() and therefore cannot be
    pickled. Since prediction is a closed-form function of (r, alpha, a, b), we
    persist the parameters and rebuild instead.
    """
    from lifetimes import BetaGeoFitter

    f = BetaGeoFitter(penalizer_coef=0.0)
    f.params_ = pd.Series({k: params[k] for k in ("r", "alpha", "a", "b")})
    return f


def rebuild_ggf(params: dict):
    """Reconstruct a GammaGammaFitter from saved parameters."""
    from lifetimes import GammaGammaFitter

    keys = ("gg_p", "gg_q", "gg_v")
    if not all(k in params for k in keys):
        return None
    f = GammaGammaFitter(penalizer_coef=0.0)
    f.params_ = pd.Series({k[3:]: params[k] for k in keys})
    return f


def build_lifetimes_summary(
    orders: pd.DataFrame, origin: pd.Timestamp
) -> pd.DataFrame:
    """Calibration RFM in the (frequency, recency, T, monetary_value) format.

    lifetimes' conventions, which differ from the Phase 2 RFM table:
      frequency = number of REPEAT purchase occasions (so 0 for one-timers)
      recency   = age of the customer *at their last purchase*, in days
      T         = age of the customer at `origin`, in days
      monetary  = mean value of REPEAT transactions
    """
    past = orders[orders["order_purchase_timestamp"] < origin].copy()
    past["purchase_day"] = past["order_purchase_timestamp"].dt.normalize()

    # Collapse same-day basket splits into one occasion, consistent with Phase 2.
    occ = (
        past.groupby([CUSTOMER_KEY, "purchase_day"], as_index=False)["order_revenue"]
        .sum()
        .sort_values([CUSTOMER_KEY, "purchase_day"])
    )

    g = occ.groupby(CUSTOMER_KEY)
    s = pd.DataFrame(
        {
            "frequency": g["purchase_day"].size() - 1,
            "first": g["purchase_day"].min(),
            "last": g["purchase_day"].max(),
        }
    )
    s["recency"] = (s["last"] - s["first"]).dt.total_seconds() / 86400
    s["T"] = (origin - s["first"]).dt.total_seconds() / 86400

    # Gamma-Gamma uses the mean value of REPEAT transactions only.
    repeat_val = (
        occ[occ.groupby(CUSTOMER_KEY).cumcount() > 0]
        .groupby(CUSTOMER_KEY)["order_revenue"]
        .mean()
    )
    s["monetary_value"] = repeat_val.reindex(s.index).fillna(0.0)

    return s[["frequency", "recency", "T", "monetary_value"]]


def gamma_gamma_assumption_check(summary: pd.DataFrame) -> tuple[float, str]:
    """Gamma-Gamma assumes frequency and monetary value are independent."""
    sub = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
    if len(sub) < 30:
        return np.nan, f"only {len(sub)} usable customers — cannot test"
    r = float(sub["frequency"].corr(sub["monetary_value"]))
    verdict = "OK" if abs(r) < 0.1 else "VIOLATED" if abs(r) > 0.3 else "marginal"
    return r, f"corr(frequency, monetary) = {r:+.4f} on n={len(sub):,} -> {verdict}"


def fit(
    orders: pd.DataFrame,
    origin: pd.Timestamp,
    horizon_days: int = 90,
    penalizer: float = 0.01,
) -> BGNBDResult:
    from lifetimes import BetaGeoFitter, GammaGammaFitter

    summary = build_lifetimes_summary(orders, origin)

    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    exp_purch = bgf.conditional_expected_number_of_purchases_up_to_time(
        horizon_days, summary["frequency"], summary["recency"], summary["T"]
    )
    exp_purch = pd.Series(np.asarray(exp_purch), index=summary.index)

    # Gamma-Gamma trains only on repeat customers with positive spend.
    gg_mask = (summary["frequency"] > 0) & (summary["monetary_value"] > 0)
    gg_data = summary[gg_mask]

    ggf = None
    gg_fitted = False
    _, gg_note = gamma_gamma_assumption_check(summary)

    if len(gg_data) >= 50:
        try:
            ggf = GammaGammaFitter(penalizer_coef=penalizer)
            ggf.fit(gg_data["frequency"], gg_data["monetary_value"])
            exp_value = pd.Series(
                np.asarray(
                    ggf.conditional_expected_average_profit(
                        summary["frequency"], summary["monetary_value"]
                    )
                ),
                index=summary.index,
            )
            gg_fitted = True
        except Exception as e:  # pragma: no cover - convergence failure path
            gg_note += f" | Gamma-Gamma fit failed: {e}"
            exp_value = pd.Series(
                gg_data["monetary_value"].mean(), index=summary.index
            )
    else:
        gg_note += f" | too few repeat customers ({len(gg_data)}) to fit Gamma-Gamma"
        exp_value = pd.Series(gg_data["monetary_value"].mean(), index=summary.index)

    predictions = exp_purch * exp_value

    params = dict(bgf.params_)
    if ggf is not None:
        params.update({f"gg_{k}": v for k, v in ggf.params_.items()})

    gg_valid, validity_note = gamma_gamma_validity(params)

    # Salvage variant: keep BG/NBD's purchase-count model (which is well
    # behaved) but replace the broken Gamma-Gamma value model with each
    # customer's own observed average order value. Comparing the two isolates
    # WHICH half of the probabilistic approach fails on this data.
    observed_aov = _observed_aov(orders, origin)
    fallback_value = observed_aov.reindex(summary.index)
    fallback_value = fallback_value.fillna(fallback_value.mean())
    predictions_empirical = exp_purch * fallback_value

    diagnostics = {
        "n_customers": int(len(summary)),
        "n_frequency_gt0": int((summary["frequency"] > 0).sum()),
        "pct_frequency_gt0": float(100.0 * (summary["frequency"] > 0).mean()),
        "expected_purchases_min": float(exp_purch.min()),
        "expected_purchases_max": float(exp_purch.max()),
        "expected_value_min": float(exp_value.min()),
        "n_negative_expected_value": int((exp_value < 0).sum()),
        "n_negative_predictions": int((predictions < 0).sum()),
        "gg_validity_note": validity_note,
    }

    return BGNBDResult(
        bgf=bgf,
        ggf=ggf,
        summary=summary,
        predictions=predictions,
        predictions_empirical=predictions_empirical,
        expected_purchases=exp_purch,
        expected_value=exp_value,
        params=params,
        gg_fitted=gg_fitted,
        gg_valid=gg_valid,
        gg_note=gg_note,
        diagnostics=diagnostics,
    )


def _observed_aov(orders: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    """Mean order value per customer, from pre-origin orders only."""
    past = orders[orders["order_purchase_timestamp"] < origin]
    return past.groupby(CUSTOMER_KEY)["order_revenue"].mean()


def simulate_identifiability(
    repeat_rates: tuple[float, ...] = (0.45, 0.30, 0.20, 0.10, 0.05, 0.02),
    size: int = 20_000,
    n_reps: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Recover known BG/NBD parameters from synthetic data at varying repeat rates.

    This is the evidence that BG/NBD is not merely *inaccurate* on Olist but
    structurally unidentifiable there. We generate data from known (r, alpha,
    a, b), refit, and measure the recovery error as the repeat rate falls.

    Repeat rate is steered by shrinking r (the purchase-rate shape parameter);
    each row reports the repeat rate actually realised.
    """
    from lifetimes import BetaGeoFitter
    from lifetimes.generate_data import beta_geometric_nbd_model

    rows = []

    for target_rate in repeat_rates:
        # Search r to hit the target repeat rate; alpha/a/b held fixed.
        r_true = _solve_r_for_rate(target_rate)
        true = dict(r=r_true, alpha=4.41, a=0.79, b=2.43)

        for rep in range(n_reps):
            # beta_geometric_nbd_model has no seed kwarg; seed globally so the
            # study is reproducible.
            np.random.seed(seed + rep * 101 + int(target_rate * 1000))
            df = beta_geometric_nbd_model(T=540, size=size, **true)
            realised = float((df["frequency"] > 0).mean())
            try:
                f = BetaGeoFitter(penalizer_coef=0.0)
                f.fit(df["frequency"], df["recency"], df["T"])
                fitted = dict(f.params_)
                err = {
                    f"{k}_err_pct": 100.0 * abs(fitted[k] - true[k]) / true[k]
                    for k in true
                }
                rows.append(
                    {
                        "target_repeat_rate": 100 * target_rate,
                        "realised_repeat_rate": 100 * realised,
                        "rep": rep,
                        **{f"{k}_true": v for k, v in true.items()},
                        **{f"{k}_fit": fitted[k] for k in true},
                        **err,
                        "mean_abs_err_pct": float(np.mean(list(err.values()))),
                        "converged": True,
                    }
                )
            except Exception:
                rows.append(
                    {
                        "target_repeat_rate": 100 * target_rate,
                        "realised_repeat_rate": 100 * realised,
                        "rep": rep,
                        "mean_abs_err_pct": np.nan,
                        "converged": False,
                    }
                )

    df = pd.DataFrame(rows)
    return (
        df.groupby("target_repeat_rate")
        .agg(
            realised_repeat_rate=("realised_repeat_rate", "mean"),
            mean_abs_err_pct=("mean_abs_err_pct", "mean"),
            worst_err_pct=("mean_abs_err_pct", "max"),
            converged=("converged", "sum"),
            n=("rep", "size"),
        )
        .reset_index()
    )


def _solve_r_for_rate(target: float) -> float:
    """Crude bisection on r to hit a target repeat rate at T=540, alpha=4.41."""
    from lifetimes.generate_data import beta_geometric_nbd_model

    lo, hi = 0.001, 2.0
    for _ in range(14):
        mid = (lo + hi) / 2
        np.random.seed(7)
        d = beta_geometric_nbd_model(T=540, size=4000, r=mid, alpha=4.41, a=0.79, b=2.43)
        rate = float((d["frequency"] > 0).mean())
        if rate < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
