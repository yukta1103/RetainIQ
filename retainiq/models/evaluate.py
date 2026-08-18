"""Evaluation for a 99.4%-zero target, with uncertainty reported everywhere.

Two things drive the design.

First, MAE and RMSE are near-meaningless here in isolation: predicting zero for
every customer achieves a better MAE than any real model. They are reported
because they are conventional, but `baseline_comparison` prints the trivial
predictors alongside so the number cannot be read as success.

Second -- and this is what an earlier version got wrong -- there are only 418
positive examples in the holdout. Every metric computed on that base carries
serious sampling error, and a bare point estimate quoted to four decimals
implies precision that does not exist. So every headline metric here ships with
a bootstrap confidence interval, and calibration is binned into QUINTILES (~84
positives each) rather than deciles (~41 each, binomial sd ~6.5, where the
non-monotonic middle was pure noise being read as structure).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

N_BOOT = 1000
TOP_FRACTION = 0.10


@dataclass(frozen=True)
class CI:
    """A point estimate with a bootstrap percentile interval."""
    point: float
    lo: float
    hi: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def fmt(self, dp: int = 4, pct: bool = False) -> str:
        s = "%" if pct else ""
        return f"{self.point:.{dp}f}{s} [{self.lo:.{dp}f}, {self.hi:.{dp}f}]"

    def beats(self, null_value: float) -> bool:
        """True only if the whole interval sits ABOVE the null."""
        return self.lo > null_value

    def loses_to(self, null_value: float) -> bool:
        """True only if the whole interval sits BELOW the null."""
        return self.hi < null_value

    def verdict_vs(self, null_value: float) -> str:
        if self.beats(null_value):
            return "yes"
        if self.loses_to(null_value):
            return "WORSE"
        return "no"


@dataclass
class Evaluation:
    name: str
    mae: float
    rmse: float
    auc: CI
    spearman: CI
    top_capture: CI          # % of actual revenue in the top-scoring 10%
    predicted_total: float
    actual_total: float
    calibration: pd.DataFrame = field(repr=False)
    n_positives: int = 0

    @property
    def top_lift(self) -> float:
        return self.top_capture.point / (100 * TOP_FRACTION)

    def summary_row(self) -> dict:
        return {
            "model": self.name,
            "MAE": self.mae,
            "RMSE": self.rmse,
            "AUC": self.auc.point,
            "AUC_lo": self.auc.lo,
            "AUC_hi": self.auc.hi,
            "Spearman": self.spearman.point,
            "Spearman_lo": self.spearman.lo,
            "Spearman_hi": self.spearman.hi,
            "top10%_capture": self.top_capture.point,
            "top10%_lo": self.top_capture.lo,
            "top10%_hi": self.top_capture.hi,
            "top10%_lift": self.top_lift,
            "beats_random": self.auc.beats(0.5),
            "verdict_vs_random": self.auc.verdict_vs(0.5),
        }


def _top_capture(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    k = max(int(len(y_true) * TOP_FRACTION), 1)
    top = np.argpartition(-y_pred, k - 1)[:k]
    return 100.0 * y_true[top].sum() / max(y_true.sum(), 1e-9)


def _bootstrap(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = N_BOOT, seed: int = 0
) -> dict[str, CI]:
    """Resample customers with replacement; recompute every ranking metric.

    One shared set of resamples is used for all three metrics so their
    intervals are mutually consistent.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = np.arange(n)
    aucs, spears, caps = [], [], []

    for _ in range(n_boot):
        s = rng.choice(idx, n, replace=True)
        yt, yp = y_true[s], y_pred[s]
        pos = yt > 0
        if pos.sum() == 0 or pos.all():
            continue
        aucs.append(roc_auc_score(pos.astype(int), yp))
        spears.append(stats.spearmanr(yt, yp).statistic)
        caps.append(_top_capture(yt, yp))

    def ci(samples: list[float], point: float) -> CI:
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return CI(float(point), float(lo), float(hi))

    return {
        "auc": ci(aucs, roc_auc_score((y_true > 0).astype(int), y_pred)),
        "spearman": ci(spears, stats.spearmanr(y_true, y_pred).statistic),
        "top_capture": ci(caps, _top_capture(y_true, y_pred)),
    }


def quantile_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 5,
    n_boot: int = 400,
    seed: int = 0,
) -> pd.DataFrame:
    """Predicted vs actual by predicted-value bin, with bootstrap bands.

    Quintiles by default. With 418 positives, deciles put ~41 in each bin
    (binomial sd ~6.5) and the resulting wobble is noise; quintiles give ~84
    and the bands make the remaining uncertainty explicit rather than implied.

    Ties are pervasive, so bins are cut on RANK, not on predicted value --
    qcut on the raw predictions would collapse into fewer than n_bins.
    """
    labels = [f"Q{i}" for i in range(1, n_bins + 1)]
    order = pd.Series(y_pred).rank(method="first", ascending=False)
    binned = pd.qcut(order, n_bins, labels=labels)

    df = pd.DataFrame({"y": y_true, "p": y_pred, "bin": binned})
    out = (
        df.groupby("bin", observed=True)
        .agg(
            customers=("y", "size"),
            mean_predicted=("p", "mean"),
            mean_actual=("y", "mean"),
            total_actual=("y", "sum"),
            n_returned=("y", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    out["return_rate"] = 100.0 * out["n_returned"] / out["customers"]
    out["pct_of_actual_revenue"] = (
        100.0 * out["total_actual"] / max(out["total_actual"].sum(), 1e-9)
    )
    out["calibration_ratio"] = out["mean_predicted"] / out["mean_actual"].replace(0, np.nan)

    # Bootstrap within each bin: how much would mean_actual and the return
    # rate move if we had drawn a different sample of these customers?
    rng = np.random.default_rng(seed)
    lo_a, hi_a, lo_r, hi_r = [], [], [], []
    for label in out["bin"]:
        vals = df.loc[df["bin"] == label, "y"].to_numpy()
        means, rates = [], []
        for _ in range(n_boot):
            s = rng.choice(len(vals), len(vals), replace=True)
            means.append(vals[s].mean())
            rates.append(100.0 * (vals[s] > 0).mean())
        lo_a.append(float(np.percentile(means, 2.5)))
        hi_a.append(float(np.percentile(means, 97.5)))
        lo_r.append(float(np.percentile(rates, 2.5)))
        hi_r.append(float(np.percentile(rates, 97.5)))

    out["actual_lo"], out["actual_hi"] = lo_a, hi_a
    out["return_rate_lo"], out["return_rate_hi"] = lo_r, hi_r
    return out


def evaluate(
    name: str, y_true: np.ndarray, y_pred: np.ndarray,
    n_boot: int = N_BOOT, n_bins: int = 5,
) -> Evaluation:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true
    b = _bootstrap(y_true, y_pred, n_boot=n_boot)

    return Evaluation(
        name=name,
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        auc=b["auc"],
        spearman=b["spearman"],
        top_capture=b["top_capture"],
        predicted_total=float(y_pred.sum()),
        actual_total=float(y_true.sum()),
        calibration=quantile_calibration(y_true, y_pred, n_bins=n_bins),
        n_positives=int((y_true > 0).sum()),
    )


def baseline_comparison(y_true: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
    """Trivial predictors, so MAE can be read in context."""
    y_true = np.asarray(y_true, dtype=float)
    rows = []
    for label, pred in [
        ("predict zero", np.zeros_like(y_true)),
        ("predict train mean", np.full_like(y_true, float(np.mean(y_train)))),
    ]:
        err = pred - y_true
        rows.append({
            "baseline": label,
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err**2))),
        })
    return pd.DataFrame(rows)


def render_calibration(cal: pd.DataFrame, title: str) -> str:
    lines = [
        f"  {title}",
        f"  {'bin':<6}{'custs':>8}{'mean pred':>11}{'mean actual':>13}"
        f"{'95% CI on actual':>22}{'returned':>10}{'ret %':>8}{'% of rev':>10}",
        "  " + "-" * 90,
    ]
    for _, r in cal.iterrows():
        band = f"[{r['actual_lo']:.2f}, {r['actual_hi']:.2f}]"
        lines.append(
            f"  {str(r['bin']):<6}{r['customers']:>8,}{r['mean_predicted']:>11.2f}"
            f"{r['mean_actual']:>13.2f}{band:>22}"
            f"{r['n_returned']:>10,}{r['return_rate']:>7.2f}%"
            f"{r['pct_of_actual_revenue']:>9.1f}%"
        )
    return "\n".join(lines)


def render_comparison(evals: list[Evaluation]) -> str:
    lines = [
        f"  {'model':<26}{'MAE':>8}{'AUC [95% CI]':>26}{'top10% capture [95% CI]':>28}"
        f"{'>random?':>10}",
        "  " + "-" * 98,
    ]
    for e in evals:
        auc = f"{e.auc.point:.3f} [{e.auc.lo:.3f}, {e.auc.hi:.3f}]"
        cap = f"{e.top_capture.point:.1f}% [{e.top_capture.lo:.1f}, {e.top_capture.hi:.1f}]"
        lines.append(
            f"  {e.name:<26}{e.mae:>8.3f}{auc:>26}{cap:>28}"
            f"{e.auc.verdict_vs(0.5):>10}"
        )
    lines.append(
        "\n  'yes' = whole AUC interval above 0.500. 'WORSE' = whole interval "
        "BELOW 0.500\n  (significantly worse than random). 'no' = interval "
        "straddles 0.500, i.e. indistinguishable."
    )
    return "\n".join(lines)
