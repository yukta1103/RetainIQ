"""Evaluation for a 99.4%-zero target.

MAE and RMSE are reported because they were asked for and because they are the
conventional CLV metrics — but on this target they are close to meaningless in
isolation: a model that predicts zero for every customer achieves a near-optimal
MAE. `baseline_comparison` makes that explicit rather than letting a flattering
MAE stand unchallenged.

What actually matters commercially is RANKING: if we can only afford to
retarget 10% of the base, does the model put the right 10% at the top? Hence
top-decile revenue capture, Spearman, and AUC on the binary return event.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


@dataclass
class Evaluation:
    name: str
    mae: float
    rmse: float
    spearman: float
    auc: float
    top_decile_capture: float
    top_decile_lift: float
    predicted_total: float
    actual_total: float
    calibration: pd.DataFrame = field(repr=False)

    def summary_row(self) -> dict:
        return {
            "model": self.name,
            "MAE": self.mae,
            "RMSE": self.rmse,
            "Spearman": self.spearman,
            "AUC": self.auc,
            "top10%_capture": self.top_decile_capture,
            "top10%_lift": self.top_decile_lift,
            "pred_total": self.predicted_total,
            "actual_total": self.actual_total,
        }


def decile_calibration(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs actual by predicted-value decile.

    Ties are pervasive (most customers score alike), so we rank with
    `first` and cut on rank rather than on value — qcut on the raw
    predictions would collapse into fewer than 10 bins.
    """
    order = pd.Series(y_pred).rank(method="first", ascending=False)
    decile = pd.qcut(order, n_bins, labels=[f"D{i}" for i in range(1, n_bins + 1)])

    df = pd.DataFrame({"y": y_true, "p": y_pred, "decile": decile})
    out = (
        df.groupby("decile", observed=True)
        .agg(
            customers=("y", "size"),
            mean_predicted=("p", "mean"),
            mean_actual=("y", "mean"),
            total_predicted=("p", "sum"),
            total_actual=("y", "sum"),
            n_returned=("y", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    out["return_rate"] = 100.0 * out["n_returned"] / out["customers"]
    out["pct_of_actual_revenue"] = 100.0 * out["total_actual"] / max(out["total_actual"].sum(), 1e-9)
    # >1 means the decile is over-predicted relative to what happened.
    out["calibration_ratio"] = out["mean_predicted"] / out["mean_actual"].replace(0, np.nan)
    return out


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Evaluation:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    returned = (y_true > 0).astype(int)
    auc = float(roc_auc_score(returned, y_pred)) if returned.sum() > 0 else np.nan
    spearman = float(stats.spearmanr(y_true, y_pred).statistic)

    cal = decile_calibration(y_true, y_pred)
    top = cal.iloc[0]
    capture = float(100.0 * top["total_actual"] / max(y_true.sum(), 1e-9))
    # Lift vs a random 10%: random would capture 10% of revenue.
    lift = capture / 10.0

    return Evaluation(
        name=name,
        mae=mae,
        rmse=rmse,
        spearman=spearman,
        auc=auc,
        top_decile_capture=capture,
        top_decile_lift=lift,
        predicted_total=float(y_pred.sum()),
        actual_total=float(y_true.sum()),
        calibration=cal,
    )


def baseline_comparison(y_true: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
    """Show what trivial predictors score, so MAE can be read in context."""
    y_true = np.asarray(y_true, dtype=float)
    rows = []
    for label, pred in [
        ("predict zero", np.zeros_like(y_true)),
        ("predict train mean", np.full_like(y_true, float(np.mean(y_train)))),
    ]:
        err = pred - y_true
        rows.append(
            {
                "baseline": label,
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err**2))),
            }
        )
    return pd.DataFrame(rows)


def render_calibration(cal: pd.DataFrame, title: str) -> str:
    lines = [
        f"  {title}",
        f"  {'decile':<8}{'custs':>8}{'mean pred':>11}{'mean actual':>13}"
        f"{'ratio':>8}{'returned':>10}{'ret %':>8}{'% of rev':>10}",
        "  " + "-" * 76,
    ]
    for _, r in cal.iterrows():
        ratio = "     -" if pd.isna(r["calibration_ratio"]) else f"{r['calibration_ratio']:>6.2f}"
        lines.append(
            f"  {str(r['decile']):<8}{r['customers']:>8,}{r['mean_predicted']:>11.2f}"
            f"{r['mean_actual']:>13.2f}{ratio:>8}{r['n_returned']:>10,}"
            f"{r['return_rate']:>7.2f}%{r['pct_of_actual_revenue']:>9.1f}%"
        )
    return "\n".join(lines)


def render_comparison(evals: list[Evaluation]) -> str:
    df = pd.DataFrame([e.summary_row() for e in evals])
    lines = [
        f"  {'model':<26}{'MAE':>9}{'RMSE':>10}{'Spearman':>10}{'AUC':>8}"
        f"{'top10% cap':>12}{'lift':>7}",
        "  " + "-" * 82,
    ]
    for _, r in df.iterrows():
        lines.append(
            f"  {r['model']:<26}{r['MAE']:>9.3f}{r['RMSE']:>10.2f}"
            f"{r['Spearman']:>10.4f}{r['AUC']:>8.4f}"
            f"{r['top10%_capture']:>11.1f}%{r['top10%_lift']:>7.2f}x"
        )
    return "\n".join(lines)
