"""Phase 3 orchestrator: build panels, fit both models, evaluate, decide.

NAMING: this trains a REPEAT-PROPENSITY RANKER, not a CLV prediction system.
It orders customers by their likelihood of spending again in the next 90 days.
It does not produce trustworthy per-customer spend forecasts -- predicting zero
for everyone beats it on MAE, and Spearman against actual spend is ~0.03. The
module keeps the `clv` filename for continuity with the saved artifact, but
every reported claim is a ranking claim.
"""

from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from retainiq.config import ARTIFACTS_DIR, CUSTOMER_KEY, ORDERS_PARQUET, PROCESSED_DIR
from retainiq.models import bgnbd, evaluate, features, gbm, roi

MODEL_BUNDLE = ARTIFACTS_DIR / "clv_model.joblib"
PREDICTIONS_PARQUET = PROCESSED_DIR / "clv_predictions.parquet"


def _banner(title: str) -> None:
    print(f"\n\n{'#' * 78}\n#  {title}\n{'#' * 78}")


def run(run_simulation: bool = True, n_boot: int = 1000) -> None:
    t0 = time.time()
    orders = pd.read_parquet(ORDERS_PARQUET)

    _banner("STAGE 0 — LEAKAGE GUARD")
    lk = features.assert_no_feature_leakage(orders, features.TEST_ORIGIN)
    print(f"  origin {lk['origin']} — pre-origin orders {lk['pre_origin_orders']:,}")
    print(f"    delivered on/after origin : {lk['orders_delivered_after_origin']:,}")
    print(f"    reviewed  on/after origin : {lk['orders_reviewed_after_origin']:,}")
    print(f"    customers affected        : {lk['customers_affected']:,} "
          f"({lk['pct_customers_affected']:.3f}%)")
    print("  Review/delivery features are masked by their OWN event timestamps,")
    print("  so none of the above reaches the model. assert_no_feature_leakage: PASSED")

    _banner("STAGE 1 — PANEL CONSTRUCTION (train / valid / test, time-ordered)")
    train_panel, valid_panel, test_panel = features.build_train_test(orders)
    print(f"  horizon: {features.HORIZON_DAYS} days")
    print(features.describe_panel("train", train_panel))
    print(features.describe_panel("valid", valid_panel))
    print(features.describe_panel("test", test_panel))
    print(f"\n  Boosting rounds are selected on the VALIDATION origin "
          f"({features.VALID_ORIGIN}), never on the test panel.")
    print(f"  Every train target window closes before {features.VALID_ORIGIN}; "
          f"the validation window closes before {features.TEST_ORIGIN}.")

    _banner("STAGE 2 — MODEL A: BG/NBD + GAMMA-GAMMA")
    origin = pd.Timestamp(features.TEST_ORIGIN)
    bg = bgnbd.fit(orders, origin, horizon_days=features.HORIZON_DAYS)
    d = bg.diagnostics
    print("  fitted parameters:")
    for k, v in bg.params.items():
        print(f"    {k:<10} {v:>12.4f}")
    print(f"\n  likelihood support: {d['n_frequency_gt0']:,} of {d['n_customers']:,} "
          f"customers ({d['pct_frequency_gt0']:.2f}%) have frequency > 0")
    print(f"  expected purchases : {d['expected_purchases_min']:.4f} .. "
          f"{d['expected_purchases_max']:.4f}   (well behaved)")
    print(f"  negative values    : {d['n_negative_expected_value']:,} customers")
    print(f"  validity           : {d['gg_validity_note']}")

    bg_pred = bg.predictions.reindex(test_panel.meta[CUSTOMER_KEY]).fillna(0.0).to_numpy()
    bg_pred_emp = (
        bg.predictions_empirical.reindex(test_panel.meta[CUSTOMER_KEY]).fillna(0.0).to_numpy()
    )

    _banner("STAGE 3 — MODEL B: LIGHTGBM (TWEEDIE)")
    gb = gbm.train(train_panel, valid_panel, test_panel)
    print(f"  objective      : {gb.params['objective']} "
          f"(variance power {gb.params['tweedie_variance_power']})")
    print(f"  best iteration : {gb.best_iteration}  (chosen on validation origin)")
    print("\n  top 10 features by gain:")
    for _, r in gb.feature_importance.head(10).iterrows():
        print(f"    {r['feature']:<24}{r['gain_pct']:>7.2f}%")

    gb_l2 = gbm.train_l2_baseline(train_panel, valid_panel, test_panel)

    _banner("STAGE 4 — EVALUATION (bootstrap CIs, n=%d)" % n_boot)
    y_true = test_panel.y.to_numpy()
    print(f"  holdout positives: {int((y_true > 0).sum()):,} of {len(y_true):,} "
          f"({100 * (y_true > 0).mean():.3f}%)\n")

    print("  Trivial baselines, so MAE can be read in context:")
    for _, r in evaluate.baseline_comparison(y_true, train_panel.y.to_numpy()).iterrows():
        print(f"    {r['baseline']:<22} MAE {r['MAE']:>8.3f}   RMSE {r['RMSE']:>8.2f}")

    evals = [
        evaluate.evaluate("BG/NBD + Gamma-Gamma", y_true, bg_pred, n_boot=n_boot),
        evaluate.evaluate("BG/NBD x observed AOV", y_true, bg_pred_emp, n_boot=n_boot),
        evaluate.evaluate("LightGBM (Tweedie)", y_true, gb.predictions, n_boot=n_boot),
        evaluate.evaluate("LightGBM (L2)", y_true, gb_l2.predictions, n_boot=n_boot),
        evaluate.evaluate("Historical spend (sort)", y_true,
                          test_panel.X["monetary"].to_numpy(), n_boot=n_boot),
    ]
    print()
    print(evaluate.render_comparison(evals))
    print("\n  'MAE' is kept only because it was specified — 'predict zero' wins on it.")
    print("  A model beats random only if its AUC interval excludes 0.500.")

    _banner("STAGE 5 — QUINTILE CALIBRATION (deciles are noise at 418 positives)")
    for e in (evals[2], evals[0]):
        print(evaluate.render_calibration(e.calibration, e.name))
        print()

    _banner("STAGE 6 — ROI: IS THERE A POSITIVE-RETURN INTERVENTION?")
    best = gb.predictions
    res = roi.target_top(y_true, best, 0.10)
    be = roi.breakeven_table(res)
    grid = roi.channel_grid(res)
    strategies = roi.compare_strategies(
        y_true,
        {
            "LightGBM (Tweedie)": best,
            "BG/NBD x observed AOV": bg_pred_emp,
            "Historical spend (sort)": test_panel.X["monetary"].to_numpy(),
        },
        fraction=0.10,
        n_boot=min(n_boot, 500),
    )
    print(roi.render(res, be, grid, strategies))

    if run_simulation:
        _banner("STAGE 7 — IS BG/NBD IDENTIFIABLE AT THIS REPEAT RATE?")
        sim = bgnbd.simulate_identifiability()
        print(f"  {'target %':>9}{'realised %':>12}{'mean param err':>17}{'worst':>9}")
        print("  " + "-" * 50)
        for _, r in sim.iterrows():
            print(f"  {r['target_repeat_rate']:>8.1f}%{r['realised_repeat_rate']:>11.2f}%"
                  f"{r['mean_abs_err_pct']:>16.1f}%{r['worst_err_pct']:>8.1f}%")
        sim.to_parquet(PROCESSED_DIR / "bgnbd_identifiability.parquet", index=False)

    _banner("STAGE 8 — SAVE ARTIFACTS")
    preds = pd.DataFrame({
        CUSTOMER_KEY: test_panel.meta[CUSTOMER_KEY].to_numpy(),
        "actual_90d_spend": y_true,
        "pred_gbm": gb.predictions,
        "pred_bgnbd": bg_pred,
        "pred_bgnbd_empirical": bg_pred_emp,
    })
    preds = preds.merge(
        test_panel.X.assign(**{CUSTOMER_KEY: test_panel.meta[CUSTOMER_KEY].to_numpy()}),
        on=CUSTOMER_KEY, how="left",
    )
    preds.to_parquet(PREDICTIONS_PARQUET, index=False)

    bundle = {
        "model_kind": "repeat-propensity ranker (not a CLV point predictor)",
        "gbm_booster": gb.booster,
        "gbm_best_iteration": gb.best_iteration,
        "gbm_params": gb.params,
        "feature_names": list(train_panel.X.columns),
        "categorical": features.CATEGORICAL,
        "categories": {c: list(train_panel.X[c].cat.categories) for c in features.CATEGORICAL},
        "bgnbd_params": bg.params,
        "bgnbd_diagnostics": bg.diagnostics,
        "bgnbd_gg_valid": bg.gg_valid,
        "horizon_days": features.HORIZON_DAYS,
        "train_origins": features.TRAIN_ORIGINS,
        "valid_origin": features.VALID_ORIGIN,
        "test_origin": features.TEST_ORIGIN,
        "leakage_guard": lk,
        "feature_importance": gb.feature_importance,
        "evaluation": pd.DataFrame([e.summary_row() for e in evals]),
        "calibration": {e.name: e.calibration for e in evals},
        "roi_breakeven": be,
        "roi_channels": grid,
        "roi_strategies": strategies,
        "n_test_positives": int((y_true > 0).sum()),
    }
    joblib.dump(bundle, MODEL_BUNDLE)
    print(f"  wrote {MODEL_BUNDLE.name}  ({MODEL_BUNDLE.stat().st_size / 1e6:,.2f} MB)")
    print(f"  wrote {PREDICTIONS_PARQUET.name}  "
          f"({PREDICTIONS_PARQUET.stat().st_size / 1e6:,.2f} MB)")
    print(f"\n  elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":  # pragma: no cover
    run()
