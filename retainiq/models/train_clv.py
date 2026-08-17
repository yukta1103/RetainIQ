"""Phase 3 orchestrator: build panels, fit both models, evaluate, save artifacts."""

from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from retainiq.config import ARTIFACTS_DIR, CUSTOMER_KEY, ORDERS_PARQUET, PROCESSED_DIR
from retainiq.models import bgnbd, evaluate, features, gbm

MODEL_BUNDLE = ARTIFACTS_DIR / "clv_model.joblib"
PREDICTIONS_PARQUET = PROCESSED_DIR / "clv_predictions.parquet"


def _banner(title: str) -> None:
    print(f"\n\n{'#' * 78}\n#  {title}\n{'#' * 78}")


def run(run_simulation: bool = True) -> None:
    t0 = time.time()
    orders = pd.read_parquet(ORDERS_PARQUET)

    _banner("STAGE 1 — PANEL CONSTRUCTION (time-ordered, leakage-checked)")
    train_panel, test_panel = features.build_train_test(orders)
    print(f"  horizon: {features.HORIZON_DAYS} days")
    print(features.describe_panel("train", train_panel))
    print(features.describe_panel("test", test_panel))
    print(
        f"\n  Every training target window closes before the test origin "
        f"{features.TEST_ORIGIN}; build_train_test() raises if that is ever violated."
    )
    print(
        f"  Stacking {len(features.TRAIN_ORIGINS)} origins lifts positives from "
        f"418 (single split) to {int((train_panel.y > 0).sum()):,}."
    )

    _banner("STAGE 2 — MODEL A: BG/NBD + GAMMA-GAMMA")
    origin = pd.Timestamp(features.TEST_ORIGIN)
    bg = bgnbd.fit(orders, origin, horizon_days=features.HORIZON_DAYS)

    print("  fitted parameters:")
    for k, v in bg.params.items():
        print(f"    {k:<10} {v:>12.4f}")
    d = bg.diagnostics
    print(
        f"\n  likelihood support: {d['n_frequency_gt0']:,} of {d['n_customers']:,} "
        f"customers ({d['pct_frequency_gt0']:.2f}%) have frequency > 0"
    )
    print(f"  Gamma-Gamma assumption: {bg.gg_note}")
    print(f"  Gamma-Gamma fitted    : {bg.gg_fitted}")

    print("\n  DIAGNOSTIC — did the fit stay in a valid region?")
    print(f"    expected purchases : {d['expected_purchases_min']:.4f} .. "
          f"{d['expected_purchases_max']:.4f}   (well behaved)")
    print(f"    expected value min : {d['expected_value_min']:,.2f}")
    print(f"    negative values    : {d['n_negative_expected_value']:,} customers")
    print(f"    validity           : {d['gg_validity_note']}")
    if not bg.gg_valid:
        print(
            "\n    -> BG/NBD's purchase-count model is fine; GAMMA-GAMMA is what\n"
            "       broke. Fitted on only 1,493 repeat customers, its q parameter\n"
            "       fell below 1 and the expected-value formula went negative.\n"
            "       Reported below both as-fitted and with the value model\n"
            "       replaced by observed AOV, to isolate the failure."
        )

    bg_pred = bg.predictions.reindex(test_panel.meta[CUSTOMER_KEY]).fillna(0.0).to_numpy()
    bg_pred_emp = (
        bg.predictions_empirical.reindex(test_panel.meta[CUSTOMER_KEY]).fillna(0.0).to_numpy()
    )

    _banner("STAGE 3 — MODEL B: LIGHTGBM (TWEEDIE)")
    gb = gbm.train(train_panel, test_panel)
    print(f"  objective        : {gb.params['objective']} "
          f"(variance power {gb.params['tweedie_variance_power']})")
    print(f"  best iteration   : {gb.best_iteration}")
    print("\n  top 12 features by gain:")
    for _, r in gb.feature_importance.head(12).iterrows():
        print(f"    {r['feature']:<24}{r['gain_pct']:>7.2f}%  ({int(r['split']):>5,} splits)")

    print("\n  fitting L2 baseline to justify the Tweedie objective...")
    gb_l2 = gbm.train_l2_baseline(train_panel, test_panel)

    _banner("STAGE 4 — EVALUATION")
    y_true = test_panel.y.to_numpy()

    print("  Trivial baselines first, so MAE can be read in context:")
    base = evaluate.baseline_comparison(y_true, train_panel.y.to_numpy())
    for _, r in base.iterrows():
        print(f"    {r['baseline']:<22} MAE {r['MAE']:>8.3f}   RMSE {r['RMSE']:>8.2f}")

    evals = [
        evaluate.evaluate("BG/NBD + Gamma-Gamma", y_true, bg_pred),
        evaluate.evaluate("BG/NBD x observed AOV", y_true, bg_pred_emp),
        evaluate.evaluate("LightGBM (Tweedie)", y_true, gb.predictions),
        evaluate.evaluate("LightGBM (L2)", y_true, gb_l2.predictions),
        evaluate.evaluate("RFM monetary (heuristic)", y_true, test_panel.X["monetary"].to_numpy()),
    ]
    print()
    print(evaluate.render_comparison(evals))
    print(
        "\n  MAE/RMSE are reported because they were specified, but note that\n"
        "  'predict zero' is competitive on them. Spearman, AUC and top-decile\n"
        "  capture are what separate the models."
    )

    _banner("STAGE 5 — DECILE CALIBRATION")
    for e in (evals[0], evals[2]):
        print(evaluate.render_calibration(e.calibration, e.name))
        print()

    if run_simulation:
        _banner("STAGE 6 — IS BG/NBD IDENTIFIABLE AT THIS REPEAT RATE?")
        print("  Recovering known parameters from synthetic BG/NBD data as the")
        print("  repeat rate falls. If recovery collapses before reaching 2%,")
        print("  the Olist fit cannot be trusted regardless of how it looks.\n")
        sim = bgnbd.simulate_identifiability()
        print(f"  {'target %':>9}{'realised %':>12}{'mean param err':>17}{'worst':>9}{'converged':>11}")
        print("  " + "-" * 60)
        for _, r in sim.iterrows():
            print(
                f"  {r['target_repeat_rate']:>8.1f}%{r['realised_repeat_rate']:>11.2f}%"
                f"{r['mean_abs_err_pct']:>16.1f}%{r['worst_err_pct']:>8.1f}%"
                f"{int(r['converged']):>7}/{int(r['n'])}"
            )
        sim.to_parquet(PROCESSED_DIR / "bgnbd_identifiability.parquet", index=False)

    _banner("STAGE 7 — SAVE ARTIFACTS")
    preds = pd.DataFrame(
        {
            CUSTOMER_KEY: test_panel.meta[CUSTOMER_KEY].to_numpy(),
            "actual_90d_spend": y_true,
            "pred_gbm": gb.predictions,
            "pred_bgnbd": bg_pred,
            "pred_bgnbd_empirical": bg_pred_emp,
            "bgnbd_expected_purchases": bg.expected_purchases.reindex(
                test_panel.meta[CUSTOMER_KEY]
            ).fillna(0.0).to_numpy(),
            "bgnbd_expected_value": bg.expected_value.reindex(
                test_panel.meta[CUSTOMER_KEY]
            ).fillna(0.0).to_numpy(),
        }
    )
    preds = preds.merge(
        test_panel.X.assign(**{CUSTOMER_KEY: test_panel.meta[CUSTOMER_KEY].to_numpy()}),
        on=CUSTOMER_KEY,
        how="left",
    )
    preds.to_parquet(PREDICTIONS_PARQUET, index=False)

    bundle = {
        "gbm_booster": gb.booster,
        "gbm_best_iteration": gb.best_iteration,
        "gbm_params": gb.params,
        "feature_names": list(train_panel.X.columns),
        "categorical": features.CATEGORICAL,
        "categories": {c: list(train_panel.X[c].cat.categories) for c in features.CATEGORICAL},
        # lifetimes fitters close over a lambda in fit() and cannot be pickled.
        # Persist parameters; bgnbd.rebuild_bgf/rebuild_ggf restore a usable
        # fitter, since prediction is closed-form in those parameters.
        "bgnbd_params": bg.params,
        "bgnbd_diagnostics": bg.diagnostics,
        "bgnbd_gg_valid": bg.gg_valid,
        "horizon_days": features.HORIZON_DAYS,
        "test_origin": features.TEST_ORIGIN,
        "train_origins": features.TRAIN_ORIGINS,
        "feature_importance": gb.feature_importance,
        "evaluation": pd.DataFrame([e.summary_row() for e in evals]),
        "calibration": {e.name: e.calibration for e in evals},
    }
    joblib.dump(bundle, MODEL_BUNDLE)
    print(f"  wrote {MODEL_BUNDLE.name}  ({MODEL_BUNDLE.stat().st_size / 1e6:,.2f} MB)")
    print(f"  wrote {PREDICTIONS_PARQUET.name}  ({PREDICTIONS_PARQUET.stat().st_size / 1e6:,.2f} MB)")
    print(f"\n  elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":  # pragma: no cover
    run()
