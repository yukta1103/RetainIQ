"""Page 4 — CLV Predictions: per-customer lookup, what-if scoring, model comparison."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app import charts, data
from app import theme as T

ARTIFACT = data.DASHBOARD_DIR.parents[1] / "artifacts" / "clv_model.joblib"


@st.cache_resource(show_spinner=False)
def load_bundle():
    if not ARTIFACT.exists():
        return None
    return joblib.load(ARTIFACT)


def render(f: data.Filters) -> None:
    st.title("CLV Predictions")
    st.caption(
        "Predicted spend over the next 90 days, scored at 2018-06-01 from "
        "features that use only prior data."
    )

    clv = data.filter_customers(data.load("clv"), f)
    segments = data.load("segments")[[data.CUSTOMER_KEY, "segment"]]
    clv = clv.merge(segments, on=data.CUSTOMER_KEY, how="left")

    if clv.empty:
        st.warning("No customers match the current filters.")
        return

    bundle = load_bundle()

    tab_lookup, tab_dist, tab_models = st.tabs(
        ["Score a customer", "Distribution by segment", "Model comparison"]
    )

    # ---------------------------------------------------------------- lookup
    with tab_lookup:
        st.markdown("#### Look up a customer")
        mode = st.radio("Pick by", ["Highest predicted value", "Customer ID", "Random"],
                        horizontal=True)

        if mode == "Highest predicted value":
            pool = clv.nlargest(200, "pred_gbm")
            label = st.selectbox(
                "Customer", pool[data.CUSTOMER_KEY].tolist(),
                format_func=lambda c: (
                    f"{c[:12]}…  ·  predicted "
                    f"R$ {float(pool.loc[pool[data.CUSTOMER_KEY] == c, 'pred_gbm'].iloc[0]):.2f}"
                ),
            )
        elif mode == "Random":
            if st.button("Draw another"):
                st.session_state["_seed"] = np.random.randint(1e6)
            seed = st.session_state.get("_seed", 0)
            label = clv.sample(1, random_state=seed)[data.CUSTOMER_KEY].iloc[0]
            st.caption(f"Customer `{label}`")
        else:
            typed = st.text_input("Customer ID (or a prefix)", "")
            hits = clv[clv[data.CUSTOMER_KEY].str.startswith(typed)] if typed else clv.head(0)
            if typed and hits.empty:
                st.info("No customer matches that prefix.")
                return
            label = hits[data.CUSTOMER_KEY].iloc[0] if not hits.empty else None

        if label is None:
            st.info("Enter a customer ID to score.")
        else:
            row = clv[clv[data.CUSTOMER_KEY] == label].iloc[0]
            _render_customer(row)

        st.divider()
        st.markdown("#### Or score a hypothetical customer")
        st.caption(
            "Adjust the inputs and the saved LightGBM model scores them live — "
            "no retraining."
        )
        _what_if(bundle, clv)

    # ---------------------------------------------------------- distribution
    with tab_dist:
        model_col = st.selectbox(
            "Model", ["pred_gbm", "pred_bgnbd_empirical", "pred_bgnbd"],
            format_func=lambda c: {
                "pred_gbm": "LightGBM (Tweedie) — recommended",
                "pred_bgnbd_empirical": "BG/NBD x observed AOV",
                "pred_bgnbd": "BG/NBD + Gamma-Gamma (broken — see note)",
            }[c],
        )
        if model_col == "pred_bgnbd":
            st.markdown(
                "<div class='caveat'><b>This model is structurally broken on this "
                "dataset.</b> Gamma-Gamma fitted q = 0.497, and since expected spend "
                "is p·v/(q−1), any q &lt; 1 makes it negative — 73,639 customers get a "
                "negative predicted CLV. It is shown so the failure is visible, not "
                "because it should be used.</div>",
                unsafe_allow_html=True,
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers scored", f"{len(clv):,}")
        c2.metric("Total predicted", data.fmt_brl(clv[model_col].sum()))
        c3.metric("Total actual", data.fmt_brl(clv["actual_90d_spend"].sum()))
        c4.metric("Actually returned", f"{int((clv['actual_90d_spend'] > 0).sum()):,}",
                  help="Customers who made a purchase in the 90-day holdout window.")

        st.plotly_chart(charts.clv_by_segment(clv, model_col), use_container_width=True)

        seg_tbl = (
            clv.groupby("segment", observed=True)
            .agg(customers=(data.CUSTOMER_KEY, "size"),
                 mean_pred=(model_col, "mean"),
                 total_pred=(model_col, "sum"),
                 mean_actual=("actual_90d_spend", "mean"),
                 returned=("actual_90d_spend", lambda s: int((s > 0).sum())))
            .reset_index()
        )
        seg_tbl["return_rate"] = 100 * seg_tbl["returned"] / seg_tbl["customers"]
        seg_tbl = seg_tbl.sort_values("mean_pred", ascending=False)
        st.dataframe(
            seg_tbl.rename(columns={
                "segment": "Segment", "customers": "Customers",
                "mean_pred": "Mean predicted", "total_pred": "Total predicted",
                "mean_actual": "Mean actual", "returned": "Returned",
                "return_rate": "Return rate"}),
            hide_index=True, use_container_width=True,
            column_config={
                "Mean predicted": st.column_config.NumberColumn(format="R$ %.2f"),
                "Total predicted": st.column_config.NumberColumn(format="R$ %.0f"),
                "Mean actual": st.column_config.NumberColumn(format="R$ %.2f"),
                "Return rate": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    # ------------------------------------------------------------- model cmp
    with tab_models:
        if bundle is None:
            st.info("Model artifact not found — run `python scripts/train_clv.py`.")
            return

        ev = bundle["evaluation"]
        st.markdown("#### How the models compare")
        st.markdown(
            "<div class='caveat'><b>Read MAE with suspicion.</b> The target is "
            "99.44% zeros, so 'predict zero for everyone' scores MAE 0.890 — better "
            "than every real model. Ranking metrics are what separate them: AUC and "
            "top-decile revenue capture.</div>",
            unsafe_allow_html=True,
        )

        st.dataframe(
            ev.rename(columns={"model": "Model", "top10%_capture": "Top-10% capture",
                               "top10%_lift": "Lift vs random"}),
            hide_index=True, use_container_width=True,
            column_config={
                "MAE": st.column_config.NumberColumn(format="%.3f"),
                "RMSE": st.column_config.NumberColumn(format="%.2f"),
                "Spearman": st.column_config.NumberColumn(format="%.4f"),
                "AUC": st.column_config.NumberColumn(format="%.4f"),
                "Top-10% capture": st.column_config.NumberColumn(format="%.1f%%"),
                "Lift vs random": st.column_config.NumberColumn(format="%.2fx"),
                "pred_total": st.column_config.NumberColumn("Predicted total", format="R$ %.0f"),
                "actual_total": st.column_config.NumberColumn("Actual total", format="R$ %.0f"),
            },
        )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.model_comparison(ev, "AUC", "Ranking quality (AUC)"),
                            use_container_width=True)
        with right:
            st.plotly_chart(
                charts.model_comparison(ev, "top10%_capture",
                                        "Share of holdout revenue in the top decile (%)"),
                use_container_width=True)

        st.markdown("#### Decile calibration")
        which = st.selectbox("Model", list(bundle["calibration"].keys()), index=2)
        cal = bundle["calibration"][which]
        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.decile_calibration(cal, f"{which}: predicted vs actual"),
                            use_container_width=True)
        with right:
            st.plotly_chart(charts.revenue_capture(cal), use_container_width=True)

        with st.expander("Top model features"):
            st.dataframe(bundle["feature_importance"].head(15), hide_index=True,
                         use_container_width=True)


def _render_customer(row: pd.Series) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted 90-day spend", data.fmt_brl(row["pred_gbm"], 2))
    c2.metric("Actual (holdout)", data.fmt_brl(row["actual_90d_spend"], 2))
    c3.metric("Segment", str(row.get("segment", "—")))
    c4.metric("Historic spend", data.fmt_brl(row["monetary"], 2))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recency", f"{row['recency']:.0f} d")
    c2.metric("Frequency", f"{row['frequency']:.0f}")
    c3.metric("Tenure", f"{row['tenure']:.0f} d")
    c4.metric("Avg order value", data.fmt_brl(row["aov"], 2))

    st.caption(
        f"State {row.get('customer_state', '—')} · "
        f"usual category {row.get('top_category', '—')} · "
        f"BG/NBD (empirical AOV) would say {data.fmt_brl(row['pred_bgnbd_empirical'], 2)}"
    )


def _what_if(bundle, clv: pd.DataFrame) -> None:
    if bundle is None:
        st.info("Model artifact not found — run `python scripts/train_clv.py`.")
        return

    c1, c2, c3 = st.columns(3)
    recency = c1.slider("Days since last purchase", 0, 600, 60)
    frequency = c2.slider("Purchase occasions", 1, 8, 1)
    monetary = c3.slider("Total spent so far (R$)", 10, 3000, 200, step=10)

    c1, c2, c3 = st.columns(3)
    tenure = c1.slider("Days since first purchase", 1, 600, 200)
    state = c2.selectbox("State", sorted(clv["customer_state"].dropna().unique().tolist()),
                         index=0)
    category = c3.selectbox("Usual category",
                            sorted(clv["top_category"].dropna().unique().tolist()), index=0)

    if tenure < recency:
        st.warning("Tenure must be at least as long as recency — a customer cannot "
                   "have last purchased before they first purchased.")
        return

    # Start from the population median so unspecified features are realistic,
    # then overwrite the ones the user controls.
    feature_names = bundle["feature_names"]
    base = _median_row(feature_names)
    row = base.copy()
    row.update({
        "recency": recency, "frequency": frequency, "monetary": monetary,
        "tenure": tenure, "aov": monetary / max(frequency, 1),
        "n_orders": frequency, "is_repeat": int(frequency > 1),
        "max_order": monetary / max(frequency, 1),
        "min_order": monetary / max(frequency, 1),
        "purchase_rate": frequency / max(tenure, 1) * 365,
        "spend_per_day": monetary / max(tenure, 1),
        "recency_over_tenure": recency / max(tenure, 1),
    })

    X = pd.DataFrame([row])[feature_names]
    for c in bundle["categorical"]:
        X[c] = pd.Categorical(
            [state if c == "customer_state" else category if c == "top_category"
             else bundle["categories"][c][0]],
            categories=bundle["categories"][c],
        )
    for c in X.columns:
        if c not in bundle["categorical"]:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    pred = float(bundle["gbm_booster"].predict(
        X, num_iteration=bundle["gbm_best_iteration"])[0])

    st.metric("Predicted 90-day spend", data.fmt_brl(pred, 2))
    pct = float((clv["pred_gbm"] < pred).mean() * 100)
    st.caption(
        f"That is higher than {pct:.1f}% of the {len(clv):,} scored customers. "
        f"For context the population mean prediction is "
        f"{data.fmt_brl(clv['pred_gbm'].mean(), 2)}."
    )


@st.cache_data(show_spinner=False)
def _median_row(feature_names: list[str]) -> dict:
    """Population medians, so what-if inputs sit in a realistic context."""
    clv = data.load("clv")
    out = {}
    for c in feature_names:
        if c in clv.columns and pd.api.types.is_numeric_dtype(clv[c]):
            out[c] = float(clv[c].median())
        else:
            out[c] = 0.0
    return out
