"""Page 3 — Cohort Retention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import charts, data


def _matrix(name: str) -> tuple[pd.DataFrame, pd.Series]:
    m = data.load(name).copy()
    m["cohort_month"] = pd.to_datetime(m["cohort_month"])
    m = m.set_index("cohort_month")
    m.columns = [int(c) for c in m.columns]
    sizes = data.load("cohort_sizes").copy()
    sizes["cohort_month"] = pd.to_datetime(sizes["cohort_month"])
    return m, sizes.set_index("cohort_month")["customers"]


def render(f: data.Filters) -> None:
    st.title("Cohort Retention")
    st.caption(
        "Customers grouped by first-purchase month, tracked forward. "
        "Blank cells are months a cohort has not lived through yet."
    )

    view = st.radio(
        "View",
        ["Cumulative repeat rate", "Classic monthly retention", "Revenue per customer"],
        horizontal=True,
        help="Cumulative is the informative view at a ~2% repeat rate; see the note below.",
    )

    max_month = st.slider("Months to show", 3, 19, 13)

    if view == "Classic monthly retention":
        mat, sizes = _matrix("cohort_retention")
        mat = mat.drop(columns=[0], errors="ignore")  # month 0 is 100% by construction
        mat = mat[[c for c in mat.columns if c <= max_month]]
        st.markdown(
            "<div class='caveat'><b>This is the honest picture, and it is nearly "
            "flat.</b> Every cell sits near 0.3%. At a 2.15% repeat rate, classic "
            "month-by-month retention carries almost no signal — which is itself "
            "the finding. The cumulative view accumulates signal instead.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            charts.cohort_heatmap(mat, sizes,
                                  "% of cohort placing an order in month N",
                                  zmax=1.0, unit="%"),
            use_container_width=True)

    elif view == "Cumulative repeat rate":
        mat, sizes = _matrix("cohort_cumulative")
        mat = mat[[c for c in mat.columns if 0 < c <= max_month]]
        st.plotly_chart(
            charts.cohort_heatmap(mat, sizes,
                                  "% of cohort that has EVER returned by month N",
                                  unit="%"),
            use_container_width=True)
        st.plotly_chart(charts.cohort_curves(mat), use_container_width=True)

    else:
        mat, sizes = _matrix("cohort_revenue")
        mat = mat[[c for c in mat.columns if c <= max_month]]
        st.plotly_chart(
            charts.cohort_heatmap(mat, sizes,
                                  "Cumulative revenue per customer by cohort age (R$)",
                                  unit="R$"),
            use_container_width=True)

    st.markdown("#### Cohort summary")
    cum, sizes = _matrix("cohort_cumulative")
    rows = []
    for cohort in sizes.index:
        row = {"Cohort": cohort.strftime("%Y-%m"), "Customers": int(sizes[cohort])}
        for h in (3, 6, 12):
            v = cum.loc[cohort, h] if h in cum.columns else np.nan
            row[f"Returned by M{h}"] = float(v) if pd.notna(v) else None
        rows.append(row)
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, use_container_width=True,
        column_config={
            f"Returned by M{h}": st.column_config.NumberColumn(format="%.2f%%")
            for h in (3, 6, 12)
        },
    )

    with st.expander("How to read this"):
        st.markdown("""
**Cohort** = the month a customer made their first purchase. Row = cohort,
column = months elapsed since then.

**Classic retention** asks "did this customer order again *in month N*". For a
subscription business that is the right question. For a marketplace where
people buy a mattress once, it produces a near-empty matrix.

**Cumulative repeat rate** asks "has this customer come back *at all* by month
N". It is monotonically increasing and actually accumulates signal — cohorts
converge to roughly 3% by month 13.

The unobservable triangle is deliberately blank rather than zero: a cohort that
has only existed for four months must not be read as 0% retention at month 12.
        """)
