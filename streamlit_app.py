"""RetainIQ dashboard entrypoint (Streamlit Cloud looks for this at repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import data, theme  # noqa: E402
from app.pages_ import clv, cohorts, insights, overview, segments  # noqa: E402

st.set_page_config(
    page_title="RetainIQ — Olist Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.register_template()
st.markdown(theme.CSS, unsafe_allow_html=True)

PAGES = {
    "Overview": overview.render,
    "Customer Segments": segments.render,
    "Cohort Retention": cohorts.render,
    "Repeat Propensity": clv.render,
    "Business Insights": insights.render,
}


def main() -> None:
    st.sidebar.markdown("## RetainIQ")
    st.sidebar.caption("Customer analytics on the Olist Brazilian e-commerce dataset")

    # Page selection is mirrored into the URL (?page=...) so individual pages
    # are deep-linkable — useful for sharing a specific view and for taking
    # reproducible screenshots without having to script clicks.
    names = list(PAGES)
    requested = st.query_params.get("page", names[0])
    if requested not in PAGES:
        requested = names[0]

    choice = st.sidebar.radio(
        "Page", names, index=names.index(requested), label_visibility="collapsed"
    )
    if choice != requested:
        st.query_params["page"] = choice

    st.sidebar.divider()

    try:
        filters = data.sidebar_filters()
    except FileNotFoundError as e:
        st.error("Dashboard data is missing.")
        st.code(str(e))
        st.stop()

    st.sidebar.divider()
    st.sidebar.caption(
        "Built from the raw 9-table Olist dataset: ETL → RFM & cohorts → "
        "repeat-propensity ranking → this dashboard."
    )

    PAGES[choice](filters)


if __name__ == "__main__":
    main()
