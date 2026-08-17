"""Cached data access and the shared filter model.

Everything the app reads comes from data/dashboard/*.parquet, which is
committed to git so a cold Streamlit Cloud boot has data immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "data" / "dashboard"

CUSTOMER_KEY = "customer_unique_id"


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing. Run:\n"
            f"  python scripts/run_etl.py\n"
            f"  python scripts/run_analytics.py\n"
            f"  python scripts/train_clv.py\n"
            f"  python scripts/export_dashboard_data.py"
        )
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def meta() -> dict[str, str]:
    return dict(load("meta")[["key", "value"]].to_numpy())


@dataclass(frozen=True)
class Filters:
    """The filter state shared across pages."""
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    states: tuple[str, ...]
    categories: tuple[str, ...]

    @property
    def is_active(self) -> bool:
        return bool(self.states or self.categories)


def apply_filters(orders: pd.DataFrame, f: Filters) -> pd.DataFrame:
    m = (orders["order_purchase_timestamp"] >= f.date_start) & (
        orders["order_purchase_timestamp"] <= f.date_end
    )
    if f.states:
        m &= orders["customer_state"].isin(f.states)
    if f.categories:
        m &= orders["top_category"].isin(f.categories)
    return orders[m]


def filter_customers(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """Apply the dimension filters to a customer-grain table."""
    if df.empty:
        return df
    m = pd.Series(True, index=df.index)
    if f.states and "customer_state" in df.columns:
        m &= df["customer_state"].isin(f.states)
    if f.categories and "top_category" in df.columns:
        m &= df["top_category"].isin(f.categories)
    return df[m]


def sidebar_filters() -> Filters:
    """Render the shared sidebar controls and return the resulting filter."""
    orders = load("orders")
    lo = orders["order_purchase_timestamp"].min().to_pydatetime()
    hi = orders["order_purchase_timestamp"].max().to_pydatetime()

    st.sidebar.markdown("### Filters")

    preset = st.sidebar.radio(
        "Date range",
        ["All time", "Last 3 months", "Last 6 months", "Last 12 months", "Custom"],
        index=0,
    )
    presets = {"Last 3 months": 90, "Last 6 months": 182, "Last 12 months": 365}
    if preset == "All time":
        start, end = lo, hi
    elif preset == "Custom":
        picked = st.sidebar.date_input(
            "Custom range", value=(lo.date(), hi.date()),
            min_value=lo.date(), max_value=hi.date(),
        )
        if isinstance(picked, tuple) and len(picked) == 2:
            start = pd.Timestamp(picked[0])
            end = pd.Timestamp(picked[1]) + pd.Timedelta(days=1)
        else:
            start, end = lo, hi
    else:
        end = hi
        start = hi - pd.Timedelta(days=presets[preset])

    all_states = sorted(orders["customer_state"].dropna().unique().tolist())
    states = st.sidebar.multiselect("Customer state", all_states, default=[])

    all_cats = sorted(orders["top_category"].dropna().unique().tolist())
    cats = st.sidebar.multiselect("Product category", all_cats, default=[])

    f = Filters(
        date_start=pd.Timestamp(start),
        date_end=pd.Timestamp(end),
        states=tuple(states),
        categories=tuple(cats),
    )

    filtered = apply_filters(orders, f)
    st.sidebar.markdown(
        f"<div style='color:#898781;font-size:0.82rem;margin-top:.6rem'>"
        f"<b>{len(filtered):,}</b> of {len(orders):,} orders selected<br>"
        f"{f.date_start:%d %b %Y} &ndash; {f.date_end:%d %b %Y}</div>",
        unsafe_allow_html=True,
    )
    if filtered.empty:
        st.sidebar.warning("No orders match these filters.")

    return f


def fmt_brl(v: float, decimals: int = 0) -> str:
    return f"R$ {v:,.{decimals}f}"


def fmt_pct(v: float, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}%"
