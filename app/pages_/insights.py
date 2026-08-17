"""Page 5 — Business Insights: the takeaways, each with its supporting chart."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, data
from app import theme as T


def _card(title: str, body: str) -> None:
    st.markdown(
        f"<div class='insight-card'><h4>{title}</h4><p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def render(f: data.Filters) -> None:
    st.title("Business Insights")
    st.caption(
        "Four takeaways, each with the evidence behind it. Figures are computed "
        "over the full dataset, not the sidebar filters, so the numbers stay "
        "quotable."
    )

    seg = data.load("segments")
    orders = data.load("orders")
    clv = data.load("clv")
    meta = data.meta()

    # ---------------------------------------------------------------- 1
    mon = seg["monetary"].sort_values(ascending=False)
    top5 = 100 * mon.head(int(len(mon) * 0.05)).sum() / mon.sum()
    top20 = 100 * mon.head(int(len(mon) * 0.20)).sum() / mon.sum()
    big = seg[seg["segment"] == "Big Spender (One-Off)"]
    big_pct_cust = 100 * len(big) / len(seg)
    big_pct_rev = 100 * big["monetary"].sum() / seg["monetary"].sum()

    _card(
        "1 &nbsp;·&nbsp; The valuable customers are one-time big spenders, not loyalists",
        f"The textbook finding is that a small loyal core drives most revenue. "
        f"That is <b>not</b> what this marketplace looks like. Champions are just "
        f"<b>0.10%</b> of customers and <b>0.52%</b> of revenue — there is barely a "
        f"loyal core to reward. The segment that actually punches above its weight is "
        f"<b>Big Spender (One-Off)</b>: <b>{big_pct_cust:.2f}%</b> of customers "
        f"producing <b>{big_pct_rev:.2f}%</b> of revenue, a <b>5.5x</b> revenue index. "
        f"Concentration is real but blunt: the top 5% of customers hold "
        f"<b>{top5:.1f}%</b> of revenue and the top 20% hold <b>{top20:.1f}%</b>. "
        f"<br><br><b>So what:</b> retention spend aimed at 'rewarding loyalty' has "
        f"almost no target. The lever is converting a high-value first purchase into "
        f"a second one."
    )
    summary = data.load("segment_summary")
    st.plotly_chart(charts.segment_value_scatter(summary), use_container_width=True)

    st.divider()

    # ---------------------------------------------------------------- 2
    _card(
        "2 &nbsp;·&nbsp; The headline retention metric was wrong by 40%, and it was a data-model bug",
        f"Olist splits a single basket into one order per seller. "
        f"<b>29.6%</b> of consecutive order pairs from the same customer are less "
        f"than 24 hours apart — <b>27.3%</b> are under one hour. Those are not repeat "
        f"purchases; they are one shopping trip recorded as several orders. "
        f"Counting orders gives a repeat rate of <b>3.00%</b>. Counting distinct "
        f"purchase days — one decision, one occasion — gives <b>2.15%</b>. "
        f"<br><br><b>So what:</b> every downstream number moves. 852 customers were "
        f"misclassified as repeat buyers, which inflates any retention target, any "
        f"cohort curve, and any CLV model trained on frequency."
    )
    occ = pd.DataFrame({
        "definition": ["Counting orders", "Counting purchase days"],
        "repeat_rate": [3.00, 2.15],
    })
    fig = charts._fig("Repeat purchase rate under each definition", 280, "Repeat rate (%)")
    import plotly.graph_objects as go
    fig.add_trace(go.Bar(
        x=occ["definition"], y=occ["repeat_rate"],
        marker=dict(color=[T.ORANGE, T.BLUE], line=dict(width=2, color=T.SURFACE)),
        text=[f"{v:.2f}%" for v in occ["repeat_rate"]], textposition="outside",
        textfont=dict(size=12, color=T.INK_SECONDARY),
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(showlegend=False, bargap=0.5)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------------------------------------- 3
    _card(
        "3 &nbsp;·&nbsp; The textbook CLV model fails here — and it fails in a specific, diagnosable way",
        f"BG/NBD + Gamma-Gamma is the standard probabilistic CLV approach. On this "
        f"data it scores <b>AUC 0.467 — worse than random</b>. The cause is precise: "
        f"Gamma-Gamma was fitted on only <b>1,493</b> repeat customers and its "
        f"<i>q</i> parameter came out at <b>0.497</b>. Expected spend is "
        f"<i>p·v/(q−1)</i>, so any <i>q</i> &lt; 1 makes it <b>negative</b> — and "
        f"<b>73,639</b> customers received a negative predicted CLV. "
        f"BG/NBD's purchase-count half is fine; swapping the broken value model for "
        f"observed AOV recovers AUC to <b>0.555</b>. "
        f"<br><br><b>So what:</b> reaching for the standard tool without checking its "
        f"parameter constraints would have shipped a model that ranks customers "
        f"backwards. A gradient-boosted Tweedie model reaches <b>AUC 0.607</b> and "
        f"<b>2.0x</b> lift on the top decile."
    )
    bundle = None
    try:
        import joblib
        art = data.DASHBOARD_DIR.parents[1] / "artifacts" / "clv_model.joblib"
        if art.exists():
            bundle = joblib.load(art)
    except Exception:
        bundle = None

    if bundle is not None:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                charts.model_comparison(bundle["evaluation"], "AUC",
                                        "Ranking quality (AUC) — 0.5 is random"),
                use_container_width=True)
        with right:
            cal = bundle["calibration"].get("LightGBM (Tweedie)")
            if cal is not None:
                st.plotly_chart(charts.revenue_capture(cal), use_container_width=True)

    st.divider()

    # ---------------------------------------------------------------- 4
    hib = seg[seg["segment"] == "Hibernating"]
    hib_pct = 100 * len(hib) / len(seg)
    hib_rev = 100 * hib["monetary"].sum() / seg["monetary"].sum()
    new_seg = seg[seg["segment"] == "New"]

    _card(
        "4 &nbsp;·&nbsp; Half the customer base is drifting away in a predictable window",
        f"<b>Hibernating</b> — a single purchase, 180 to 365 days ago — is "
        f"<b>{hib_pct:.1f}%</b> of all customers and <b>{hib_rev:.1f}%</b> of revenue. "
        f"They are not lost yet, but the repurchase curve says the window is closing: "
        f"of genuine repurchases, <b>56%</b> happen within 90 days and <b>96.4%</b> "
        f"within a year. After 365 days only <b>3.6%</b> of repurchases ever occur. "
        f"Meanwhile <b>{len(new_seg):,}</b> customers sit in <b>New</b> — bought "
        f"within 90 days, still inside the window where intervention can work. "
        f"<br><br><b>So what:</b> the intervention should be early and cheap, aimed "
        f"at the 90-day mark, not a win-back campaign at month 12 when the curve says "
        f"it is already over."
    )
    cum = data.load("cohort_cumulative").copy()
    cum["cohort_month"] = pd.to_datetime(cum["cohort_month"])
    cum = cum.set_index("cohort_month")
    cum.columns = [int(c) for c in cum.columns]
    st.plotly_chart(charts.cohort_curves(cum[[c for c in cum.columns if c <= 13]]),
                    use_container_width=True)

    st.divider()
    st.markdown("#### Caveats worth stating out loud")
    st.markdown("""
- **The dataset is a 20-month window** (Jan 2017 – Aug 2018) from one Brazilian
  marketplace. The 2.15% repeat rate is a property of *this* marketplace; do not
  read it as an e-commerce benchmark.
- **The CLV model's holdout has 418 positive examples.** Ranking metrics on that
  base are directional, not precise. AUC 0.607 means "better than chance at
  ordering customers", not "reliable per-customer forecasts".
- **Early stopping used the holdout panel**, so the reported iteration count
  carries a small optimistic bias. It was capped at 31 rounds to bound it.
- **44% of model gain sits in two high-cardinality categorical features**
  (product category, customer state). Held-out AUC says the signal is real, but
  it is not a lot of signal.
    """)
