"""Page 5 — Business Insights: the takeaways, each with its supporting chart."""

from __future__ import annotations

import numpy as np
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
        "Five takeaways, each with the evidence behind it. Figures are computed "
        "over the full dataset, not the sidebar filters, so the numbers stay "
        "quotable."
    )

    seg = data.load("segments")
    orders = data.load("orders")
    clv = data.load("clv")
    meta = data.meta()

    # ---------------------------------------------------------------- 1
    mon = seg["monetary"].sort_values(ascending=False)
    top1 = 100 * mon.head(int(len(mon) * 0.01)).sum() / mon.sum()
    top5 = 100 * mon.head(int(len(mon) * 0.05)).sum() / mon.sum()
    top10 = 100 * mon.head(int(len(mon) * 0.10)).sum() / mon.sum()
    top20 = 100 * mon.head(int(len(mon) * 0.20)).sum() / mon.sum()
    champ_pct = 100 * len(seg[seg["segment"] == "Champions"]) / len(seg)

    # Gini on the spend distribution: 0 = every customer spends identically,
    # 1 = one customer holds all revenue. Like the percentiles above, it is a
    # property of the distribution and owes nothing to the segmentation.
    x = np.sort(seg["monetary"].to_numpy())
    n = len(x)
    gini = float((2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n)

    _card(
        "1 &nbsp;·&nbsp; Revenue is <i>less</i> concentrated than the 80/20 rule predicts",
        f"Measured directly from the spend distribution — independent of any "
        f"segmentation — the top <b>1%</b> of customers hold <b>{top1:.1f}%</b> of "
        f"revenue, the top <b>5%</b> hold <b>{top5:.1f}%</b>, the top <b>10%</b> hold "
        f"<b>{top10:.1f}%</b>, and the top <b>20%</b> hold <b>{top20:.1f}%</b>. "
        f"The Pareto expectation is 20% → 80%. This base is materially <b>flatter</b> "
        f"than that, with a <b>Gini coefficient of {gini:.3f}</b> — moderate "
        f"inequality, not the winner-take-most curve the playbook assumes. "
        f"<br><br>The reason is structural: with 97.85% of customers buying exactly "
        f"once, spend is driven almost entirely by <i>basket size</i> rather than by "
        f"purchase count. There is no compounding — the mechanism that normally "
        f"produces Pareto concentration in retail is absent. Only <b>{champ_pct:.2f}%</b> "
        f"of customers qualify as Champions, so there is barely a loyal core to "
        f"concentrate around. "
        f"<br><br><b>So what:</b> a 'focus on your best customers' strategy has a much "
        f"weaker payoff here than the usual playbook assumes. Capturing the top 20% "
        f"still leaves {100-top20:.0f}% of revenue on the table, and those customers "
        f"are not identifiable in advance — they are one-time buyers who happened to "
        f"buy something expensive."
    )
    st.plotly_chart(charts.lorenz_curve(seg["monetary"]), use_container_width=True)
    st.caption(
        "Computed from the raw spend distribution. Nothing here depends on how "
        "customers were segmented, so it cannot restate a segment definition."
    )

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
        f"data it scores <b>AUC 0.467, 95% CI [0.436, 0.497]</b> — the whole interval "
        f"sits <i>below</i> 0.5, so it ranks customers <b>significantly worse than "
        f"random</b>. The cause is precise: Gamma-Gamma was fitted on only <b>1,493</b> "
        f"repeat customers and its <i>q</i> parameter came out at <b>0.497</b>. "
        f"Expected spend is <i>p·v/(q−1)</i>, so any <i>q</i> &lt; 1 makes it "
        f"<b>negative</b> — <b>73,639</b> customers received a negative score. "
        f"BG/NBD's purchase-count half is fine; swapping the broken value model for "
        f"observed AOV recovers AUC to <b>0.555 [0.527, 0.585]</b>. "
        f"<br><br><b>So what:</b> reaching for the standard tool without checking its "
        f"parameter constraints would have shipped a model that ranks customers "
        f"backwards. A gradient-boosted Tweedie ranker reaches <b>0.585 "
        f"[0.557, 0.613]</b> — genuinely better than chance, but only just."
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
        st.plotly_chart(charts.auc_intervals(bundle["evaluation"]),
                        use_container_width=True)
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

    # ---------------------------------------------------------------- 5
    _card(
        "5 &nbsp;·&nbsp; The model works, and it is still barely worth deploying",
        f"Targeting the top-scoring 10% reaches <b>7,513</b> customers, of whom "
        f"<b>72</b> actually returned, carrying <b>R$ 12,939</b> — <b>19.4%</b> of "
        f"holdout revenue at <b>1.94x</b> random. That sounds good until you price "
        f"it. The campaign only earns the <i>incremental</i> revenue it causes; most "
        f"of that R$ 12,939 was going to arrive anyway. At a 10% incremental uplift "
        f"the break-even cost is <b>R$ 0.17 per contact</b>. "
        f"<br><br>Email (R$ 0.02) and push (R$ 0.05) clear that bar. SMS (R$ 0.30) "
        f"needs ~20% uplift to break even. Paid social and direct mail "
        f"<b>destroy value at every plausible uplift</b> — paid retargeting loses "
        f"about R$ 10,000. And the best case is small in absolute terms: roughly "
        f"<b>R$ 1,100</b> net against a <b>R$ 15.4M</b> revenue base, under 0.01%. "
        f"<br><br><b>So what:</b> ship it to the email list, and nowhere else. The "
        f"honest recommendation is not 'deploy a CLV platform' — it is 'add a "
        f"ranked email segment and spend the saved effort on first-purchase "
        f"conversion, where the volume actually is.'"
    )
    if bundle is not None and "roi_channels" in bundle:
        st.dataframe(bundle["roi_channels"], hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("#### Caveats worth stating out loud")
    st.markdown("""
- **The dataset is a 20-month window** (Jan 2017 – Aug 2018) from one Brazilian
  marketplace. The 2.15% repeat rate is a property of *this* marketplace; do not
  read it as an e-commerce benchmark.
- **The holdout has 418 positive examples.** Every metric carries real sampling
  error, which is why nothing here is quoted without a 95% interval. AUC 0.585
  [0.557, 0.613] means "reliably better than chance at ordering customers" — not
  "reliable per-customer spend forecasts".
- **The model does not clearly beat a simple sort on revenue capture.** It wins
  on AUC (0.585 [0.557, 0.613] vs 0.513 [0.483, 0.541] for sorting by past
  spend), but on revenue captured in the top decile the intervals overlap:
  19.4% [14.3, 25.7] vs 15.1% [10.1, 21.1]. Claiming a decisive win there would
  overstate the evidence.
- **Removing the early-stopping leak cost real accuracy.** Selecting boosting
  rounds on the test panel gave AUC 0.608; with a proper validation origin it is
  0.585. That ~0.023 gap was optimism, not skill.
- **Quintile calibration is not monotonic** (Q3 captures more revenue than Q2),
  and the error bars overlap across the middle bins. Only the top quintile is
  clearly separated from the bottom.
- **~44% of model gain sits in two high-cardinality categoricals**
  (product category, customer state). Held-out AUC says the signal is real, but
  it is not a lot of signal.
    """)
