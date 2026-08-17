"""Chart palette and a shared Plotly template.

Palette values come from the validated reference instance. The three
categorical slots used here cleared the all-pairs gate on the light surface
(worst CVD dE 9.2, worst normal-vision dE 24.0). AQUA carries a contrast WARN
(2.74:1), so it is only used where direct labels or a table view provide
relief -- never as the sole carrier of meaning.

The app is pinned to light mode in .streamlit/config.toml, so only the light
steps are defined; a dark set would need its own validation run rather than
an automatic flip.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --- Categorical slots (fixed order, never cycled) ------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
VIOLET = "#4a3aa7"

CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]

# --- Sequential ramp (single hue, light -> dark) --------------------------
SEQUENTIAL = [
    [0.00, "#cde2fb"],
    [0.17, "#9ec5f4"],
    [0.33, "#6da7ec"],
    [0.50, "#3987e5"],
    [0.67, "#256abf"],
    [0.83, "#184f95"],
    [1.00, "#0d366b"],
]

# --- Status (reserved; never reused as a series) --------------------------
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

# --- Chrome and ink -------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def register_template() -> None:
    """Install 'retainiq' as the default Plotly template."""
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=13, color=INK_SECONDARY),
        title=dict(font=dict(size=15, color=INK_PRIMARY), x=0, xanchor="left", pad=dict(b=12)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=CATEGORICAL,
        # Recessive chrome: hairline horizontal grid only, no vertical clutter.
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor=BASELINE,
            linewidth=1,
            ticks="outside",
            tickcolor=BASELINE,
            ticklen=4,
            tickfont=dict(size=12, color=INK_MUTED),
            automargin=True,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRIDLINE,
            gridwidth=1,
            zeroline=False,
            showline=False,
            tickfont=dict(size=12, color=INK_MUTED),
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12, color=INK_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=8, r=8, t=48, b=8),
        hoverlabel=dict(
            font=dict(family=FONT, size=12),
            bgcolor="#ffffff",
            bordercolor=BASELINE,
        ),
        hovermode="x unified",
    )
    pio.templates["retainiq"] = t
    pio.templates.default = "retainiq"


# Segment colours: semantic, not rank-based, so a filter that removes segments
# never repaints the survivors. Good/at-risk/lost read as status-like without
# borrowing the reserved status hexes for series duty.
SEGMENT_COLORS = {
    "Champions": BLUE,
    "Loyal": "#3987e5",
    "Can't Lose": VIOLET,
    "At Risk": ORANGE,
    "Big Spender (One-Off)": AQUA,
    "New": "#6da7ec",
    "Promising": YELLOW,
    "Hibernating": "#9ec5f4",
    "Lost": INK_MUTED,
    "Needs Attention": BASELINE,
}

CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
  [data-testid="stMetricValue"] {{ font-size: 1.75rem; color: {INK_PRIMARY}; }}
  [data-testid="stMetricLabel"] {{ color: {INK_MUTED}; }}
  .kpi-note {{ color: {INK_MUTED}; font-size: 0.82rem; margin-top: -0.5rem; }}
  .insight-card {{
      background: {SURFACE};
      border: 1px solid rgba(11,11,11,0.10);
      border-radius: 10px;
      padding: 1.1rem 1.3rem;
      margin-bottom: 1rem;
  }}
  .insight-card h4 {{ margin: 0 0 .4rem 0; color: {INK_PRIMARY}; font-size: 1.02rem; }}
  .insight-card p {{ margin: 0; color: {INK_SECONDARY}; font-size: 0.92rem; line-height: 1.5; }}
  .caveat {{
      border-left: 3px solid {WARNING};
      padding: .55rem .85rem; margin: .5rem 0 1rem 0;
      background: rgba(250,178,25,0.07);
      color: {INK_SECONDARY}; font-size: 0.87rem;
  }}
</style>
"""
