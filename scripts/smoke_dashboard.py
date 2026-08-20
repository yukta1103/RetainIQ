"""Automated smoke test: visit every page, exercise every filter, assert no errors.

Drives the running Streamlit app with Playwright against the locally installed
Chrome. Fails loudly on: a Streamlit exception, a page with zero charts, an
empty chart, or any lingering 'CLV prediction' wording in the UI.

    pip install -r requirements-dev.txt
    python scripts/smoke_dashboard.py [--url http://localhost:8501]
"""

from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

PAGES = [
    ("Overview", 5),
    ("Customer Segments", 4),
    ("Cohort Retention", 2),
    ("Repeat Propensity", 4),
    ("Business Insights", 5),
]

# Wording that must never appear in the UI. The disclaimer sentence is the one
# legitimate use of "CLV" and is allowed through explicitly.
BANNED = ["CLV Prediction", "Predicted 90-day spend", "Highest predicted value",
          "Predicted 90-day CLV", "CLV models"]
ALLOWED_CLV = "not a CLV forecast"

failures: list[str] = []
notes: list[str] = []


def scroll_through(page) -> None:
    """Streamlit renders charts lazily as they enter the viewport.

    Without this, a long page reports far fewer charts than it has -- the
    Business Insights page showed 2 of 5 purely because the rest were below
    the fold. Scroll to the bottom in steps, then back to the top.
    """
    page.evaluate("""() => new Promise(res => {
        let y = 0;
        const step = () => {
            y += 600;
            window.scrollTo(0, y);
            if (y < document.body.scrollHeight) setTimeout(step, 120);
            else { window.scrollTo(0, 0); res(); }
        };
        step();
    })""")
    page.wait_for_timeout(2500)


def check(page, label: str, min_charts: int) -> None:
    page.wait_for_timeout(2500)
    scroll_through(page)

    errs = page.query_selector_all('[data-testid="stException"]')
    if errs:
        failures.append(f"{label}: {len(errs)} Streamlit exception(s)")

    charts = page.evaluate("() => document.querySelectorAll('.js-plotly-plot').length")
    if charts < min_charts:
        failures.append(f"{label}: {charts} charts, expected >= {min_charts}")

    empty = page.evaluate("""() => [...document.querySelectorAll('.js-plotly-plot')]
        .filter(d => !d.data || d.data.length === 0 ||
                     d.data.every(t => (!t.y || t.y.length === 0) &&
                                       (!t.z || t.z.length === 0) &&
                                       (!t.x || t.x.length === 0))).length""")
    if empty:
        failures.append(f"{label}: {empty} chart(s) rendered EMPTY")

    text = page.inner_text("body")
    for phrase in BANNED:
        if phrase in text:
            failures.append(f"{label}: banned wording present -> '{phrase}'")
    stray = text.count("CLV") - text.count(ALLOWED_CLV)
    if stray > 0:
        ctxs = [text[max(0, i - 50):i + 50].replace("\n", " ")
                for i in range(len(text)) if text.startswith("CLV", i)]
        # Legitimate mentions: the disclaimer, and passages describing the
        # actual BG/NBD+Gamma-Gamma CLV model and why it fails here.
        allowed = (ALLOWED_CLV, "textbook CLV model", "CLV platform",
                   "CLV model trained", "probabilistic CLV approach")
        keep = [c for c in ctxs if not any(a in c for a in allowed)]
        if keep:
            failures.append(f"{label}: unexplained 'CLV' -> {keep[:2]}")

    notes.append(f"  {label:<20} charts={charts:<3} exceptions=0  chars={len(text):,}")


def main() -> int:
    url = "http://localhost:8501"
    if "--url" in sys.argv:
        url = sys.argv[sys.argv.index("--url") + 1]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.goto(url, wait_until="networkidle", timeout=60_000)
        page.wait_for_selector('[data-testid="stSidebar"]', timeout=60_000)

        print("PAGE SWEEP")
        for label, min_charts in PAGES:
            page.goto(f"{url}/?page={label.replace(' ', '+')}",
                      wait_until="networkidle", timeout=60_000)
            check(page, label, min_charts)
        print("\n".join(notes))

        print("\nFILTER SWEEP (on Overview)")
        page.goto(f"{url}/?page=Overview", wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(2000)

        for preset in ["Last 3 months", "Last 6 months", "Last 12 months", "All time"]:
            lab = page.query_selector(f'label:has-text("{preset}")')
            if not lab:
                failures.append(f"date preset '{preset}' not found")
                continue
            lab.click()
            page.wait_for_timeout(2600)
            body = page.inner_text("body")
            if page.query_selector_all('[data-testid="stException"]'):
                failures.append(f"date preset '{preset}': exception")
            rev = [l for l in body.split("\n") if l.startswith("R$ ")]
            print(f"  {preset:<16} revenue={rev[0] if rev else 'MISSING':<16}")
            if not rev:
                failures.append(f"date preset '{preset}': no revenue metric rendered")

        print("\nDIMENSION FILTER (state = SP)")
        box = page.query_selector('div[data-testid="stMultiSelect"]')
        if box:
            box.click()
            page.wait_for_timeout(700)
            page.keyboard.type("SP")
            page.wait_for_timeout(900)
            page.keyboard.press("Enter")
            page.wait_for_timeout(3000)
            if page.query_selector_all('[data-testid="stException"]'):
                failures.append("state filter SP: exception")
            body = page.inner_text("body")
            sel = [l for l in body.split("\n") if "orders selected" in l]
            print(f"  {sel[0] if sel else 'MISSING selection caption'}")
            if not sel:
                failures.append("state filter: selection caption missing")
        else:
            failures.append("state multiselect not found")

        print("\nTAB SWEEP (Repeat Propensity)")
        page.goto(f"{url}/?page=Repeat+Propensity", wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(2500)
        tabs = page.query_selector_all('button[role="tab"]')
        print(f"  found {len(tabs)} tabs")
        for i in range(len(tabs)):
            t = page.query_selector_all('button[role="tab"]')[i]
            name = t.inner_text().strip()
            t.click()
            page.wait_for_timeout(2800)
            if page.query_selector_all('[data-testid="stException"]'):
                failures.append(f"tab '{name}': exception")
            n = page.evaluate("() => document.querySelectorAll('.js-plotly-plot').length")
            print(f"  tab {name:<24} charts={n}  exceptions=0")

        browser.close()

    print("\n" + "=" * 70)
    if failures:
        print(f"SMOKE TEST FAILED — {len(failures)} problem(s):")
        for f in failures:
            print(f"  * {f}")
        return 1
    print("SMOKE TEST PASSED — every page, filter and tab exercised, no errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
