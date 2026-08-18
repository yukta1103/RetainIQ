"""Capture full-page screenshots of every dashboard page.

Used to generate the README images. Drives the locally-installed Chrome via
Playwright (channel="chrome"), so no separate browser download is needed.

Chrome's own --screenshot flag is not usable here: Streamlit renders over a
websocket after load, and --virtual-time-budget does not wait for it, so the
capture lands on the loading skeleton. This script waits for real content
(a rendered Plotly canvas or a metric value) before shooting.

Usage:
    python scripts/screenshot_dashboard.py [--url http://localhost:8502]
                                           [--out docs/screenshots]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

# (slug, page name, tab to click or None). Tabs matter: the model comparison,
# calibration and ROI content lives behind a tab that is not selected by
# default, so a plain page shot would miss all of it.
PAGES = [
    ("01-overview", "Overview", None),
    ("02-segments", "Customer Segments", None),
    ("03-cohorts", "Cohort Retention", None),
    ("04-repeat-propensity", "Repeat Propensity", None),
    ("05-model-comparison", "Repeat Propensity", "Model comparison"),
    ("06-business-insights", "Business Insights", None),
]

VIEWPORT = {"width": 1500, "height": 1000}
MAX_HEIGHT = 9000  # guard against a runaway measurement


def capture(base_url: str, out_dir: Path, timeout_ms: int = 60_000) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = ctx.new_page()

        for slug, name, tab in PAGES:
            url = f"{base_url}/?page={name.replace(' ', '+')}"
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)

            # Streamlit paints a skeleton first. Wait for the page heading, then
            # for the spinner to clear, then for charts to finish drawing.
            page.wait_for_selector("h1", timeout=timeout_ms)
            page.wait_for_function(
                "() => !document.body.innerText.includes('RUNNING...')",
                timeout=timeout_ms,
            )

            if tab:
                page.wait_for_selector('button[role="tab"]', timeout=timeout_ms)
                page.get_by_role("tab", name=tab).click()
                page.wait_for_timeout(1500)
                page.wait_for_function(
                    "() => !document.body.innerText.includes('RUNNING...')",
                    timeout=timeout_ms,
                )

            try:
                page.wait_for_selector(".js-plotly-plot", timeout=15_000)
                page.wait_for_function(
                    "() => [...document.querySelectorAll('.js-plotly-plot')]"
                    ".every(d => d.querySelector('.main-svg'))",
                    timeout=20_000,
                )
            except Exception:
                pass  # Not every page is guaranteed to hold a Plotly figure.

            # Streamlit scrolls an INNER container, so document.body is only
            # ever one viewport tall and full_page=True would silently crop
            # everything below the fold. Measure the real content height from
            # the scrolling element, then grow the viewport to fit it.
            content_h = page.evaluate(
                "() => { const els = [document.scrollingElement, document.body,"
                " ...document.querySelectorAll('section, [data-testid=\"stMain\"],"
                " [data-testid=\"stAppViewContainer\"], .main')];"
                " return Math.max(...els.filter(Boolean).map(e => e.scrollHeight)); }"
            )
            target_h = min(max(int(content_h) + 120, VIEWPORT["height"]), MAX_HEIGHT)
            page.set_viewport_size({"width": VIEWPORT["width"], "height": target_h})
            page.wait_for_timeout(1200)

            # Scroll through so lazy content and canvas tables paint, then settle.
            page.evaluate(
                "async () => { const el = document.scrollingElement || document.body;"
                " const h = el.scrollHeight;"
                " for (let y = 0; y < h; y += 500) { el.scrollTo(0, y);"
                " await new Promise(r => setTimeout(r, 100)); }"
                " el.scrollTo(0, 0); }"
            )
            page.wait_for_timeout(2500)

            path = out_dir / f"{slug}.png"
            page.screenshot(path=str(path), full_page=True)
            page.set_viewport_size(VIEWPORT)
            kb = path.stat().st_size / 1024
            print(f"  {slug:<24} {kb:>8,.0f} KB   {name}")
            written.append(path)

        ctx.close()
        browser.close()
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8502")
    ap.add_argument("--out", default="docs/screenshots")
    a = ap.parse_args()

    out = Path(a.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out

    print(f"Capturing {len(PAGES)} pages from {a.url} -> {out}\n")
    written = capture(a.url, out)
    print(f"\nWrote {len(written)} screenshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
