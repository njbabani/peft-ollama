"""Screenshot the running Gradio demo for the README.

    uv run groundedqa demo --run-dir runs/full --ollama     # in another terminal
    uv run python scripts/capture_demo.py

Uses the locally installed Google Chrome through Playwright (no browser download).
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860/?__theme=light"
QUESTIONS = {
    "demo-answerable.png": "What percentage of the Portuguese people are Roman Catholic?",
    "demo-unanswerable.png": "What belonging to Henry VI is in the museum?",
}


def main(out_dir: Path = Path("docs/assets")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1000}, device_scale_factor=2)
        page.goto(URL, wait_until="networkidle")
        for name, question in QUESTIONS.items():
            page.get_by_label("Question").fill(question)
            page.get_by_role("button", name="Ask").click()
            page.wait_for_function("() => document.body.innerText.includes('P(no answer)')", timeout=180_000)
            page.wait_for_timeout(500)
            page.screenshot(path=str(out_dir / name), full_page=True)
            print("wrote", out_dir / name)
            page.goto(URL, wait_until="networkidle")
        browser.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/assets"))
