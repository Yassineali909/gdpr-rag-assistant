"""
Download the FULL official GDPR text and save it as data/full_gdpr.txt.

The GDPR (Regulation (EU) 2016/679) is published on EUR-Lex and is free to reuse.
This script fetches the HTML version and strips it down to clean text.

After it runs, point config.SOURCE_FILE to the downloaded file:

    SOURCE_FILE = DATA_DIR / "full_gdpr.txt"

then rebuild the index:  python build_index.py
"""
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# EUR-Lex HTML version of Regulation (EU) 2016/679 (GDPR), English.
GDPR_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
    "?uri=CELEX:32016R0679"
)

OUT_FILE = Path(__file__).parent.parent / "data" / "full_gdpr.txt"


def download() -> str:
    print(f"[download] Fetching GDPR from EUR-Lex ...")
    headers = {"User-Agent": "Mozilla/5.0 (compliance-rag project)"}
    resp = requests.get(GDPR_URL, headers=headers, timeout=60)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Drop scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def main():
    try:
        html = download()
    except Exception as e:
        print(f"[download] FAILED: {e}")
        print(
            "\nIf your network blocks EUR-Lex, download the GDPR HTML manually from:\n"
            f"  {GDPR_URL}\n"
            "save it, and run clean_html() on it, or just paste the text into "
            "data/full_gdpr.txt yourself."
        )
        sys.exit(1)

    text = clean_html(html)
    OUT_FILE.write_text(text, encoding="utf-8")
    print(f"[download] Saved {len(text):,} characters -> {OUT_FILE}")
    print("\nNext:")
    print("  1. In config.py set: SOURCE_FILE = DATA_DIR / 'full_gdpr.txt'")
    print("  2. Run: python build_index.py")


if __name__ == "__main__":
    main()
