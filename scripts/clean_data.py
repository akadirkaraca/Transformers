"""Clean suder.csv and write data/suder_clean.csv.

Cleaning steps:
  1. Drop rows with missing/empty News or Title
  2. Drop duplicate News articles (keep first)
  3. Strip HTML tags
  4. Remove Turkish boilerplate patterns (Haberin Devamı, Kaynak: ...)
  5. Apply existing preprocessing pipeline (URL/email/date/whitespace)
  6. Drop rows where News < 50 chars or Title < 5 chars after cleaning
  7. Report counts at each step

Usage:
    uv run python scripts/clean_data.py
    uv run python scripts/clean_data.py --input data/suder.csv --output data/suder_clean.csv
"""
import argparse
import re

import pandas as pd

from transformer.data.preprocessing import preprocess_article, preprocess_title

# ---- Patterns specific to Turkish news boilerplate ----
_HTML_RE = re.compile(r"<[^>]+>")
_HABERIN_DEVAMI_RE = re.compile(r"haberin devam[ıi]\b.*", re.IGNORECASE)
_KAYNAK_RE = re.compile(r"kaynak\s*:\s*\S+", re.IGNORECASE)
_ISTE_DETAYLAR_RE = re.compile(r"i[sş]te detaylar\b.*", re.IGNORECASE)

MIN_NEWS_LEN = 50   # characters after cleaning
MIN_TITLE_LEN = 5   # characters after cleaning


def clean_text(text: str) -> str:
    """Remove HTML and boilerplate before the standard preprocessing pipeline."""
    if not isinstance(text, str):
        return ""
    text = _HTML_RE.sub(" ", text)
    text = _HABERIN_DEVAMI_RE.sub("", text)
    text = _KAYNAK_RE.sub("", text)
    text = _ISTE_DETAYLAR_RE.sub("", text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/suder.csv")
    parser.add_argument("--output", default="data/suder_clean.csv")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input, index_col=0)
    print(f"  Rows: {len(df):,}")

    # Step 1 — drop missing / empty
    before = len(df)
    df = df.dropna(subset=["News", "Title"])
    df = df[df["News"].str.strip().ne("") & df["Title"].str.strip().ne("")]
    print(f"  After drop missing:     {len(df):,}  (removed {before - len(df):,})")

    # Step 2 — drop duplicate articles
    before = len(df)
    df = df.drop_duplicates(subset=["News"], keep="first")
    print(f"  After drop duplicates:  {len(df):,}  (removed {before - len(df):,})")

    # Step 3+4+5 — clean + preprocess
    print("  Cleaning and preprocessing ...")
    df["News"] = df["News"].apply(lambda t: preprocess_article(clean_text(t)))
    df["Title"] = df["Title"].apply(lambda t: preprocess_title(clean_text(t)))

    # Step 6 — drop too-short rows
    before = len(df)
    df = df[df["News"].str.len().ge(MIN_NEWS_LEN) & df["Title"].str.len().ge(MIN_TITLE_LEN)]
    print(f"  After length filter:    {len(df):,}  (removed {before - len(df):,})")

    # Save
    df = df.reset_index(drop=True)
    df.to_csv(args.output)
    print(f"\nSaved {len(df):,} rows → {args.output}")


if __name__ == "__main__":
    main()
