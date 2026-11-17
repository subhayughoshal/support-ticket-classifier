"""
Task 1 - Data preparation.

Loads the CSV, reports data quality, removes duplicates, handles missing
values and cleans the ticket text. Also imported by train.py / predict.py so
the SAME cleaning is applied at training time and prediction time.
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "tickets.csv"


def clean_text(text: str) -> str:
    """Lowercase -> strip punctuation/digits -> collapse spaces.

    Stop-words are deliberately NOT removed: words like "not", "cannot",
    "unable" carry meaning in support tickets (e.g. "cannot login").
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # keep letters only
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_data(path: Path = DATA_PATH, verbose: bool = True) -> pd.DataFrame:
    df = load_raw(path)

    if verbose:
        print("=" * 60)
        print("RAW DATA (first 5 rows)")
        print("=" * 60)
        print(df.head().to_string())
        print(f"\nShape: {df.shape}")
        print("\nMissing values per column:")
        print(df.isna().sum().to_string())
        print(f"\nExact duplicate rows (all columns): {df.duplicated().sum()}")
        print(f"Duplicate descriptions (ignoring ticket_id, case, spacing): "
              f"{df['ticket_description'].astype(str).str.lower().str.strip().duplicated().sum()}")

    n_start = len(df)

    # 1) Rows with no description (or no category) cannot be used for a text
    #    classifier -> drop them.
    df = df.dropna(subset=["ticket_description", "category"]).copy()
    n_after_missing = len(df)

    # 2) Missing priority/status are not used by the model -> fill, don't drop.
    df["priority"] = df["priority"].fillna("Unknown")
    df["status"] = df["status"].fillna("Unknown")

    # 3) Text cleaning
    df["clean_description"] = df["ticket_description"].apply(clean_text)

    # 4) Remove duplicates AFTER cleaning, so "Hi, I forgot my password!" and
    #    "hi i forgot my password" count as the same ticket.
    df = df.drop_duplicates(subset=["clean_description"], keep="first")
    n_after_dups = len(df)

    df = df.reset_index(drop=True)

    if verbose:
        print("\n" + "=" * 60)
        print("CLEANING SUMMARY")
        print("=" * 60)
        print(f"Rows at start                         : {n_start}")
        print(f"Dropped (missing description/category): {n_start - n_after_missing}")
        print(f"Dropped (duplicate descriptions)      : {n_after_missing - n_after_dups}")
        print(f"Rows after cleaning                   : {len(df)}")
        print("\nSample of cleaned text:")
        print(df[["ticket_description", "clean_description"]].head(5).to_string())
        print("\nDataset info:")
        df.info()
    return df


if __name__ == "__main__":
    prepare_data()
