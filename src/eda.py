"""
Task 2 - Exploratory Data Analysis. Saves charts into screenshots/.

Run:  python src/eda.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import prepare_data

OUT = Path(__file__).resolve().parent.parent / "screenshots"
OUT.mkdir(exist_ok=True)


def main():
    df = prepare_data(verbose=False)
    sns.set_theme(style="whitegrid")

    print(f"Total tickets: {len(df)}")
    for col in ["category", "priority", "status"]:
        print(f"\nTickets by {col}:")
        print(df[col].value_counts().to_string())

    # Chart 1 - category distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = df["category"].value_counts().index
    sns.countplot(data=df, y="category", order=order, hue="category", legend=False, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, padding=3)
    ax.set_title("Chart 1 - Ticket count per category")
    ax.set_xlabel("Number of tickets"); ax.set_ylabel("")
    plt.tight_layout(); plt.savefig(OUT / "analysis_category.png", dpi=150); plt.close()

    # Chart 2 - priority by category (stacked)
    ct = df[df["priority"] != "Unknown"].groupby(["category", "priority"]).size().unstack(fill_value=0)
    ct = ct[[c for c in ["High", "Medium", "Low"] if c in ct.columns]]
    ax = ct.plot(kind="bar", stacked=True, figsize=(8, 4.5), color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_title("Chart 2 - Priority distribution within each category")
    ax.set_xlabel(""); ax.set_ylabel("Number of tickets")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); plt.savefig(OUT / "analysis_priority.png", dpi=150); plt.close()

    # Chart 3 - status distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    order = df["status"].value_counts().index
    sns.countplot(data=df, x="status", order=order, hue="status", legend=False, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, padding=3)
    ax.set_title("Chart 3 - Ticket status")
    plt.tight_layout(); plt.savefig(OUT / "analysis_status.png", dpi=150); plt.close()

    # Chart 4 - ticket length by category
    df["word_count"] = df["clean_description"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x="category", y="word_count", hue="category", legend=False, ax=ax)
    ax.set_title("Chart 4 - Words per ticket description")
    ax.set_xlabel(""); ax.set_ylabel("Words")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); plt.savefig(OUT / "analysis_length.png", dpi=150); plt.close()

    print("\nAverage description length (words):", round(df["word_count"].mean(), 1))
    print("\nCharts saved to", OUT)


if __name__ == "__main__":
    main()
