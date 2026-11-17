"""
Tasks 3 & 4 - Model building, training and testing.

Pipeline:  cleaned text -> TF-IDF (word 1-2 grams) -> classifier -> category

Three simple models are compared with 5-fold cross-validation on the TRAINING
data only. The best one is refit and evaluated ONCE on the held-out test set.

Run:  python src/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import prepare_data

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "ticket_classifier.joblib"
RANDOM_STATE = 42


def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("clf", clf),
    ])


def main():
    df = prepare_data(verbose=False)
    X, y = df["clean_description"], df["category"]

    # 80/20 split; stratify keeps category proportions equal in both parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    candidates = {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=10),
        "Linear SVM": LinearSVC(C=1.0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    print("\n5-fold cross-validation on TRAINING data (accuracy):")
    for name, clf in candidates.items():
        scores = cross_val_score(make_pipeline(clf), X_train, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = (scores.mean(), scores.std())
        print(f"  {name:<20} {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Choose Logistic Regression: competitive accuracy AND it gives calibrated-ish
    # probabilities (needed for the "confidence" shown in the app).
    # If another model is clearly better in CV we would switch - see report.
    best_name = "Logistic Regression"
    print(f"\nSelected model: {best_name}")
    model = make_pipeline(candidates[best_name])
    model.fit(X_train, y_train)

    # ---- Evaluate on the unseen test set ----
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {p:.3f} (weighted)")
    print(f"Recall   : {r:.3f} (weighted)")
    print(f"F1 score : {f1:.3f} (weighted)\n")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(labels)
    print(cm)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False)
    ax.set_title("Confusion matrix - test set")
    plt.tight_layout()
    (ROOT / "screenshots").mkdir(exist_ok=True)
    plt.savefig(ROOT / "screenshots" / "confusion_matrix.png", dpi=150)
    plt.close()

    # Show the misclassified tickets (useful for error analysis & interview)
    wrong = [(t, a, pr) for t, a, pr in zip(X_test, y_test, y_pred) if a != pr]
    print(f"\nMisclassified test tickets: {len(wrong)}")
    for t, a, pr in wrong:
        print(f"  '{t}'\n     actual={a} | predicted={pr}")

    # Top words per category (explains HOW the model decides)
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    feats = np.array(tfidf.get_feature_names_out())
    print("\nTop 6 features per category:")
    top = {}
    for i, cls in enumerate(clf.classes_):
        idx = np.argsort(clf.coef_[i])[-6:][::-1]
        top[cls] = feats[idx].tolist()
        print(f"  {cls:<18} {', '.join(top[cls])}")

    # ---- Save everything ----
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics = {
        "train_size": len(X_train), "test_size": len(X_test),
        "cv_accuracy": {k: {"mean": round(v[0], 4), "std": round(v[1], 4)} for k, v in cv_results.items()},
        "selected_model": best_name,
        "test_accuracy": round(acc, 4), "test_precision_weighted": round(p, 4),
        "test_recall_weighted": round(r, 4), "test_f1_weighted": round(f1, 4),
        "labels": labels, "confusion_matrix": cm.tolist(),
        "misclassified": [{"text": t, "actual": a, "predicted": pr} for t, a, pr in wrong],
        "top_features": top,
    }
    (ROOT / "model" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ROOT / "model" / "classification_report.txt").write_text(report)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
