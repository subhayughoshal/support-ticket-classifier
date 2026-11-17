"""
Task 5 - Predict the category of a new ticket.

Usage:
    python src/predict.py              # interactive command-line app (Task 6, Option A)
    python src/predict.py --demo       # run the documented test cases
    python src/predict.py "my text"    # one-off prediction
"""
import sys
from pathlib import Path

import joblib

from preprocessing import clean_text
from responses import suggested_response

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "ticket_classifier.joblib"
LOW_CONFIDENCE = 0.50

# 10 NEW tickets, written separately from the training data (different wording).
TEST_CASES = [
    ("I don't remember my password anymore", "Login Issue"),
    ("My account is blocked and I can't get in", "Login Issue"),
    ("The application shows an error when I press submit", "Application Error"),
    ("Software crashed and I lost my work", "Application Error"),
    ("Need a sales report for the last quarter", "Report"),
    ("Can I get a summary of all pending orders in a sheet", "Report"),
    ("Please change the phone number linked to my profile", "Account Update"),
    ("I moved to a new city, update my address", "Account Update"),
    ("The system is running super slow since morning", "Performance"),
    ("Screens take too long to open", "Performance"),
]


def load_model():
    if not MODEL_PATH.exists():
        sys.exit("Model not found. Run  python src/train.py  first.")
    return joblib.load(MODEL_PATH)


def predict_ticket(model, text: str):
    """Returns (category, confidence between 0 and 1)."""
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Ticket description is empty.")
    probs = model.predict_proba([cleaned])[0]
    i = probs.argmax()
    return model.classes_[i], float(probs[i])


def run_demo(model):
    print(f"{'Test Input':<58}{'Expected':<19}{'Predicted':<19}{'Conf.':<7}OK?")
    print("-" * 108)
    correct = 0
    for text, expected in TEST_CASES:
        pred, conf = predict_ticket(model, text)
        ok = pred == expected
        correct += ok
        print(f"{text:<58}{expected:<19}{pred:<19}{conf:>5.0%}  {'Y' if ok else 'N'}")
    print(f"\n{correct}/{len(TEST_CASES)} correct")


def interactive(model):
    print("=" * 40)
    print(" Customer Support Classifier")
    print("=" * 40)
    print("Type a ticket description (or 'q' to quit).\n")
    while True:
        text = input("Enter Ticket Description:\n> ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        try:
            cat, conf = predict_ticket(model, text)
        except ValueError as e:
            print(f"  {e}\n")
            continue
        print(f"\nPredicted Category: {cat}")
        print(f"Confidence: {conf:.1%}")
        if conf < LOW_CONFIDENCE:
            print("(Low confidence - please review this ticket manually.)")
        print(f"Suggested Response: {suggested_response(cat)}\n")


if __name__ == "__main__":
    mdl = load_model()
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo(mdl)
    elif len(sys.argv) > 1:
        c, p = predict_ticket(mdl, " ".join(sys.argv[1:]))
        print(f"Predicted Category: {c}\nConfidence: {p:.1%}")
    else:
        interactive(mdl)
