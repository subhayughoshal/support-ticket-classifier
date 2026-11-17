"""
Task 6 (Option B) + Bonus - Simple Flask web interface.

Run:  python src/app.py     then open http://127.0.0.1:5000
"""
from flask import Flask, render_template_string, request

from predict import LOW_CONFIDENCE, load_model, predict_ticket
from responses import suggested_response

app = Flask(__name__)
model = load_model()

PAGE = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Customer Support Classifier</title>
<style>
  body{font-family:Segoe UI,Arial,sans-serif;background:#f4f6f8;margin:0;padding:40px 16px}
  .card{max-width:640px;margin:auto;background:#fff;padding:28px;border-radius:10px;box-shadow:0 2px 10px #0001}
  h1{margin-top:0;font-size:1.5rem}
  textarea{width:100%;height:110px;padding:10px;font-size:1rem;box-sizing:border-box}
  button{margin-top:12px;padding:10px 24px;font-size:1rem;background:#1f6feb;color:#fff;border:0;border-radius:6px;cursor:pointer}
  .res{margin-top:22px;padding:16px;background:#eef5ff;border-left:4px solid #1f6feb}
  .warn{color:#b35900;font-size:.9rem}.err{color:#c00}
</style></head><body><div class="card">
<h1>Customer Support Classifier</h1>
<form method="post">
  <label for="t">Ticket Description:</label>
  <textarea id="t" name="ticket" placeholder="e.g. The application is very slow today">{{ text }}</textarea>
  <button type="submit">Predict</button>
</form>
{% if error %}<p class="err">{{ error }}</p>{% endif %}
{% if category %}
<div class="res">
  <p><b>Predicted Category:</b> {{ category }}</p>
  <p><b>Confidence:</b> {{ "%.1f"|format(conf*100) }}%</p>
  {% if conf < low %}<p class="warn">Low confidence - please review this ticket manually.</p>{% endif %}
  <p><b>Suggested Response:</b> {{ reply }}</p>
</div>
{% endif %}
</div></body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = dict(text="", category=None, conf=0, reply="", error=None, low=LOW_CONFIDENCE)
    if request.method == "POST":
        text = request.form.get("ticket", "").strip()
        ctx["text"] = text
        try:
            cat, conf = predict_ticket(model, text)
            ctx.update(category=cat, conf=conf, reply=suggested_response(cat))
        except ValueError:
            ctx["error"] = "Please enter a ticket description."
    return render_template_string(PAGE, **ctx)


if __name__ == "__main__":
    app.run(debug=False)
