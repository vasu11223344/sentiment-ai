from flask import Flask, render_template, request
import joblib
import numpy as np
from textblob import TextBlob

app = Flask(__name__)

# Load pre-trained model and vectorizer
model = joblib.load("model/sentiment3_model.pkl")
vectorizer = joblib.load("model/vectorizer3.pkl")

labels = ["Negative", "Neutral", "Positive"]

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    color = ""
    text = ""
    probs = [0, 0, 0]

    if request.method == "POST":
        text = request.form["text"]

        # Transform text and predict
        vec = vectorizer.transform([text])
        prob_array = model.predict_proba(vec)[0]
        probs = np.round(prob_array, 3).tolist()
        pred_idx = int(np.argmax(probs))
        confidence = probs[pred_idx]

        # Neutral correction logic
        polarity = TextBlob(text).sentiment.polarity
        if (
            confidence < 0.5
            or abs(probs[0] - probs[2]) < 0.15
            or abs(polarity) < 0.15
            or ("not" in text.lower() and ("good" in text.lower() or "bad" in text.lower()))
        ):
            sentiment = "Neutral"
        else:
            sentiment = labels[pred_idx]

        # Set color
        if sentiment == "Positive":
            color = "green"
        elif sentiment == "Neutral":
            color = "gray"
        else:
            color = "red"

    return render_template(
        "index.html",
        sentiment=sentiment,
        text=text,
        color=color,
        probs=probs,
        labels=labels
    )

if __name__ == "__main__":
    app.run(debug=True)