from flask import Flask, render_template, request
from transformers import pipeline

# Load fine-tuned models
# emotion_classifier = pipeline("sentiment-analysis")  # Replace with your emotion model
# company_extractor = pipeline("ner")  # Replace with your company extraction model

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]

        # Predict emotion
        emotion_prediction = emotion_classifier(text)[0]

        # Predict company
        company_prediction = company_extractor(text)[0]["entities"]
        company_name = company_prediction[0]["word"] if company_prediction else None

        return render_template("index.html", text=text, emotion=emotion_prediction["label"], company=company_name)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
