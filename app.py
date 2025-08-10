from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Load trained model and preprocessing pipeline
model = joblib.load("model.pkl")

app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Ensure required fields are present
        expected_fields = [
            "age", "job", "marital", "education", "default",
            "balance", "housing", "loan", "contact", "month", "day_of_week"
        ]
        missing = [f for f in expected_fields if f not in data or data[f] == ""]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        # Convert single JSON to DataFrame
        df = pd.DataFrame([data])

        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Probability for "yes"

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "✅ Bank Marketing Prediction API is running!"


if __name__ == "__main__":
    app.run(debug=True)


