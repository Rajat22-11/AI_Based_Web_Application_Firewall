from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import os

app = Flask(__name__)

# Configure CORS to allow requests from your frontend origin
CORS(app, resources={r"/predict*": {"origins": "http://localhost:5173"}})

# Paths to your models and vectorizers
SQLI_MODEL_PATH = os.path.join("models", "sqli_model.pkl")
SQLI_VECTORIZER_PATH = os.path.join("models", "sqli_vectorizer.pkl")
XSS_MODEL_PATH = os.path.join("models", "xss_model.pkl")
XSS_VECTORIZER_PATH = os.path.join("models", "xss_vectorizer.pkl")

# Load SQLi model and vectorizer
with open(SQLI_MODEL_PATH, "rb") as file:
    sqli_model = pickle.load(file)

with open(SQLI_VECTORIZER_PATH, "rb") as file:
    sqli_vectorizer = pickle.load(file)

# Load XSS model and vectorizer
with open(XSS_MODEL_PATH, "rb") as file:
    xss_model = pickle.load(file)

with open(XSS_VECTORIZER_PATH, "rb") as file:
    xss_vectorizer = pickle.load(file)


def preprocess_query(query):
    """Preprocess the query by lowering case and removing special characters."""
    query = query.lower()
    query = re.sub(r"[<>;'\"/]", " ", query)  # Remove specific characters
    query = re.sub(r"\s+", " ", query).strip()  # Remove extra whitespace
    return query


@app.route("/predict/sqli", methods=["POST", "OPTIONS"])
def predict_sqli():
    if request.method == "OPTIONS":
        # CORS preflight request
        return jsonify({"status": "OK"}), 200

    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "No query provided."}), 400

        query = data["query"]
        query_processed = preprocess_query(query)
        query_transformed = sqli_vectorizer.transform([query_processed])

        sqli_prediction = sqli_model.predict(query_transformed)
        prediction = int(sqli_prediction[0])

        result = {
            "Prediction": prediction,
            "Type": "SQL Injection" if prediction == 1 else "Safe",
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/xss", methods=["POST", "OPTIONS"])
def predict_xss():
    if request.method == "OPTIONS":
        # CORS preflight request
        return jsonify({"status": "OK"}), 200

    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "No query provided."}), 400

        query = data["query"]
        query_processed = preprocess_query(query)
        query_transformed = xss_vectorizer.transform([query_processed])

        xss_prediction = xss_model.predict(query_transformed)
        prediction = int(xss_prediction[0])

        result = {
            "Prediction": prediction,
            "Type": "XSS Attack" if prediction == 1 else "Safe",
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running."}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
