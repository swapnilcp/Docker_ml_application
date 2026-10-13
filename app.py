
from flask import Flask, request, jsonify # type: ignore
from prometheus_client import Counter, generate_latest # type: ignore
from prometheus_client import CONTENT_TYPE_LATEST # type: ignore
import pickle

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "ml_app_requests_total",
    "Total HTTP requests received by the ML application"
)

PREDICTION_COUNT = Counter(
    "ml_app_predictions_total",
    "Total successful salary predictions"
)

# Load the trained ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.before_request
def count_request():
    REQUEST_COUNT.inc()


@app.route("/")
def home():
    return "ML Application is Running!"


@app.route("/predict", methods=["GET"])
def predict():
    try:
        experience = float(request.args.get("experience"))

        prediction = model.predict([[experience]])

        PREDICTION_COUNT.inc()

        return jsonify({
            "YearsExperience": experience,
            "PredictedSalary": round(float(prediction[0]), 2)
        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "Provide a valid numeric experience parameter"
        }), 400


@app.route("/metrics")
def metrics():
    return (
        generate_latest(),
        200,
        {"Content-Type": CONTENT_TYPE_LATEST}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
