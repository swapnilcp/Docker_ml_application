from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return "ML Application is Running!"


@app.route("/predict", methods=["GET"])
def predict():

    experience = float(request.args.get("experience"))

    prediction = model.predict([[experience]])

    return jsonify({
        "YearsExperience": experience,
        "PredictedSalary": round(prediction[0], 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)