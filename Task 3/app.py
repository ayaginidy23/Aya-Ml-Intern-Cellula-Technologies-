from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# ---------------- Load Models & Preprocessor ----------------
logistic_model = joblib.load("logistic_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Keep the selected features in correct order
selected_features = [
    "average price",
    "total_guests",
    "total_nights",
    "room type",
    "type of meal",
    "market segment type",
    "month",
    "year"
]

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs from form
        avg_price = float(request.form["average_price"])
        total_guests = int(request.form["total_guests"])
        total_nights = int(request.form["total_nights"])
        room_type = request.form["room_type"]
        meal_type = request.form["type_of_meal"]
        market_segment = request.form["market_segment_type"]
        month = int(request.form["month"])
        year = int(request.form["year"])
        chosen_model = request.form["model_choice"]

        # Create DataFrame with the same feature names
        input_data = pd.DataFrame([[
            avg_price, total_guests, total_nights, room_type,
            meal_type, market_segment, month, year
        ]], columns=selected_features)

        # Preprocess
        input_transformed = preprocessor.transform(input_data)

        # Choose model
        if chosen_model == "logistic":
            prediction = logistic_model.predict(input_transformed)[0]
        else:
            prediction = rf_model.predict(input_transformed)[0]

        # Convert prediction to label
        result = "Canceled" if prediction == 1 else "Not Canceled"

        return render_template("index.html", prediction_text=f"Booking Status Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
