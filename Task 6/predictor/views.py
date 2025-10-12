import webbrowser
import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render

# Load pre-trained model and transformers
model = joblib.load("predictor/models/best_XGB_model.pkl")
encoders = joblib.load("predictor/models/label_encoders.joblib")
scaler = joblib.load("predictor/models/final_scaler.pkl")

def predict_fare(request):
    if request.method == "POST":
        user_name = request.POST.get('user_name')
        driver_name = request.POST.get('driver_name')

        # Numeric inputs
        jfk_dist = abs(float(request.POST.get('jfk_dist')))
        year = int(request.POST.get('year'))
        hour = int(request.POST.get('hour'))
        month = int(request.POST.get('month'))
        day = abs(int(request.POST.get('day')))
        weekday = int(request.POST.get('weekday'))
        passenger_count = abs(int(request.POST.get('passenger_count')))
        distance = abs(float(request.POST.get('distance')))
        bearing = abs(float(request.POST.get('bearing')))

        # Categorical inputs
        Weather = request.POST.get('Weather')
        Car_Condition = request.POST.get('Car_Condition')
        Traffic_Condition = request.POST.get('Traffic_Condition')

        # Encode
        Weather_encoded = encoders["Weather"].transform([Weather])[0]
        Car_Condition_encoded = encoders["Car Condition"].transform([Car_Condition])[0]
        Traffic_Condition_encoded = encoders["Traffic Condition"].transform([Traffic_Condition])[0]

        # Create dataframe
        input_data = pd.DataFrame([[
            Car_Condition_encoded, Weather_encoded,Traffic_Condition_encoded,passenger_count,
            hour,day,month, weekday,year, jfk_dist, distance, bearing
        ]], columns=[
            'Car Condition', 'Weather', 'Traffic Condition', 'passenger_count',
            'hour', 'day', 'month','weekday','year','jfk_dist','distance','bearing'
        ])


        # Scale numeric features
        input_data[['distance', 'jfk_dist', 'bearing']] = scaler.transform(
            input_data[['distance', 'jfk_dist', 'bearing']]
        )

        # Predict
        fare_pred = round(float(model.predict(input_data)[0]), 2)

        return render(request, "predictor/form.html", {
            "prediction": fare_pred,
            "user_name": user_name,
            "driver_name": driver_name
        })

    return render(request, "predictor/form.html")

# === Open the browser automatically when Django starts ===
def open_browser():
    webbrowser.open("http://127.0.0.1:8000/")

import threading
threading.Timer(1.5, open_browser).start()
