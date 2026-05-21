from flask import Flask, render_template, request

import pandas as pd

import joblib

import random

app = Flask(__name__)

# Load Model
model = joblib.load("maintenance_model.pkl")

# Load Bus Data
bus_data = pd.read_csv("bus_data.csv")


# Home Page
@app.route('/')
def home():

    return render_template(

        'index.html',

        buses=bus_data.to_dict(orient='records')

    )


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    # Bus Information
    driver_name = request.form['driver_name']

    contact = request.form['contact']

    bus_no = request.form['bus_no']

    source = request.form['source']

    destination = request.form['destination']

    # Sensor Inputs
    distance = float(request.form['distance'])

    temperature = float(request.form['temperature'])

    oil_level = float(request.form['oil_level'])

    vibration = float(request.form['vibration'])

    # Create DataFrame
    new_data = pd.DataFrame(

        [[

            distance,

            temperature,

            oil_level,

            vibration

        ]],

        columns=[

            'Distance',

            'Temperature',

            'Oil_Level',

            'Vibration'

        ]

    )

    # Model Prediction
    result = model.predict(new_data)

    # SMART RISK SCORE
    temp_score = (temperature / 120) * 35

    oil_score = ((100 - oil_level) / 100) * 30

    vibration_score = (vibration / 10) * 25

    distance_score = (distance / 30000) * 10

    probability = (

        temp_score +

        oil_score +

        vibration_score +

        distance_score

    )

    # Limit Risk Score
    if probability > 100:

        probability = 100

    if probability < 0:

        probability = 0

    # Prediction Status
    if probability >= 70:

        prediction = "Maintenance Required"

    else:

        prediction = "Bus Running Normally"

    # Emergency Alert
    if (

        probability >= 70

        or temperature > 110

        or vibration > 8

        or oil_level < 30

    ):

        alert = "⚠ Emergency Maintenance Needed!"

    else:

        alert = "✅ Bus Condition Stable"

    # Priority
    if probability < 40:

        priority = "LOW"

    elif probability < 70:

        priority = "MEDIUM"

    else:

        priority = "HIGH"

    # Health Score
    health_score = 100 - probability

    # Breakdown Chance
    breakdown = random.randint(

        max(1, int(probability - 10)),

        int(probability)

    )

    # Safe Distance
    if probability >= 70:

        safe_distance = random.randint(50, 120)

    elif probability >= 40:

        safe_distance = random.randint(120, 300)

    else:

        safe_distance = random.randint(300, 700)

    # AI Suggestions
    suggestions = []

    if temperature > 110:

        suggestions.append(

            "🔥 Check Engine Cooling System"

        )

    if vibration > 8:

        suggestions.append(

            "⚙ Inspect Engine Vibration"

        )

    if oil_level < 30:

        suggestions.append(

            "🛢 Refill Engine Oil"

        )

    if distance > 20000:

        suggestions.append(

            "🚌 Schedule Full Bus Service"

        )

    if probability >= 70:

        suggestions.append(

            "🚨 Immediate Maintenance Required"

        )

    if len(suggestions) == 0:

        suggestions.append(

            "✅ Bus Running Smoothly"

        )

    # Google Maps Link
    service_center = "https://www.google.com/maps/search/bus+service+center+near+me"

    # Send Data To HTML
    return render_template(

        'result.html',

        prediction=prediction,

        probability=round(probability, 2),

        alert=alert,

        priority=priority,

        health_score=round(health_score, 2),

        breakdown=breakdown,

        safe_distance=safe_distance,

        suggestions=suggestions,

        service_center=service_center,

        driver_name=driver_name,

        contact=contact,

        bus_no=bus_no,

        source=source,

        destination=destination,

        temperature=temperature,

        oil_level=oil_level,

        vibration=vibration

    )


# Run Flask App
if __name__ == '__main__':

    app.run(debug=True)