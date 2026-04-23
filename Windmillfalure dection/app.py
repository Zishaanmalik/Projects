from flask import Flask, render_template, request, redirect
import numpy as np
import joblib
import csv
import os

app = Flask(__name__)

model = joblib.load("model.pkl")

HISTORY_FILE = "history.csv"


def save_history(values, prediction):
    write_header = not os.path.exists(HISTORY_FILE)

    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Rotor", "Wind", "Power", "Gearbox", "Bearing",
                "Vibration", "Ambient", "Humidity", "Result"
            ])
        writer.writerow(values + [prediction])


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return list(reader)


@app.route("/")
def home():
    return redirect("/input")


@app.route("/input")
def input_form():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        rotor = float(request.form["rotor"])
        wind = float(request.form["wind"])
        power = float(request.form["power"])
        gearbox = float(request.form["gearbox"])
        bearing = float(request.form["bearing"])
        vibration = float(request.form["vibration"])
        ambient = float(request.form["ambient"])
        humidity = float(request.form["humidity"])
    except:
        return "Invalid Input"

    features = np.array([[rotor, wind, power, gearbox, bearing, vibration, ambient, humidity]])
    pred = int(model.predict(features)[0])

    save_history([rotor, wind, power, gearbox, bearing, vibration, ambient, humidity], pred)

    return render_template("result.html", result=pred)


@app.route("/dashboard")
def dashboard():
    history = load_history()
    return render_template("dashboard.html", history=history)


if __name__ == "__main__":
    app.run(debug=True)
