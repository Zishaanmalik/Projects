from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- Model Loading ---
MODEL_PATH = "wind_power_prediction.pkl"
NOTEBOOK_PATH = "model_code.ipynb"  # Ensure this file exists for the download route

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}.")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {MODEL_PATH}. Prediction will fail.")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


@app.route("/")
def index():
    """Renders the main prediction dashboard."""
    return render_template("index.html", input_data=None, result=None, error=False)


@app.route("/notebook")
def download_notebook():
    """Route to handle downloading the Jupyter Notebook file."""
    try:
        # Use as_attachment=True to prompt the user to download the file
        return send_file(NOTEBOOK_PATH, as_attachment=True)
    except FileNotFoundError:
        return "Notebook file not found on server.", 404


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests by first checking for JSON (JS fetch) and falling
    back to form data (JS fallback or standard form submit).
    """

    if model is None:
        if request.is_json:
            return jsonify({'error': 'Model is not loaded on the server.'}), 500
        return render_template("index.html", result="Error: Model is not loaded on the server.", error=True)

    try:
        # 1. Determine the data source (JSON or Form)
        if request.is_json:
            # Data sent by JavaScript fetch
            data = request.get_json()
            source = "JSON"
        else:
            # Data sent by form POST fallback
            data = request.form
            source = "Form"

        # NOTE: Keys are expected to be lowercase ('windspeed')

        # 2. Extract and validate all 7 features
        WindSpeed = float(data['windspeed'])
        WindDirection = float(data['winddirection'])
        Temperature = float(data['temperature'])
        Pressure = float(data['pressure'])
        AirDensity = float(data['airdensity'])
        RotorRPM = float(data['rotorrpm'])
        BladePitchAngle = float(data['bladepitchangle'])  # The 7th feature

        # 3. Create the feature vector and predict
        x = [[WindSpeed, WindDirection, Temperature, Pressure, AirDensity, RotorRPM, BladePitchAngle]]
        pred = model.predict(x)[0]

        # 4. Return the appropriate response type
        if source == "JSON":
            # Return JSON for client-side JavaScript to process (preferred path)
            return jsonify({
                'prediction': float(pred),
                'message': 'Prediction successful'
            })
        else:
            # Return rendered template for standard form/fallback path
            return render_template("index.html",
                                   result=f"{pred:.2f} kW",
                                   error=False,
                                   input_data=request.form)

    # --- Error Handling ---
    except KeyError as e:
        error_msg = f"Missing required input: {str(e).replace('KeyError: ', '').strip()}. Did you include all 7 fields?"
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        return render_template("index.html", result=f"Error: {error_msg}", error=True, input_data=request.form)

    except ValueError:
        error_msg = "Invalid input type. Please ensure all fields contain valid numbers."
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        return render_template("index.html", result=f"Error: {error_msg}", error=True, input_data=request.form)

    except Exception as e:
        error_msg = f"An unexpected server error occurred: {e}"
        print(f"Prediction Failure: {error_msg}")
        if request.is_json:
            return jsonify({'error': error_msg}), 500
        return render_template("index.html", result=f"Error: {error_msg}", error=True, input_data=request.form)


if __name__ == "__main__":
    app.run(debug=True)