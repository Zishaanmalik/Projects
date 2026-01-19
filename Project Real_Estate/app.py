from flask import Flask, render_template, request, jsonify
from pathlib import Path
import joblib, json, numpy as np

BASE = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE / "templates"))

COLUMNS_PATH = BASE / "columns.json"
MODEL_PATH = BASE / "real_estate.pkl"

def load_columns(p):
    if not p.exists(): return None
    data = json.load(p.open("r", encoding="utf-8"))
    if isinstance(data, dict):
        for k in ("columns","cols","column_names","features"):
            if k in data and isinstance(data[k], list): return data[k]
        for v in data.values():
            if isinstance(v, list): return v
        return None
    if isinstance(data, list): return data
    return None

columns = load_columns(COLUMNS_PATH) or []
model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(str(MODEL_PATH))
    except Exception:
        model = None

def col_index(name):
    if not columns: return None
    try: return columns.index(name)
    except ValueError: pass
    lname = name.lower()
    for i,c in enumerate(columns):
        if c.lower() == lname: return i
    return None

@app.route("/")
def index():
    locs = []
    base = {"tos","total_sqft","total_sq_ft","bhk","bathroom","balcony","bedrooms"}
    for c in columns:
        if c.lower() not in base: locs.append(c)
    return render_template("index.html", locations=locs, columns=columns, result=None, error=False)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        if request.is_json: return jsonify({"error":"Model not loaded."}), 500
        return render_template("index.html", result="Model not loaded.", error=True, columns=columns)
    data = request.get_json() if request.is_json else request.form
    def get(k): return data.get(k) if data is not None else None
    location = (get("location") or "").strip()
    try:
        tos = float(get("tos"))
        bhk = float(get("bhk"))
        bathroom = float(get("bathroom"))
        balcony = float(get("balcony"))
    except Exception:
        if request.is_json: return jsonify({"error":"Invalid numeric input."}), 400
        return render_template("index.html", result="Invalid numeric input.", error=True, columns=columns, input_data=request.form)
    xi = np.zeros(len(columns), dtype=float)
    for name in ("tos","total_sqft","total_sq_ft","TOS"):
        i = col_index(name)
        if i is not None:
            xi[i] = tos; break
    for name,val in (("bhk",bhk),("bedrooms",bhk)):
        i = col_index(name)
        if i is not None:
            xi[i] = val; break
    for name,val in (("bathroom",bathroom),("bathrooms",bathroom),("bath",bathroom)):
        i = col_index(name)
        if i is not None:
            xi[i] = val; break
    for name,val in (("balcony",balcony),("balconies",balcony)):
        i = col_index(name)
        if i is not None:
            xi[i] = val; break
    loc_assigned = False
    if location:
        for i,c in enumerate(columns):
            if c.lower() == location.lower():
                xi[i] = 1; loc_assigned = True; break
    try:
        pred = float(model.predict([xi])[0])
    except Exception as e:
        if request.is_json: return jsonify({"error":f"Prediction failed: {e}"}), 500
        return render_template("index.html", result=f"Prediction failed: {e}", error=True, columns=columns, input_data=request.form)
    if request.is_json:
        return jsonify({"prediction":pred,"location_assigned":loc_assigned})
    return render_template("index.html", result=f"{pred:.2f}", error=False, columns=columns, input_data=request.form)

if __name__ == "__main__":
    app.run(debug=True)
