import numpy as np
import joblib
import streamlit as st

# ---------- CONFIG ----------
st.set_page_config(
    page_title="AI-Powered Adaptive Renewable Energy Predictor",
    page_icon="⚡",
    layout="wide",
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    /* Global background gradient */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 40%, #000000 100%);
        color: #e5e7eb;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Main title styling */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        background: linear-gradient(90deg, #22d3ee, #a855f7, #f97316);
        -webkit-background-clip: text;
        color: transparent;
    }

    .subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
        margin-top: -8px;
    }

    /* Glassmorphism cards */
    .glass-card {
        padding: 1.2rem 1.4rem;
        border-radius: 1.3rem;
        background: rgba(15, 23, 42, 0.80);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(18px);
    }

    .prediction-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #bbf7d0;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #9ca3af;
    }

    .help-text {
        font-size: 0.75rem;
        color: #9ca3af;
    }

    .footer-note {
        font-size: 0.75rem;
        color: #6b7280;
        text-align: right;
        margin-top: 0.8rem;
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, #22d3ee, #a855f7);
    }

    </style>
""", unsafe_allow_html=True)

# ---------- MODEL LOADING ----------
@st.cache_resource
def load_model(path: str = "system_production_model.pkl"):
    model = joblib.load(path)
    return model


model = load_model()

# ---------- HEADER ----------
st.markdown('<div class="main-title">⚡ AI-Powered Adaptive Renewable Energy Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict system production in real-time using weather & environmental conditions.</div>',
    unsafe_allow_html=True
)

st.markdown("")  # small spacing

# ---------- LAYOUT ----------
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Input Environment Parameters")

    st.caption("Tune the sliders to simulate different weather conditions and see the impact on predicted system production.")

    # You might need to adjust these ranges to match your dataset
    date_hour = st.number_input(
        "Date-Hour (numeric)",
        help="Use the same numeric encoding that was used while training the model (e.g., normalized timestamp / NMT feature).",
        min_value=0.0,
        max_value=100000.0,
        value=1000.0,
        step=1.0
    )

    wind_speed = st.slider(
        "Wind Speed (m/s)",
        min_value=0.0,
        max_value=30.0,
        value=8.0,
        step=0.1,
        help="Higher wind speed usually boosts wind-turbine-based production."
    )

    sunshine = st.slider(
        "Sunshine Duration (hours)",
        min_value=0.0,
        max_value=12.0,
        value=6.0,
        step=0.1,
        help="Total sunshine hours in the period of interest."
    )

    air_pressure = st.slider(
        "Air Pressure (hPa)",
        min_value=930.0,
        max_value=1050.0,
        value=1013.0,
        step=0.5,
        help="Typical sea-level pressure is around 1013 hPa."
    )

    radiation = st.slider(
        "Solar Radiation (W/m²)",
        min_value=0.0,
        max_value=1200.0,
        value=600.0,
        step=5.0,
        help="Instantaneous global horizontal irradiance."
    )

    air_temperature = st.slider(
        "Air Temperature (°C)",
        min_value=-10.0,
        max_value=50.0,
        value=30.0,
        step=0.5,
        help="Ambient temperature around the system."
    )

    relative_humidity = st.slider(
        "Relative Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
        help="Amount of moisture in air relative to saturation."
    )

    st.markdown("---")

    predict_clicked = st.button("🚀 Predict System Production", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)  # close glass-card

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Prediction Output")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if predict_clicked:
        features = np.array([[date_hour,
                              wind_speed,
                              sunshine,
                              air_pressure,
                              radiation,
                              air_temperature,
                              relative_humidity]])

        prediction = float(model.predict(features).item())

        # Save to history for visualization
        st.session_state["history"].append(
            {
                "date_hour": date_hour,
                "prediction": prediction,
                "wind_speed": wind_speed,
                "radiation": radiation,
            }
        )

        st.markdown('<div class="prediction-badge">● Live Model Inference</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Predicted System Production</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{prediction:,.2f}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="help-text">Units depend on your training data (e.g., kW, kWh). Add this explicitly in your report.</div>',
            unsafe_allow_html=True
        )

    else:
        st.info("Adjust the parameters on the left and click **“🚀 Predict System Production”** to see the output here.")

    # Simple history chart
    if st.session_state["history"]:
        import pandas as pd
        import altair as alt

        st.markdown("### Scenario History")
        df = pd.DataFrame(st.session_state["history"])

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("date_hour:Q", title="Date-Hour (numeric)"),
                y=alt.Y("prediction:Q", title="Predicted Production"),
                tooltip=[
                    alt.Tooltip("date_hour:Q", title="Date-Hour"),
                    alt.Tooltip("prediction:Q", title="Prediction"),
                    alt.Tooltip("wind_speed:Q", title="Wind Speed (m/s)"),
                    alt.Tooltip("radiation:Q", title="Radiation (W/m²)"),
                ],
            )
            .properties(
                height=260,
            )
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown(
        '<div class="footer-note">Model: system_production_model.pkl · Powered by AI & Renewable Energy Analytics ⚡</div>',
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)  # close glass-card
