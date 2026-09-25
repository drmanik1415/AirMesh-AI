import os
import urllib.request

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# AIRMESH AI — DEPLOYMENT CONFIGURATION
# ============================================================

MODEL_FILE = "airmesh_deployment_v1.joblib"
DATA_FILE = "airmesh_demo_data.csv"

MODEL_URL = (
    "https://github.com/drmanik1415/AirMesh-AI/"
    "releases/download/v1.0/"
    "airmesh_deployment_v1.joblib"
)


# ============================================================
# DOWNLOAD MODEL IF NOT PRESENT
# ============================================================

def download_model():

    if os.path.exists(MODEL_FILE):
        print("Model already available.")
        return

    print("Downloading AirMesh AI model...")
    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_FILE
    )

    print("Model downloaded successfully.")


download_model()


# ============================================================
# LOAD DEPLOYMENT PACKAGE
# ============================================================

package = joblib.load(MODEL_FILE)

rf_classifier = package["rf_classifier"]
cluster_model = package["cluster_model"]
cluster_scaler = package["cluster_scaler"]

model_features = package["model_features"]
cluster_features = package["cluster_features"]

warning_threshold = package["warning_threshold"]
state_names = package["state_names"]

print("AirMesh AI models loaded successfully.")


# ============================================================
# LOAD DEMONSTRATION SENSOR DATA
# ============================================================

demo_data = pd.read_csv(DATA_FILE)

demo_data["timestamp"] = pd.to_datetime(
    demo_data["timestamp"]
)

demo_streams = {}

for zone in ["Zone A", "Zone B", "Zone C"]:

    demo_streams[zone] = (
        demo_data[
            demo_data["zone"] == zone
        ]
        .drop(columns=["zone"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# SENSOR HISTORY VALIDATION
# ============================================================

def validate_zone_history(zone_history):

    required_columns = [
        "timestamp",
        "co2",
        "pm25",
        "temperature",
        "humidity"
    ]

    df = zone_history.copy()

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    sensor_cols = [
        "co2",
        "pm25",
        "temperature",
        "humidity"
    ]

    for col in sensor_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(df) < 6:
        raise ValueError(
            "At least 6 readings are required."
        )

    recent = df.iloc[-6:].copy()

    if recent[required_columns].isna().any().any():
        raise ValueError(
            "Recent sensor history contains invalid values."
        )

    time_diffs = (
        recent["timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60
    )

    if not (time_diffs == 5).all():
        raise ValueError(
            "Readings must be consecutive 5-minute observations."
        )

    return recent


# ============================================================
# TIME-SERIES FEATURE ENGINEERING
# ============================================================

def create_airmesh_features(zone_history):

    df = zone_history.copy()

    sensor_cols = [
        "co2",
        "pm25",
        "temperature",
        "humidity"
    ]

    latest_features = {}

    for col in sensor_cols:

        latest_features[col] = (
            df[col].iloc[-1]
        )

        latest_features[
            f"{col}_lag5"
        ] = df[col].iloc[-2]

        latest_features[
            f"{col}_change15"
        ] = (
            df[col].iloc[-1]
            - df[col].iloc[-4]
        )

        latest_features[
            f"{col}_mean30"
        ] = (
            df[col]
            .iloc[-6:]
            .mean()
        )

    return latest_features


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

def predict_deterioration(feature_values):

    input_df = pd.DataFrame(
        [feature_values]
    )[model_features]

    probability = (
        rf_classifier
        .predict_proba(input_df)[0, 1]
    )

    warning = (
        probability >= warning_threshold
    )

    return {
        "probability":
            float(probability),

        "probability_percent":
            float(probability * 100),

        "early_warning":
            bool(warning),

        "ml_status":
            (
                "DETERIORATION_WARNING"
                if warning
                else "NORMAL"
            )
    }


# ============================================================
# K-MEANS ENVIRONMENTAL STATE
# ============================================================

def identify_environmental_state(latest):

    current_values = pd.DataFrame([{
        "co2": latest["co2"],
        "pm25": latest["pm25"],
        "temperature": latest["temperature"],
        "humidity": latest["humidity"]
    }])

    scaled = cluster_scaler.transform(
        current_values[cluster_features]
    )

    state_number = int(
        cluster_model.predict(scaled)[0]
    )

    return {
        "state": state_number,
        "name": state_names[state_number]
    }


# ============================================================
# COMPLETE ZONE ANALYSIS
# ============================================================

def analyze_zone(zone_history, zone):

    history = validate_zone_history(
        zone_history
    )

    feature_values = create_airmesh_features(
        history
    )

    prediction = predict_deterioration(
        feature_values
    )

    latest = history.iloc[-1]

    environmental_state = (
        identify_environmental_state(
            latest
        )
    )

    return {
        "zone": zone,

        "timestamp":
            str(latest["timestamp"]),

        "co2":
            float(latest["co2"]),

        "pm25":
            float(latest["pm25"]),

        "temperature":
            float(latest["temperature"]),

        "humidity":
            float(latest["humidity"]),

        "environmental_state":
            environmental_state["state"],

        "environmental_state_name":
            environmental_state["name"],

        "probability_percent":
            prediction["probability_percent"],

        "early_warning":
            prediction["early_warning"],

        "ml_status":
            prediction["ml_status"]
    }


# ============================================================
# TREND GRAPH
# ============================================================

def create_co2_chart():

    fig, ax = plt.subplots(
        figsize=(8, 3.5)
    )

    for zone in [
        "Zone A",
        "Zone B",
        "Zone C"
    ]:

        history = demo_streams[zone]

        ax.plot(
            history["timestamp"],
            history["co2"],
            marker="o",
            label=zone
        )

    ax.set_title(
        "Recent CO₂ Trend"
    )

    ax.set_ylabel(
        "CO₂ (ppm)"
    )

    ax.set_xlabel(
        "Time"
    )

    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    return fig


# ============================================================
# DASHBOARD ANALYSIS
# ============================================================

def update_dashboard():

    results = {}

    for zone in [
        "Zone A",
        "Zone B",
        "Zone C"
    ]:

        results[zone] = analyze_zone(
            demo_streams[zone],
            zone
        )

    highest_risk_zone = max(
        results,
        key=lambda z:
            results[z][
                "probability_percent"
            ]
    )

    highest = results[
        highest_risk_zone
    ]

    zone_outputs = []

    for zone in [
        "Zone A",
        "Zone B",
        "Zone C"
    ]:

        r = results[zone]

        status = (
            "⚠️ DETERIORATION WARNING"
            if r["early_warning"]
            else "✅ NORMAL"
        )

        summary = f"""
{status}

Environmental State:
{r['environmental_state_name']}

CO₂: {r['co2']:.0f} ppm
PM2.5: {r['pm25']:.1f} µg/m³

Temperature: {r['temperature']:.2f} °C
Humidity: {r['humidity']:.2f} %

~15-min Deterioration Probability:
{r['probability_percent']:.1f} %
"""

        zone_outputs.append(summary)

    if highest["early_warning"]:

        master_status = f"""
⚠️ PRIORITY ZONE: {highest_risk_zone}

Highest predicted deterioration probability:
{highest['probability_percent']:.1f} %

Environmental State:
{highest['environmental_state_name']}

ML Recommendation:
VENTILATION REVIEW — {highest_risk_zone.upper()}

The Master ESP32 validates this advisory against
local sensor readings and deterministic failsafe
rules before any physical ventilation action.
"""

    else:

        master_status = """
✅ NO ML EARLY WARNING

All monitored zones are below the prototype
ML operating threshold.

Master ESP32 continues local monitoring.
"""

    return (
        zone_outputs[0],
        zone_outputs[1],
        zone_outputs[2],
        master_status,
        create_co2_chart()
    )


# ============================================================
# GRADIO USER INTERFACE
# ============================================================

with gr.Blocks(
    title="AirMesh AI Control Center"
) as app:

    gr.Markdown(
        """
# 🌐 AirMesh AI Control Center

### Predictive Multi-Zone Classroom Air Quality

**SENSE → UNDERSTAND → PREDICT → ACT → VERIFY**

> **DEMONSTRATION MODE**
>
> Recorded real-world lecture-hall environmental
> data is used to demonstrate the AirMesh analytics
> pipeline. These measurements were not collected
> by the physical AirMesh prototype.
"""
    )

    analyze_button = gr.Button(
        "Analyze AirMesh Network",
        variant="primary"
    )

    gr.Markdown(
        "## Zone Intelligence"
    )

    with gr.Row():

        zone_a = gr.Textbox(
            label="ZONE A",
            lines=11
        )

        zone_b = gr.Textbox(
            label="ZONE B",
            lines=11
        )

        zone_c = gr.Textbox(
            label="ZONE C",
            lines=11
        )

    gr.Markdown(
        "## Master ESP32 Decision Support"
    )

    master_output = gr.Textbox(
        label="Master Controller Advisory",
        lines=10
    )

    gr.Markdown(
        "## Recent Environmental Trend"
    )

    trend_plot = gr.Plot()

    gr.Markdown(
        """
---

### AI Architecture

**Environmental State:** K-Means clustering  
**Predictive Model:** Random Forest classifier  
**Prediction Horizon:** approximately 15 minutes  
**Prototype ML Operating Threshold:** 35%

The AI provides predictive decision support.
Final physical control remains with the Master
ESP32 and deterministic failsafe logic.
"""
    )

    analyze_button.click(
        fn=update_dashboard,
        outputs=[
            zone_a,
            zone_b,
            zone_c,
            master_output,
            trend_plot
        ]
    )


# ============================================================
# RENDER SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )
