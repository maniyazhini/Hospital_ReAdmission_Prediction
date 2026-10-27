import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "model_bundle.joblib"

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Hospital Readmission Prediction")
st.caption("DMT Project — 30-day readmission risk prediction")

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. First run `python train_model.py` "
        "from the project root."
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
features = bundle["features"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]
category_options = bundle["category_options"]
numeric_defaults = bundle["numeric_defaults"]
model_name = bundle["model_name"]

st.info(
    f"Best model loaded: **{model_name}**. "
    "This is an educational prototype, not a clinically validated medical tool."
)

st.sidebar.header("Patient / Encounter Information")

input_data = {}

# Categorical fields
for col in categorical_features:
    options = category_options.get(col, [])
    if not options:
        input_data[col] = ""
        continue

    # Make labels easier to read
    label = col.replace("_", " ").title()

    # The original dataset stores age ranges like [50-60).
    default_index = 0
    if col == "age" and "[50-60)" in options:
        default_index = options.index("[50-60)")

    input_data[col] = st.sidebar.selectbox(
        label,
        options,
        index=default_index,
    )

# Numeric fields
for col in numeric_features:
    label = col.replace("_", " ").title()
    default = numeric_defaults.get(col, 0.0)

    if col in {"time_in_hospital", "num_lab_procedures", "num_procedures",
               "num_medications", "number_outpatient", "number_emergency",
               "number_inpatient", "number_diagnoses"}:
        input_data[col] = st.sidebar.number_input(
            label,
            min_value=0,
            value=int(round(default)),
            step=1,
        )
    else:
        input_data[col] = st.sidebar.number_input(
            label,
            value=float(default),
        )

st.subheader("Prediction")

if st.button("🔍 Predict Readmission Risk", type="primary", use_container_width=True):
    row = pd.DataFrame([input_data], columns=features)

    prediction = int(pipeline.predict(row)[0])
    probability = float(pipeline.predict_proba(row)[0, 1])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("🔴 HIGHER RISK — likely readmission within 30 days")
        else:
            st.success("🟢 LOWER RISK — not predicted to be readmitted within 30 days")

    with col2:
        st.metric("Estimated 30-Day Readmission Probability", f"{probability:.1%}")

    st.progress(probability)

    with st.expander("View input data"):
        st.dataframe(row, use_container_width=True)

st.divider()
st.caption(
    "Academic demonstration only. Predictions should not be used for diagnosis "
    "or real patient-care decisions."
)
