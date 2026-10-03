"""Streamlit website for the heart-disease prediction model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import FEATURE_BOUNDS, FEATURE_LABELS, FEATURES, load_bundle, predict

ROOT = Path(__file__).parent
ARTIFACT = ROOT / "artifacts" / "heart_model.joblib"

st.set_page_config(page_title="Heart Health Predictor", page_icon="❤️", layout="wide")
st.title("Heart disease prediction")

try:
    bundle = load_bundle(ARTIFACT)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.code("uv run python -m src.train\nuv run streamlit run app.py")
    st.stop()

with st.sidebar:
    st.header("Model card")
    metrics = bundle.get("metrics", {})
    st.metric("Test ROC-AUC", metrics.get("roc_auc", "n/a"))
    st.metric("Test accuracy", metrics.get("accuracy", "n/a"))
    st.write(f"Training rows: {metrics.get('dataset_rows', 'n/a')}")
    st.write(f"CV accuracy: {metrics.get('cv_accuracy', 'n/a')}")
    st.write(f"Model version: {metrics.get('model_version', 'n/a')}")
    st.write(f"Data hash: {metrics.get('data_hash', 'n/a')}")
    st.warning("Use a qualified clinician for any real health decision.")

tab_single, tab_batch, tab_xray = st.tabs(
    ["Single patient", "CSV batch prediction", "X-ray (extension)"]
)

with tab_single:
    st.subheader("Enter patient measurements")
    defaults = {
        "age": 55.0, "sex": 1, "cp": 1, "trestbps": 130.0, "chol": 240.0,
        "fbs": 0, "restecg": 1, "thalach": 150.0, "exang": 0, "oldpeak": 1.0,
        "slope": 1, "ca": 0, "thal": 2,
    }
    values: dict[str, float] = {}
    columns = st.columns(2)
    for index, feature in enumerate(FEATURES):
        with columns[index % 2]:
            step = 1.0 if feature in {"age", "trestbps", "chol", "thalach"} else 0.1
            low, high = FEATURE_BOUNDS[feature]
            values[feature] = st.number_input(
                FEATURE_LABELS[feature],
                value=float(defaults[feature]),
                step=step,
                min_value=float(low),
                max_value=float(high),
            )
    if st.button("Predict", type="primary"):
        result = predict(bundle, pd.DataFrame([values]))
        probability = float(result.iloc[0]["risk_probability"])
        st.progress(probability, text=f"Estimated model probability: {probability:.1%}")
        st.info(result.iloc[0]["risk_label"])

with tab_batch:
    st.subheader("Upload real-world tabular data")
    st.write("Your CSV must contain the 13 feature columns shown in the model card.")
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            results = predict(bundle, batch)
            combined = pd.concat(
                [batch.reset_index(drop=True), results.reset_index(drop=True)], axis=1
            )
            st.dataframe(combined)
            st.download_button(
                "Download predictions",
                results.to_csv(index=False),
                file_name="heart_predictions.csv",
                mime="text/csv",
            )
        except (ValueError, pd.errors.ParserError) as error:
            st.error(f"Could not score this file: {error}")

with tab_xray:
    st.subheader("Chest X-ray prediction")
    st.info(
        "Not enabled in this MVP. A chest X-ray is an image modality and needs a separately "
        "trained, clinically validated model (for example, on NIH ChestX-ray14 or CheXpert). "
        "Do not treat a future image classifier as a substitute for a cardiologist."
    )
    st.file_uploader(
        "X-ray image (reserved for a separately trained model)",
        type=["png", "jpg", "jpeg"],
        disabled=True,
    )
