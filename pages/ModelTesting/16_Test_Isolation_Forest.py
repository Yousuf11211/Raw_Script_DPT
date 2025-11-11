# Moved from root pages/16_Test_Isolation_Forest.py
import streamlit as st
import os
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav, common_header
from utils import test_isolation_forest_on_csv

st.set_page_config(page_title="Test Isolation Forest (Benign)", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="ModelTesting/16_Test_Isolation_Forest")

header = common_header(
    "Test Isolation Forest — Benign vs Anomaly",
    num_inputs=2,
    input_specs=[
        {"label": "Model path (.joblib)", "kind": "file", "allowed_exts": [".joblib"]},
        {"label": "Test CSV", "kind": "file", "allowed_exts": [".csv"]},
    ],
    default_output_folder=""
)
model_path, test_csv_path = header['input_paths']

st.info("Use this to evaluate an Isolation Forest model on a labeled dataset. True labels are mapped to 1 (Benign) and -1 (Attack).")

label_col = st.text_input("Label column name", value="label")
benign_name = st.text_input("Benign label value", value="Benign")

if st.button("Run Isolation Test", use_container_width=True):
    if not model_path or not test_csv_path or not os.path.isfile(model_path) or not os.path.isfile(test_csv_path):
        st.error("Model or Test CSV not found.")
    else:
        with st.spinner("Running Isolation Forest on numeric features only..."):
            res = test_isolation_forest_on_csv(model_path, test_csv_path, label_col=label_col, benign_label=benign_name)
        st.subheader("Prediction Counts (raw)")
        st.json(res.get('prediction_counts_raw', {}))
        if 'report' in res:
            st.subheader("Classification Report")
            st.text(res['report'])
        if 'confusion_matrix' in res:
            st.subheader("Confusion Matrix")
            st.dataframe(res['confusion_matrix'], use_container_width=True)
