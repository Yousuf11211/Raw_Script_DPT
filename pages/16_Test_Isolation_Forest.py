import streamlit as st
import os
from utils.ui_helpers import initialize_state, get_resource_metrics
from utils import test_isolation_forest_on_csv

st.set_page_config(page_title="Test Isolation Forest (Benign)", layout="wide")
initialize_state()

st.title("🧪 Test Isolation Forest — Benign vs Anomaly")
st.info("Use this to evaluate an Isolation Forest model on a labeled dataset. True labels are mapped to 1 (Benign) and -1 (Attack).")

with st.sidebar:
    st.header("System Health")
    m = get_resource_metrics()
    st.metric("CPU %", f"{m['CPU %']:.1f}%")
    st.metric("RAM %", f"{m['RAM %']:.1f}%")

col1, col2, col3 = st.columns(3)
with col1:
    model_path = st.text_input("Model path (.joblib)", value="Training_isolation_model_cleaned/isolation.joblib")
with col2:
    test_csv_path = st.text_input("Test CSV path", value="Testing_isolation_model_cleaned/Benign_part_2.csv")
with col3:
    label_col = st.text_input("Label column name", value="label")

benign_name = st.text_input("Benign label value", value="Benign")

if st.button("Run Isolation Test", use_container_width=True):
    if not os.path.isfile(model_path) or not os.path.isfile(test_csv_path):
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

