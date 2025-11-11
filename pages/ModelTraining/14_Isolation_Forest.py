# Moved from root pages/14_Isolation_Forest.py
import streamlit as st
import os
import joblib
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav, common_header
from utils import train_isolation_forest_on_csv

st.set_page_config(page_title="Isolation Forest (Benign Only)", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="ModelTraining/14_Isolation_Forest")

hdr = common_header(
    "Isolation Forest — Benign Data Model",
    num_inputs=1,
    input_specs=[{"label": "Benign CSV", "kind": "file", "allowed_exts": [".csv"]}],
    default_output_folder="Training_isolation_model_cleaned"
)
sel_benign_csv = hdr['input_paths'][0]
sel_out_folder = hdr['output_folder'] or "Training_isolation_model_cleaned"

st.info("This page trains an Isolation Forest on benign-only data. Do not use attack data here.")

st.subheader("Training Settings")
col1, col2, col3 = st.columns(3)
with col1:
    benign_csv = st.text_input("Benign CSV path", value=sel_benign_csv or "Training_isolation_model_cleaned/Benign_part_2.csv")
with col2:
    out_model = st.text_input("Output model path", value=os.path.join(sel_out_folder, "isolation.joblib"))
with col3:
    chunk_size = st.number_input("Chunk size", min_value=10000, max_value=5000000, value=2000000, step=10000)

colA, colB, colC, colD = st.columns(4)
with colA:
    sample_fraction = st.slider("Sample fraction per chunk", 0.01, 1.0, 0.10, 0.01)
with colB:
    n_estimators = st.number_input("n_estimators", min_value=50, max_value=1000, value=100, step=10)
with colC:
    contamination = st.text_input("contamination ('auto' or 0.0-0.5)", value="auto")
with colD:
    max_samples = st.text_input("max_samples ('auto' or count/fraction)", value="auto")

colE, colF = st.columns(2)
with colE:
    max_features = st.number_input("max_features (0-1)", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
with colF:
    random_state = st.number_input("random_state", min_value=0, max_value=10_000, value=42, step=1)

if st.button("Train Isolation Forest", use_container_width=True):
    if not os.path.isfile(benign_csv):
        st.error("Benign CSV file not found.")
    else:
        try:
            cont = contamination if contamination == 'auto' else float(contamination)
            ms = max_samples if max_samples == 'auto' else (int(max_samples) if max_samples.isdigit() else float(max_samples))
        except Exception:
            st.error("Invalid contamination or max_samples value.")
            st.stop()
        with st.spinner("Training Isolation Forest on numeric features only..."):
            model, stats = train_isolation_forest_on_csv(
                benign_csv,
                chunk_size=int(chunk_size),
                sample_fraction=float(sample_fraction),
                n_estimators=int(n_estimators),
                contamination=cont,
                max_samples=ms,
                max_features=float(max_features),
                random_state=int(random_state)
            )
        joblib.dump(model, out_model)
        st.success(f"Model saved to {out_model}")
        st.json(stats)
