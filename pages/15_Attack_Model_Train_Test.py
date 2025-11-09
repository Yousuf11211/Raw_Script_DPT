import streamlit as st
import os
import joblib
import pandas as pd
from utils.ui_helpers import initialize_state, get_resource_metrics
from utils import train_xgb_on_csv, test_sklearn_model_on_csv

st.set_page_config(page_title="Attack Model (Train & Test)", layout="wide")
initialize_state()

st.title("🎯 Attack Classification Model — Train & Test")

with st.sidebar:
    st.header("System Health")
    m = get_resource_metrics()
    st.metric("CPU %", f"{m['CPU %']:.1f}%")
    st.metric("RAM %", f"{m['RAM %']:.1f}%")

# --- Train Section ---
st.subheader("Train XGBoost on Attacks Dataset")
col1, col2 = st.columns(2)
with col1:
    train_csv = st.text_input("Training CSV path", value="Balanced_Training_2018/training.csv")
with col2:
    model_out = st.text_input("Output model path", value="Attack_Model/training_model.pkl")

st.markdown("Parameters")
colA, colB, colC, colD = st.columns(4)
with colA:
    n_estimators = st.number_input("n_estimators", min_value=50, max_value=1000, value=100, step=10)
with colB:
    max_depth = st.number_input("max_depth", min_value=1, max_value=50, value=10, step=1)
with colC:
    learning_rate = st.number_input("learning_rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%f")
with colD:
    subsample = st.number_input("subsample", min_value=0.5, max_value=1.0, value=1.0, step=0.1)

train_full = st.checkbox("Train on full data (uncheck for 80/20 split)", value=True)

if st.button("Train XGBoost Model", use_container_width=True):
    if not os.path.isfile(train_csv):
        st.error("Training CSV not found.")
    else:
        params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'learning_rate': float(learning_rate),
            'subsample': float(subsample),
        }
        try:
            with st.spinner("Training XGBoost model..."):
                model, label_map, eval_dict = train_xgb_on_csv(train_csv, params, train_full=train_full)
            joblib.dump(model, model_out)
            # Save label mapping text file next to model
            mapping_path = model_out.replace('.pkl', '_label_mapping.txt')
            with open(mapping_path, 'w', encoding='utf-8') as f:
                f.write("Label Encoding Mapping:\n")
                f.write("="*40 + "\n")
                for cls, idx in label_map.items():
                    f.write(f"{cls:<30}: {idx}\n")
            st.success(f"Model saved to {model_out}")
            st.info(f"Label mapping saved to {mapping_path}")
            if eval_dict is not None:
                st.subheader("Validation Report (80/20)")
                st.text(eval_dict['report'])
                st.dataframe(eval_dict['confusion_matrix'], use_container_width=True)
        except Exception as e:
            st.error(f"Training failed: {e}")

st.markdown("---")
# --- Test Section ---
st.subheader("Test a Saved Attack Model")
colT1, colT2, colT3 = st.columns(3)
with colT1:
    model_path = st.text_input("Model file (.pkl/.joblib)", value="Attack_Model/training_model.pkl")
with colT2:
    test_csv = st.text_input("Test CSV path", value="Balanced_Test_2018/full.csv")
with colT3:
    label_map_path = st.text_input("Label mapping file (optional)", value="Attack_Model/training_label_mapping.txt")

if st.button("Run Test", use_container_width=True):
    if not os.path.isfile(model_path) or not os.path.isfile(test_csv):
        st.error("Model or Test CSV not found.")
    else:
        with st.spinner("Running predictions..."):
            res = test_sklearn_model_on_csv(model_path, test_csv, label_map_path if os.path.isfile(label_map_path) else None)
        st.subheader("Prediction Counts")
        st.json(res.get('prediction_counts', {}))
        if 'report' in res:
            st.subheader("Classification Report")
            st.text(res['report'])
        if 'confusion_matrix' in res:
            st.subheader("Confusion Matrix")
            st.dataframe(res['confusion_matrix'], use_container_width=True)

