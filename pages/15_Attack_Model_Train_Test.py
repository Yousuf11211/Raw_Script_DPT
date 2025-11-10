import streamlit as st
import os
import joblib
import pandas as pd
from utils.ui_helpers import initialize_state, inject_global_styles, render_global_nav, common_header
from utils import train_xgb_on_csv, test_sklearn_model_on_csv

st.set_page_config(page_title="Attack Model (Train & Test)", layout="wide")
initialize_state()
inject_global_styles()
render_global_nav(active_page_hint="Model Training")

# Training header: select training CSV and output folder for model
train_header = common_header(
    "🎯 Attack Classification Model — Train",
    num_inputs=1,
    input_specs=[{"label": "Training CSV", "kind": "file", "allowed_exts": [".csv"]}],
    default_output_folder="Attack_Model"
)
train_csv = train_header['input_paths'][0]
model_out_folder = train_header['output_folder'] or "Attack_Model"
model_out = os.path.join(model_out_folder, "training_model.pkl")

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
    if not train_csv or not os.path.isfile(train_csv):
        st.error("Training CSV not found.")
    else:
        params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'learning_rate': float(learning_rate),
            'subsample': float(subsample),
        }
        try:
            os.makedirs(model_out_folder, exist_ok=True)
            with st.spinner("Training XGBoost model..."):
                model, label_map, eval_dict = train_xgb_on_csv(train_csv, params, train_full=train_full)
            joblib.dump(model, model_out)
            # Save label mapping text file next to model
            mapping_path = os.path.join(model_out_folder, 'training_label_mapping.txt')
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
# Testing header: model file, test CSV, optional label map, and output folder for artifacts
test_header = common_header(
    "🧪 Test a Saved Attack Model",
    num_inputs=3,
    input_specs=[
        {"label": "Model file (.pkl/.joblib)", "kind": "file", "allowed_exts": [".pkl", ".joblib"]},
        {"label": "Test CSV", "kind": "file", "allowed_exts": [".csv"]},
        {"label": "Label mapping file (optional)", "kind": "file", "allowed_exts": [".txt"]},
    ],
    default_output_folder="Test_Reports"
)
model_path, test_csv, label_map_path = test_header['input_paths']
output_folder = test_header['output_folder']

save_cols = st.columns(4)
with save_cols[0]:
    save_report = st.checkbox("Save report", value=False)
with save_cols[1]:
    save_cm = st.checkbox("Save confusion matrix", value=False)
with save_cols[2]:
    save_preds_csv = st.checkbox("Save predictions CSV", value=False)
with save_cols[3]:
    save_counts = st.checkbox("Save counts summary", value=False)

if st.button("Run Test", use_container_width=True):
    if not model_path or not test_csv or not os.path.isfile(model_path) or not os.path.isfile(test_csv):
        st.error("Model or Test CSV not found.")
    else:
        os.makedirs(output_folder, exist_ok=True)
        with st.spinner("Running predictions..."):
            res = test_sklearn_model_on_csv(model_path, test_csv, label_map_path if (label_map_path and os.path.isfile(label_map_path)) else None)
        st.subheader("Prediction Counts")
        st.json(res.get('prediction_counts', {}))
        base_name = os.path.splitext(os.path.basename(test_csv))[0]
        if 'report' in res:
            st.subheader("Classification Report")
            st.text(res['report'])
            if save_report:
                with open(os.path.join(output_folder, f"{base_name}_report.txt"), 'w', encoding='utf-8') as f:
                    f.write(res['report'])
        if 'confusion_matrix' in res:
            st.subheader("Confusion Matrix")
            st.dataframe(res['confusion_matrix'], use_container_width=True)
            if save_cm:
                res['confusion_matrix'].to_csv(os.path.join(output_folder, f"{base_name}_confusion_matrix.csv"), index=True)
        if save_counts and 'prediction_counts' in res:
            with open(os.path.join(output_folder, f"{base_name}_predicted_counts.txt"), 'w', encoding='utf-8') as f:
                for k,v in sorted(res['prediction_counts'].items()):
                    f.write(f"{k}: {v}\n")
        if save_preds_csv and 'predicted_labels' in res:
            df_full = pd.read_csv(test_csv, low_memory=False)
            df_full['predicted_label'] = res['predicted_labels']
            df_full.to_csv(os.path.join(output_folder, f"{base_name}_predictions.csv"), index=False)
            st.info(f"Saved full predictions CSV -> {os.path.join(output_folder, f'{base_name}_predictions.csv')}")
