import streamlit as st
import os
from utils.ui_helpers import initialize_state, inject_global_styles, render_global_nav, common_header
from utils import shap_explain_tree_model

st.set_page_config(page_title="SHAP Explanations", layout="wide")
initialize_state()
inject_global_styles()
render_global_nav(active_page_hint="Model Testing")

header = common_header(
    "🧠 SHAP Explanations for Tree Models",
    num_inputs=3,
    input_specs=[
        {"label": "Model path (.pkl/.joblib)", "kind": "file", "allowed_exts": [".pkl", ".joblib"]},
        {"label": "Test CSV", "kind": "file", "allowed_exts": [".csv"]},
        {"label": "Label mapping file (optional)", "kind": "file", "allowed_exts": [".txt"]},
    ],
    default_output_folder=""
)
model_path, test_csv, label_map = header['input_paths']

st.caption("Explain predictions for a saved RandomForest/XGBoost model on a test CSV. One explanation per class plus per-row top features.")

colA, colB = st.columns(2)
with colA:
    sample_rows = st.number_input("Sample rows (0 = all)", min_value=0, max_value=200000, value=0, step=1000)
with colB:
    top_k = st.number_input("Top features per row", min_value=1, max_value=10, value=3, step=1)

if st.button("Run SHAP", use_container_width=True):
    if not model_path or not test_csv or not os.path.isfile(model_path) or not os.path.isfile(test_csv):
        st.error("Model or Test CSV not found.")
    else:
        try:
            with st.spinner("Computing SHAP values (may take time on large data)..."):
                explanations, per_row_df = shap_explain_tree_model(
                    model_path=model_path,
                    test_csv_path=test_csv,
                    label_mapping_path=label_map if (label_map and os.path.isfile(label_map)) else None,
                    sample_rows=int(sample_rows) if sample_rows > 0 else None,
                    top_k=int(top_k)
                )
            st.subheader("Per-Class Example Explanations")
            for cls, text in explanations.items():
                st.write(f"- {text}")
            st.subheader("Per-Row Top Contributions (preview)")
            st.dataframe(per_row_df.head(50), use_container_width=True)
            st.download_button("Download per-row SHAP summary CSV", per_row_df.to_csv(index=False), file_name="shap_per_row_summary.csv")
        except Exception as e:
            st.error(f"SHAP failed: {e}")
