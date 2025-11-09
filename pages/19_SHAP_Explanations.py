import streamlit as st
import os
from utils.ui_helpers import initialize_state, get_resource_metrics
from utils import shap_explain_tree_model

st.set_page_config(page_title="SHAP Explanations", layout="wide")
initialize_state()

st.title("🧠 SHAP Explanations for Tree Models")
st.caption("Explain predictions for a saved RandomForest/XGBoost model on a test CSV. One explanation per class plus per-row top features.")

with st.sidebar:
    st.header("System Health")
    m = get_resource_metrics()
    st.metric("CPU %", f"{m['CPU %']:.1f}%")
    st.metric("RAM %", f"{m['RAM %']:.1f}%")

col1, col2, col3 = st.columns(3)
with col1:
    model_path = st.text_input("Model path (.pkl/.joblib)", value="Attack_Model/training_model.pkl")
with col2:
    test_csv = st.text_input("Test CSV path", value="Balanced_Test_2018/full.csv")
with col3:
    label_map = st.text_input("Label mapping file (optional)", value="Attack_Model/training_label_mapping.txt")

colA, colB = st.columns(2)
with colA:
    sample_rows = st.number_input("Sample rows (0 = all)", min_value=0, max_value=200000, value=0, step=1000)
with colB:
    top_k = st.number_input("Top features per row", min_value=1, max_value=10, value=3, step=1)

if st.button("Run SHAP", use_container_width=True):
    if not os.path.isfile(model_path) or not os.path.isfile(test_csv):
        st.error("Model or Test CSV not found.")
    else:
        try:
            with st.spinner("Computing SHAP values (may take time on large data)..."):
                explanations, per_row_df = shap_explain_tree_model(
                    model_path=model_path,
                    test_csv_path=test_csv,
                    label_mapping_path=label_map if os.path.isfile(label_map) else None,
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

