import streamlit as st
from utils.ui_helpers import initialize_state, get_resource_metrics

# --- INITIALIZATION ---
initialize_state()

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Thesis Data Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LANDING CONTENT ---
st.title("🚀 Thesis Data Tool")
st.subheader("by Syed Yousuf Uddin")

st.markdown(
    """
Welcome! This workspace provides a full toolkit for data preparation, analysis, and model workflows. 
Use the left sidebar to navigate between pages. Each page now includes its own header to select files and output folders—no need to load data on the home page.

What you can do here:
- Data quality and cleanup:
  - 1) Data Validation & Deduplication
  - 2) INF Value Analysis & Handling
  - 3) Dominance & Reports
  - 4) Constant & Low-Variance Analysis
  - 5) Mixed-Type Analysis
  - 6) Encoding Candidates (coercions, IP/datetime encodings)
  - 10) Delete Columns (dynamic)
- Sampling & balancing:
  - 7) Class Balancing (SMOTE, BorderlineSMOTE, ADASYN)
  - 8) Downscale Dataset (create small Benign/Attack sets)
- Feature engineering & importance:
  - 11) Feature Importance (RandomForest, XGBoost)
  - 19) SHAP Explanations for tree models
- Modeling:
  - 14) Train Isolation Forest on benign-only data
  - 16) Test Isolation Forest on labeled data
  - 13) Hyperparameter Tuning (RF/XGB) + Refit & Evaluate
  - 15) Attack Model (Train & Test) with label mapping support
- Merging, shuffling, and comparisons:
  - 17) Merge & Shuffle CSVs (Polars)
  - 9) Compare Raw vs Processed folders
- Frontend testing utilities:
  - 20) Frontend Test Batch Generator (benign/attack mixes)
  - 18) Outlier Detection & Handling (IQR)

Tips:
- Each page header lets you select the needed input(s) and an output folder.
- Use the "Save results" toggle where available to control writes.
- System health metrics are shown in the sidebar.
    """
)

# --- SIDEBAR ---
with st.sidebar:
    st.header("System Health")
    metrics = get_resource_metrics()
    col_cpu, col_ram = st.columns(2)
    with col_cpu:
        st.metric("CPU Usage", f"{metrics['CPU %']:.1f}%")
    with col_ram:
        st.metric("RAM Used", f"{metrics['RAM Used (GB)']:.2f} GB")
        st.metric("RAM %", f"{metrics['RAM %']:.1f}%")
    st.divider()
    st.caption("Navigate using the sidebar pages. Each page has its own file selection header.")
