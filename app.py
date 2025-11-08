import streamlit as st
import polars as pl
import os
from utils.ui_helpers import initialize_state, data_source_selector, get_resource_metrics

# --- INITIALIZATION ---
# Run the initialization function to set up session state and other variables
initialize_state()

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Thesis Data Tool - Home",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PAGE CONTENT ---

st.title("🚀 Thesis Data Tool - Home")
st.write("Use the pages in the left sidebar to perform cleaning, INF handling, and reporting tasks.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("System Health")
    # 1. Fetch the metrics
    metrics = get_resource_metrics()

    # 2. Display them using columns and st.metric
    col_cpu, col_ram = st.columns(2)

    with col_cpu:
        st.metric("CPU Usage", f"{metrics['CPU %']:.1f}%")

    with col_ram:
        st.metric("RAM Used", f"{metrics['RAM Used (GB)']:.2f} GB")
        st.metric("RAM %", f"{metrics['RAM %']:.1f}%")

    st.divider()
    st.caption("Select or upload a dataset below.")

# --- DATA SOURCE SELECTION ON HOME PAGE ---
lf = data_source_selector()

# --- MAIN CONTENT ---
if st.session_state.get('current_lazy_frame') is not None:
    lf_current = st.session_state['current_lazy_frame']
    try:
        row_count = lf_current.select(pl.count()).collect().item()
        col_count = len(lf_current.columns)
        st.success("Dataset loaded.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{row_count:,}")
        c2.metric("Columns", f"{col_count:,}")
        c3.metric("Filters Applied", len(st.session_state['applied_filters']))
        st.subheader("Preview")
        st.dataframe(lf_current.limit(5).collect().to_pandas(), use_container_width=True)
        st.markdown("### Next Steps")
        st.write("Navigate to:")
        st.markdown("- Data Validation & Dedup (Page 1) for cleaning operations")
        st.markdown("- INF Handling (Page 2) for infinite values removal/imputation")
        st.markdown("- Dominance & Reports (Page 3) for analysis and exports")
    except Exception as e:
        st.error(f"Failed to compute preview: {e}")
else:
    st.info("No dataset selected yet. Upload or browse to load a CSV.")

