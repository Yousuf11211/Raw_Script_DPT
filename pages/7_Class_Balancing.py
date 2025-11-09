import streamlit as st
import polars as pl
from utils.ui_helpers import initialize_state, get_resource_metrics
from utils.balancing import balance_dataframe, label_distribution
from utils.io_utils import write_lazyframe_to_csv, default_output_path

st.set_page_config(page_title="Class Balancing", layout="wide")
initialize_state()

st.title("⚖️ Class Balancing (SMOTE / BorderlineSMOTE / ADASYN)")

with st.sidebar:
    st.header("System Health")
    m = get_resource_metrics()
    st.metric("CPU %", f"{m['CPU %']:.1f}%")
    st.metric("RAM %", f"{m['RAM %']:.1f}%")

lf = st.session_state.get('current_lazy_frame')
file_path = st.session_state.get('current_file_path')
if lf is None:
    st.info("Go to Home to select or upload a CSV first.")
    st.stop()

# We need an eager pandas DataFrame for imbalanced-learn
with st.spinner("Collecting data into memory for re-sampling (only do this on reasonably sized datasets)..."):
    df = lf.collect().to_pandas()

if 'label' not in df.columns:
    st.error("This dataset has no 'label' column. Please ensure labels are available for balancing.")
    st.stop()

st.subheader("Current Label Distribution")
st.dataframe(label_distribution(df, 'label'), use_container_width=True)

st.markdown("---")
st.subheader("Balancing Settings")

ratio = st.slider("Minority/Majority target ratio", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
method_choice = st.radio("Oversampling method", ["SMOTE", "BorderlineSMOTE", "ADASYN", "All"], index=0, horizontal=True)

if st.button("Run Balancing", use_container_width=True):
    methods_to_run = ["SMOTE", "BorderlineSMOTE", "ADASYN"] if method_choice == "All" else [method_choice]
    results = []
    for method_name in methods_to_run:
        with st.spinner(f"Balancing with {method_name}..."):
            balanced_df, dist = balance_dataframe(df, 'label', ratio, method_name)
            results.append((method_name, balanced_df, dist))

    for name, bdf, dist in results:
        st.subheader(f"Balanced with {name}")
        st.dataframe(dist, use_container_width=True)
        with st.expander("Preview Balanced Data (first 20 rows)"):
            st.dataframe(bdf.head(20), use_container_width=True)
        # Save buttons for each method
        default_path = default_output_path(file_path, suffix=f"balanced_{name.lower()}")
        out_path = st.text_input(f"Output CSV path for {name}", value=default_path, key=f"out_{name}")
        if st.button(f"Save CSV ({name})"):
            # Convert back to Polars LazyFrame for sink_csv if available
            pl_df = pl.from_pandas(bdf)
            lf_out = pl_df.lazy()
            ok = write_lazyframe_to_csv(lf_out, out_path)
            if ok:
                st.success(f"Saved to {out_path}")
