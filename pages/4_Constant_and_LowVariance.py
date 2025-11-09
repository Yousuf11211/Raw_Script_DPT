import streamlit as st
import polars as pl
from utils.ui_helpers import initialize_state, get_resource_metrics, common_header, get_lazy_data_reader
from utils.data_quality import analyze_constant_low_variance, drop_columns_lazy
from utils.io_utils import write_lazyframe_to_csv, default_output_path

st.set_page_config(page_title="Constant & Low-Variance", layout="wide")
initialize_state()

# Header for dataset selection
hdr = common_header("🧬 Constant & Low-Variance Column Analysis", num_inputs=1, input_specs=[{"label": "Input CSV", "kind": "file", "allowed_exts": [".csv"]}], default_output_folder="")
if hdr['input_paths'][0]:
    path = hdr['input_paths'][0]
    st.session_state['current_file_path'] = path
    lf_loaded = get_lazy_data_reader(path)
    if lf_loaded is not None:
        st.session_state['current_lazy_frame'] = lf_loaded

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

row_count = lf.select(pl.count()).collect().item()
col_count = len(lf.columns)
st.metric("Rows", f"{row_count:,}")
st.metric("Columns", f"{col_count:,}")

st.subheader("Analyze Unique Value Counts")
st.write("This will classify columns as constant (1 unique value) or low-variance (between 2 and your threshold).")
threshold = st.number_input("Low-Variance Max Unique Values", min_value=2, max_value=50, value=3, step=1,
                             help="Columns with unique count between 2 and this value are considered low-variance.")

if st.button("Run Constant/Low-Variance Analysis", use_container_width=True):
    with st.spinner("Computing unique counts and classifying columns..."):
        constant_cols, low_var_cols, report_df = analyze_constant_low_variance(lf, low_var_threshold=int(threshold))
    st.session_state['const_cols'] = constant_cols
    st.session_state['low_var_cols'] = low_var_cols
    st.session_state['unique_counts_df'] = report_df

report_df = st.session_state.get('unique_counts_df')
if report_df is not None and not report_df.empty:
    with st.expander("All Unique Counts (Full Report)"):
        st.dataframe(report_df.sort_values('UniqueCount'), use_container_width=True)

    constant_cols = st.session_state.get('const_cols', [])
    low_var_cols = st.session_state.get('low_var_cols', [])

    st.markdown("### Constant Columns")
    if constant_cols:
        st.success(f"Found {len(constant_cols)} constant columns.")
        st.write(constant_cols)
    else:
        st.info("No constant columns detected.")

    st.markdown("### Low-Variance Columns")
    if low_var_cols:
        st.warning(f"Found {len(low_var_cols)} low-variance columns (threshold <= {threshold}).")
        st.write(low_var_cols)
    else:
        st.info("No low-variance columns detected with current threshold.")

    # Selection for deletion
    st.markdown("---")
    st.subheader("Select Columns to Drop")
    combined_options = constant_cols + [c for c in low_var_cols if c not in constant_cols]
    if combined_options:
        to_drop = st.multiselect("Choose columns to drop:", options=combined_options)
        if st.button("Apply Drop Selected Columns (Lazy)", use_container_width=True, disabled=not to_drop):
            lf_new = drop_columns_lazy(lf, to_drop)
            st.session_state['current_lazy_frame'] = lf_new
            st.session_state['applied_filters'].append(f"Drop const/low-var ({len(to_drop)} cols)")
            st.success(f"Scheduled lazy drop of {len(to_drop)} columns.")
    else:
        st.info("No columns available to drop.")

    with st.expander("Save dataset to disk (after const/low-var drops)"):
        default_path = default_output_path(file_path, suffix="variance_cleaned")
        out_path = st.text_input("Output CSV path", value=default_path)
        if st.button("Save CSV (Variance Cleaned)"):
            ok = write_lazyframe_to_csv(st.session_state['current_lazy_frame'], out_path)
            if ok:
                st.success(f"Saved to {out_path}")
