import streamlit as st
import polars as pl
from utils.ui_helpers import initialize_state, data_source_selector, get_resource_metrics, common_header, get_lazy_data_reader
from utils.data_cleaning import (
    get_validation_report_and_filter_plan,
    get_duplicate_columns,
    drop_duplicate_columns_lazy,
    get_row_and_duplicate_counts,
    drop_duplicate_rows_lazy,
)
from utils.io_utils import write_lazyframe_to_csv, default_output_path

st.set_page_config(page_title="Data Validation & Dedup", layout="wide")
initialize_state()

# Common header to select a CSV if desired
hdr = common_header("🧹 Data Validation & Deduplication", num_inputs=1, input_specs=[{"label": "Input CSV", "kind": "file", "allowed_exts": [".csv"]}], default_output_folder="")
if hdr['input_paths'][0]:
    path = hdr['input_paths'][0]
    st.session_state['current_file_path'] = path
    lf_loaded = get_lazy_data_reader(path)
    if lf_loaded is not None:
        st.session_state['current_lazy_frame'] = lf_loaded

# Ensure a dataset is selected; allow selection here too for convenience
with st.sidebar:
    st.header("System Health")
    m = get_resource_metrics()
    st.metric("CPU %", f"{m['CPU %']:.1f}%")
    st.metric("RAM %", f"{m['RAM %']:.1f}%")
    st.divider()
    st.caption("You can pick/replace the dataset on the Home page.")

lf = st.session_state.get('current_lazy_frame')
file_path = st.session_state.get('current_file_path')

if lf is None:
    st.info("Go to Home to select or upload a CSV first.")
    st.stop()

# Overview
row_count = lf.select(pl.count()).collect().item()
col_count = len(lf.columns)
st.metric("Rows", f"{row_count:,}")
st.metric("Columns", f"{col_count:,}")
st.dataframe(lf.limit(5).collect().to_pandas(), use_container_width=True)

st.divider()
st.subheader("1) Row Validation (Negative values & Ports)")
st.write("Removes rows where non-negative fields are < 0 or ports are outside [0, 65535].")

if st.button("Run Validation & Apply Filter", use_container_width=True):
    with st.spinner("Finding invalid rows..."):
        lf_validated, report = get_validation_report_and_filter_plan(lf)
    st.session_state['current_lazy_frame'] = lf_validated
    st.session_state['validation_report'] = report
    st.session_state['applied_filters'].append("Row Validation & Filtering")
    st.success(f"Removed {report['invalid_count']:,} invalid rows.")
    if report.get('label_breakdown'):
        with st.expander("Label breakdown of removed rows"):
            st.json(report['label_breakdown'])
    lf = lf_validated

# Save section for this step
with st.expander("Save dataset to disk (after validation)"):
    default_path = default_output_path(file_path, suffix="validated")
    out_path = st.text_input("Output CSV path", value=default_path)
    if st.button("Save CSV (Validated)"):
        ok = write_lazyframe_to_csv(lf, out_path)
        if ok:
            st.success(f"Saved to {out_path}")

st.divider()
st.subheader("2) Duplicate Columns")

if st.button("Check Duplicate Columns", use_container_width=True):
    if not file_path:
        st.warning("Original file path unknown; header scan uses file path.")
    else:
        total_cols, duplicate_names = get_duplicate_columns(file_path)
        st.info(f"Total Columns: {total_cols}")
        if duplicate_names:
            st.warning(f"Found duplicates: {', '.join(duplicate_names)}")
            if st.button("Apply Drop Duplicate Columns (Lazy)"):
                lf = drop_duplicate_columns_lazy(lf, duplicate_names)
                st.session_state['current_lazy_frame'] = lf
                st.session_state['applied_filters'].append(f"Drop duplicate columns ({len(duplicate_names)})")
                st.success("Scheduled lazy drop of duplicate columns.")
        else:
            st.success("No duplicate columns found.")

# Save section for this step
with st.expander("Save dataset to disk (after column dedup)"):
    default_path = default_output_path(file_path, suffix="col_dedup")
    out_path = st.text_input("Output CSV path", value=default_path, key="save_cols")
    if st.button("Save CSV (After Column Dedup)"):
        ok = write_lazyframe_to_csv(lf, out_path)
        if ok:
            st.success(f"Saved to {out_path}")

st.divider()
st.subheader("3) Duplicate Rows")

if st.button("Calculate Duplicate Rows", use_container_width=True):
    with st.spinner("Counting duplicates..."):
        total_rows, duplicate_rows = get_row_and_duplicate_counts(lf)
    st.info(f"Initial Total Rows: {total_rows:,}")
    st.info(f"Duplicate Rows: {duplicate_rows:,}")
    if duplicate_rows > 0:
        if st.button("Apply Drop Duplicate Rows (Lazy)"):
            lf = drop_duplicate_rows_lazy(lf)
            st.session_state['current_lazy_frame'] = lf
            st.session_state['applied_filters'].append("Drop duplicate rows")
            st.success("Scheduled lazy drop of duplicate rows.")
    else:
        st.success("No duplicate rows detected.")

with st.expander("Save dataset to disk (after row dedup)"):
    default_path = default_output_path(file_path, suffix="row_dedup")
    out_path = st.text_input("Output CSV path", value=default_path, key="save_rows")
    if st.button("Save CSV (After Row Dedup)"):
        ok = write_lazyframe_to_csv(lf, out_path)
        if ok:
            st.success(f"Saved to {out_path}")
