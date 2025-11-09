import streamlit as st
import os
from utils.ui_helpers import initialize_state, get_resource_metrics
from utils.compare_datasets import get_reference_columns, compare_rows_between_folders

st.set_page_config(page_title="Compare Raw vs Processed", layout="wide")
initialize_state()

st.title("🔍 Compare Raw vs Processed CSV Folders")

with st.sidebar:
    st.header("System Health")
    m = get_resource_metrics()
    st.metric("CPU %", f"{m['CPU %']:.1f}%")
    st.metric("RAM %", f"{m['RAM %']:.1f}%")

st.markdown("Use this tool to verify column consistency and row coverage between a raw data folder and a processed output folder.")

colA, colB = st.columns(2)
with colA:
    raw_folder = st.text_input("Raw folder", value="Raw_Data_2017")
with colB:
    processed_folder = st.text_input("Processed folder", value="Processed_Data_2017")

st.markdown("### Column Consistency")
if st.button("Check Columns", use_container_width=True):
    if not os.path.isdir(raw_folder):
        st.error(f"Raw folder does not exist: {raw_folder}")
    elif not os.path.isdir(processed_folder):
        st.error(f"Processed folder does not exist: {processed_folder}")
    else:
        with st.spinner("Reading reference columns and scanning for mismatches..."):
            raw_info = get_reference_columns(raw_folder)
            proc_info = get_reference_columns(processed_folder)
        if raw_info.get('error'):
            st.error(raw_info['error'])
        else:
            st.success(f"Raw folder reference columns loaded ({raw_info.get('file_count',0)} files).")
            st.code(raw_info['reference'])
            if raw_info['mismatches']:
                st.warning(f"{len(raw_info['mismatches'])} raw files have mismatched columns.")
                with st.expander("Raw mismatches details"):
                    st.json(raw_info['mismatches'])
        if proc_info.get('error'):
            st.error(proc_info['error'])
        else:
            st.success(f"Processed folder reference columns loaded ({proc_info.get('file_count',0)} files).")
            if proc_info['reference'] and raw_info.get('reference') and proc_info['reference'] != raw_info['reference']:
                st.warning("Processed reference columns differ from raw reference.")
            st.code(proc_info['reference'])
            if proc_info['mismatches']:
                st.warning(f"{len(proc_info['mismatches'])} processed files have mismatched columns.")
                with st.expander("Processed mismatches details"):
                    st.json(proc_info['mismatches'])

st.markdown("### Row Set Comparison")
mode = st.radio("Row comparison mode", ["hash", "full"], index=0, help="Hash mode is memory efficient; full stores entire row tuples (large memory).")
hash_method = st.selectbox("Hash method", ["md5", "sha1", "sha256"], index=0)
sample_limit = st.number_input("Sample rows to show", min_value=1, max_value=50, value=10)

if st.button("Compare Rows", use_container_width=True):
    if not os.path.isdir(raw_folder):
        st.error(f"Raw folder does not exist: {raw_folder}")
    elif not os.path.isdir(processed_folder):
        st.error(f"Processed folder does not exist: {processed_folder}")
    else:
        with st.spinner("Computing row signatures across folders..."):
            result = compare_rows_between_folders(raw_folder, processed_folder, mode=mode, hash_method=hash_method, sample_limit=int(sample_limit))
        if result.get('error'):
            st.error(result['error'])
        else:
            st.success("Row comparison done.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Unique Raw Rows", f"{result['raw_unique_rows']:,}")
            c2.metric("Unique Processed Rows", f"{result['processed_unique_rows']:,}")
            c3.metric("Missing (Raw→Processed)", f"{result['missing_rows_count']:,}")
            c4.metric("Extra (Processed only)", f"{result['extra_rows_count']:,}")

            if result['missing_rows_count'] or result['extra_rows_count']:
                with st.expander("Sample Missing Rows"):
                    if result['sample_missing_rows']:
                        st.write(result['sample_missing_rows'])
                    else:
                        st.write("No samples to show.")
                with st.expander("Sample Extra Rows"):
                    if result['sample_extra_rows']:
                        st.write(result['sample_extra_rows'])
                    else:
                        st.write("No samples to show.")

            st.caption(f"Mode: {result['mode']} | Hash: {result.get('hash_method','n/a')} | Sample limit: {int(sample_limit)}")

