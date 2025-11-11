# Moved from root pages/17_Merge_Shuffle_Polars.py
import streamlit as st
import os
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav, common_header
from utils import merge_shuffle_partitioned, merge_shuffle_single

st.set_page_config(page_title="Polars Merge & Shuffle", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="pages/17_Merge_Shuffle_Polars.py")

hdr = common_header("Merge & Shuffle CSVs (Polars)", num_inputs=1, input_specs=[{"label": "Input folder", "kind": "folder"}], default_output_folder="Processed_Polars")
sel_input_folder = hdr['input_paths'][0]
sel_out_folder = hdr['output_folder'] or "Processed_Polars"

st.caption("Merge many CSVs efficiently using Polars; write partitioned shuffled shards or a single shuffled file.")

col1, col2 = st.columns(2)
with col1:
    input_folder = st.text_input("Input folder", value=sel_input_folder or "Raw_Data_2018")
with col2:
    pattern = st.text_input("File pattern", value="*.csv")

colA, colB, colC = st.columns(3)
with colA:
    num_parts = st.number_input("Number of partitions", min_value=1, max_value=500, value=20, step=1)
with colB:
    seed = st.number_input("Shuffle seed", min_value=0, max_value=1_000_000, value=42, step=1)
with colC:
    infer_len = st.number_input("Infer schema length", min_value=50, max_value=100000, value=1000, step=50)

recursive = st.checkbox("Search recursively", value=True)

mode = st.radio("Output mode", ["Partitioned", "Single File"], index=0, horizontal=True)

out_folder = st.text_input("Output folder (for partitioned or temp)", value=sel_out_folder)
single_file_path = st.text_input("Single file output path (if Single File mode)", value=os.path.join(sel_out_folder, "merged_shuffled.csv"))

run_btn = st.button("Run Merge & Shuffle", use_container_width=True)

if run_btn:
    if not os.path.isdir(input_folder):
        st.error(f"Input folder not found: {input_folder}")
    else:
        with st.spinner("Merging & Shuffling (Polars lazy scan)..."):
            if mode == "Partitioned":
                meta = merge_shuffle_partitioned(
                    input_folder=input_folder,
                    output_folder=out_folder,
                    pattern=pattern,
                    num_parts=int(num_parts),
                    seed=int(seed),
                    infer_schema_length=int(infer_len),
                    recursive=recursive,
                )
            else:
                temp_folder = os.path.join(out_folder, "_temp_parts")
                meta = merge_shuffle_single(
                    input_folder=input_folder,
                    output_file=single_file_path,
                    temp_folder=temp_folder,
                    pattern=pattern,
                    num_parts=int(num_parts),
                    seed=int(seed),
                    infer_schema_length=int(infer_len),
                    recursive=recursive,
                )
        if meta.get("error"):
            st.error(meta["error"])
        else:
            st.success("Merge & shuffle completed.")
            st.write(f"Total input files: {len(meta['input_files'])}")
            st.write(f"Total rows (estimated): {meta.get('total_rows','(not counted)')}")
            if mode == "Partitioned":
                with st.expander("Partitioned output files"):
                    for f in meta['created_files']:
                        st.write(f)
            else:
                st.info(f"Single shuffled file: {meta.get('single_file')}")
                with st.expander("Temporary shard files"):
                    for f in meta['created_files']:
                        st.write(f)
