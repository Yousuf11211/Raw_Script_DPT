import streamlit as st
import polars as pl
from utils.ui_helpers import initialize_state, get_resource_metrics
from utils.data_quality import map_requested_columns, drop_columns_lazy
from utils.io_utils import write_lazyframe_to_csv, default_output_path

st.set_page_config(page_title="Delete Columns", layout="wide")
initialize_state()

st.title("🗑️ Delete Columns (Dynamic)")

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

# Show column list and a text area to paste names
st.subheader("Select columns to delete")
all_columns = list(lf.columns)

with st.expander("Available columns"):
    st.write(all_columns)

input_mode = st.radio("Input mode", ["Type/Paste names", "Pick from list"], index=0, horizontal=True)

requested = []
if input_mode == "Type/Paste names":
    typed = st.text_area("Columns to delete (comma or newline separated)", height=120)
    if typed:
        # Split by comma/newline
        parts = [p.strip() for p in typed.replace("\n", ",").split(",")]
        requested = [p for p in parts if p]
else:
    requested = st.multiselect("Pick columns", options=all_columns)

found, not_found = map_requested_columns(all_columns, requested)

if requested:
    st.info(f"Requested: {len(requested)} | Found: {len(found)} | Not found: {len(not_found)}")
    if not_found:
        with st.expander("Requested but not found"):
            st.write(not_found)
    if found:
        st.subheader("Columns marked for deletion")
        st.write(found)
        if st.button("Apply Delete (Lazy)", use_container_width=True):
            lf_new = drop_columns_lazy(lf, found)
            st.session_state['current_lazy_frame'] = lf_new
            st.session_state['applied_filters'].append(f"Delete columns ({len(found)} cols)")
            st.success("Scheduled lazy deletion of selected columns.")
else:
    st.caption("Provide column names to delete.")

with st.expander("Save dataset to disk (after deletion)"):
    default_path = default_output_path(file_path, suffix="columns_deleted")
    out_path = st.text_input("Output CSV path", value=default_path)
    if st.button("Save CSV (Columns Deleted)"):
        ok = write_lazyframe_to_csv(st.session_state['current_lazy_frame'], out_path)
        if ok:
            st.success(f"Saved to {out_path}")

