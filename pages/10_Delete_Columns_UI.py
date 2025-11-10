import streamlit as st
from utils.ui_helpers import initialize_state, inject_global_styles, render_global_nav, common_header, get_lazy_data_reader
from utils.data_quality import map_requested_columns, drop_columns_lazy
from utils.io_utils import write_lazyframe_to_csv, default_output_path

st.set_page_config(page_title="Delete Columns", layout="wide")
initialize_state()
inject_global_styles()
render_global_nav(active_page_hint="Data Cleaning")

# Header for dataset selection
hdr = common_header("🗑️ Delete Columns (Dynamic)", num_inputs=1, input_specs=[{"label": "Input CSV", "kind": "file", "allowed_exts": [".csv"]}], default_output_folder="")
if hdr['input_paths'][0]:
    path = hdr['input_paths'][0]
    st.session_state['current_file_path'] = path
    lf_loaded = get_lazy_data_reader(path)
    if lf_loaded is not None:
        st.session_state['current_lazy_frame'] = lf_loaded

lf = st.session_state.get('current_lazy_frame')
file_path = st.session_state.get('current_file_path')
if lf is None:
    st.info("Select a CSV using the header above.")
    st.stop()

# Get current columns early for use in actions below
all_columns = list(lf.columns)

BASE_COLUMNS_TO_REMOVE = [
    'flow_id','src_ip','dst_ip','timestamp','active_cov','active_max','active_mean','active_median',
    'active_min','active_mode','active_skewness','active_std','active_variance','bwd_cwr_flag_counts',
    'bwd_cwr_flag_percentage_in_bwd_packets','bwd_cwr_flag_percentage_in_total','bwd_payload_bytes_min',
    'bwd_urg_flag_counts','bwd_urg_flag_percentage_in_bwd_packets','bwd_urg_flag_percentage_in_total',
    'fwd_payload_bytes_min','fwd_urg_flag_counts','fwd_urg_flag_percentage_in_fwd_packets',
    'fwd_urg_flag_percentage_in_total','handshake_state','idle_cov','idle_max','idle_mean','idle_median',
    'idle_min','idle_mode','idle_skewness','idle_std','idle_variance','median_bwd_header_bytes_delta_len',
    'median_fwd_header_bytes_delta_len','median_header_bytes_delta_len','mode_bwd_header_bytes_delta_len',
    'mode_fwd_header_bytes_delta_len','payload_bytes_min','urg_flag_counts','protocol',
    'urg_flag_percentage_in_total','cov_bwd_payload_bytes_delta_len','cov_fwd_header_bytes_delta_len',
    'cov_fwd_packets_delta_len','cov_fwd_payload_bytes_delta_len','cov_header_bytes_delta_len',
    'cov_packets_delta_len','cov_payload_bytes_delta_len','protocol',
    'mean_payload_bytes_delta_len','fwd_payload_bytes_mode','mode_header_bytes_delta_len','payload_bytes_mode',
    'min_header_bytes_delta_len','bwd_ece_flag_percentage_in_bwd_packets','avg_fwd_bytes_per_bulk','avg_fwd_packets_per_bulk',
    'packets_IAT_mode','fwd_syn_flag_counts','fwd_bulk_per_packet','fwd_bulk_total_size','bwd_rst_flag_counts',
    'bwd_syn_flag_counts','fwd_bulk_state_count','bwd_ece_flag_percentage_in_total','bwd_ece_flag_counts',
    'mean_fwd_payload_bytes_delta_len','mode_fwd_payload_bytes_delta_len','mode_payload_bytes_delta_len','std_bwd_packets_delta_time',
    'cov_packets_delta_time','cov_fwd_packets_delta_time','mean_packets_delta_time','variance_bwd_packets_delta_time',
    'fwd_payload_bytes_cov','mode_packets_delta_len','min_fwd_packets_delta_time','avg_fwd_bulk_rate','mean_bwd_packets_delta_time',
    'fwd_packets_IAT_min','fwd_payload_bytes_median','rst_flag_counts','skewness_packets_delta_time','skewness_fwd_packets_delta_time',
    'variance_packets_delta_time','bwd_rst_flag_percentage_in_bwd_packets','fwd_variance_header_bytes','bwd_fin_flag_counts',
    'bwd_fin_flag_percentage_in_total','rst_flag_percentage_in_total','handshake_duration','std_packets_delta_time','fwd_packets_IAT_mode',
    'psh_flag_percentage_in_total','payload_bytes_median','variance_fwd_packets_delta_time','bwd_fin_flag_percentage_in_bwd_packets',
    'fwd_std_header_bytes','max_bwd_packets_delta_time','mode_fwd_packets_delta_time','skewness_bwd_header_bytes_delta_len',
    'mode_bwd_packets_delta_time','bwd_packets_IAT_mode','max_bwd_header_bytes_delta_len','mean_bwd_header_bytes_delta_len',
    'skewness_fwd_payload_bytes_delta_len','skewness_payload_bytes_delta_len','median_bwd_packets_delta_len','packet_IAT_min',
    'bwd_variance_header_bytes','std_fwd_packets_delta_time','mean_fwd_packets_delta_time','median_fwd_packets_delta_len',
    'median_bwd_payload_bytes_delta_len','min_fwd_header_bytes_delta_len','fwd_psh_flag_percentage_in_fwd_packets','cov_bwd_packets_delta_time',
    'fwd_packets_count','fwd_ack_flag_counts','mean_fwd_packets_delta_len','bwd_rst_flag_percentage_in_total','avg_bwd_bulk_rate',
    'fwd_cov_header_bytes','fwd_psh_flag_counts','fwd_payload_bytes_skewness','fwd_rst_flag_percentage_in_fwd_packets',
    'fwd_syn_flag_percentage_in_total'
]

with st.expander("Base columns to remove (from batch script)"):
    st.write(BASE_COLUMNS_TO_REMOVE)
    if st.button("Delete ALL Base Columns", key="delete_all_base_cols"):
        base_found, base_not_found = map_requested_columns(all_columns, BASE_COLUMNS_TO_REMOVE)
        if not base_found:
            st.warning("None of the base columns were found in the current dataset.")
        else:
            lf_new = drop_columns_lazy(lf, base_found)
            st.session_state['current_lazy_frame'] = lf_new
            st.session_state['applied_filters'].append(f"Delete ALL base columns ({len(base_found)} found)")
            st.success(f"Scheduled lazy deletion of {len(base_found)} base columns.")
            if base_not_found:
                with st.expander("Base columns not found in current dataset"):
                    st.write(base_not_found)
            lf = lf_new
            all_columns = list(lf.columns)

st.subheader("Select columns to delete")
with st.expander("Available columns"):
    st.write(all_columns)

input_mode = st.radio("Input mode", ["Type/Paste names", "Pick from list"], index=0, horizontal=True)

requested = []
if input_mode == "Type/Paste names":
    typed = st.text_area("Columns to delete (comma or newline separated)", height=120)
    if typed:
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
