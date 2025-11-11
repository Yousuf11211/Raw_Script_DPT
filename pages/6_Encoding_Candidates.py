# Moved from root pages/6_Encoding_Candidates.py
import streamlit as st
import polars as pl
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav, common_header, get_lazy_data_reader
from utils.data_quality import analyze_encoding_candidates, drop_columns_lazy, coerce_columns_to_numeric, sample_string_values, coerce_columns_to_datetime, coerce_ipv4_to_integer
from utils.io_utils import write_lazyframe_to_csv, default_output_path

st.set_page_config(page_title="Encoding Candidates", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="pages/6_Encoding_Candidates.py")

hdr = common_header("Encoding Candidates Analysis", num_inputs=1, input_specs=[{"label": "Input CSV", "kind": "file", "allowed_exts": [".csv"]}], default_output_folder="")
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

row_count = lf.select(pl.count()).collect().item()
col_count = len(lf.columns)
st.metric("Rows", f"{row_count:,}")
st.metric("Columns", f"{col_count:,}")

st.subheader("Analyze Columns for Encoding Needs")
if st.button("Run Encoding Analysis", use_container_width=True):
    with st.spinner("Scanning columns for categorical encoding needs..."):
        enc_df = analyze_encoding_candidates(lf)
    st.session_state['encoding_df'] = enc_df

enc_df = st.session_state.get('encoding_df')
if enc_df is not None and not enc_df.empty:
    with st.expander("Full Encoding Candidate Report"):
        st.dataframe(enc_df, use_container_width=True)

    needing = enc_df[enc_df['NeedsEncoding']]
    datetime_like = enc_df[enc_df.get('IsDatetime', False) == True]
    ip_like = enc_df[enc_df.get('IsIP', False) == True]
    st.markdown("### Columns That Likely Need Encoding")
    if not needing.empty:
        st.warning(f"{len(needing)} columns flagged for encoding.")
        st.dataframe(needing[['Feature','UniqueCount','CardinalityLabel','SuggestedEncoding']], use_container_width=True)
        for col in needing['Feature']:
            with st.expander(f"Sample string values in '{col}'"):
                samples = sample_string_values(lf, col, max_samples=10)
                if samples:
                    st.write(samples)
                else:
                    st.info("No string values found or unable to sample.")
    else:
        st.success("No columns require encoding based on heuristics.")

    if not datetime_like.empty:
        st.markdown("### Columns Detected as Datetime-Like")
        st.info("These columns mostly parse as datetime; consider coercing to datetime instead of encoding.")
        st.dataframe(datetime_like[['Feature','UniqueCount','SuggestedEncoding']], use_container_width=True)

    if not ip_like.empty:
        st.markdown("### Columns Detected as IP-Like")
        st.info("These columns look like IP addresses; consider coercing to integer representation.")
        st.dataframe(ip_like[['Feature','UniqueCount','SuggestedEncoding']], use_container_width=True)

    st.markdown("---")
    st.subheader("Actions")
    st.caption("You can drop columns or coerce them to numeric (if they are numeric-like with stray strings).")

    colA, colB = st.columns(2)
    with colA:
        to_coerce = st.multiselect("Columns to coerce to numeric:", options=list(enc_df['Feature']))
        if st.button("Apply Coerce (Lazy)"):
            lf_new = coerce_columns_to_numeric(lf, to_coerce, dtype=pl.Float64())
            st.session_state['current_lazy_frame'] = lf_new
            st.session_state['applied_filters'].append(f"Coerce encoding cols ({len(to_coerce)} cols)")
            st.success("Scheduled lazy coercion.")

    with colB:
        to_drop = st.multiselect("Columns to drop:", options=list(enc_df['Feature']), key='drop_encoding_cols')
        if st.button("Apply Drop (Lazy)"):
            lf_new = drop_columns_lazy(lf, to_drop)
            st.session_state['current_lazy_frame'] = lf_new
            st.session_state['applied_filters'].append(f"Drop encoding cols ({len(to_drop)} cols)")
            st.success("Scheduled lazy drop.")

    with st.expander("Datetime Coercion"):
        to_dt = st.multiselect("Columns to coerce to Datetime:", options=list(enc_df.get('Feature', [])))
        if st.button("Apply Coerce to Datetime (Lazy)"):
            lf_new = coerce_columns_to_datetime(lf, to_dt)
            st.session_state['current_lazy_frame'] = lf_new
            st.session_state['applied_filters'].append(f"Coerce to datetime ({len(to_dt)} cols)")
            st.success("Scheduled lazy datetime coercion.")

    with st.expander("IP Coercion"):
        to_ip = st.multiselect("Columns to coerce IPv4 to integer:", options=list(enc_df.get('Feature', [])))
        if st.button("Apply Coerce IPv4 to Integer (Lazy)"):
            lf_new = coerce_ipv4_to_integer(lf, to_ip)
            st.session_state['current_lazy_frame'] = lf_new
            st.session_state['applied_filters'].append(f"Coerce IPv4 to int ({len(to_ip)} cols)")
            st.success("Scheduled lazy IPv4 coercion.")

    with st.expander("Save dataset to disk (after encoding actions)"):
        default_path = default_output_path(file_path, suffix="encoding_cleaned")
        out_path = st.text_input("Output CSV path", value=default_path)
        if st.button("Save CSV (Encoding Cleaned)"):
            ok = write_lazyframe_to_csv(st.session_state['current_lazy_frame'], out_path)
            if ok:
                st.success(f"Saved to {out_path}")
