import streamlit as st
import polars as pl
from utils.ui_helpers import initialize_state, get_resource_metrics, common_header, get_lazy_data_reader
from utils.data_quality import analyze_mixed_types, coerce_columns_to_numeric, drop_columns_lazy
from utils.io_utils import write_lazyframe_to_csv, default_output_path

st.set_page_config(page_title="Mixed-Type Analysis", layout="wide")
initialize_state()

# Header for dataset selection
hdr = common_header("🧪 Mixed-Type Column Analysis", num_inputs=1, input_specs=[{"label": "Input CSV", "kind": "file", "allowed_exts": [".csv"]}], default_output_folder="")
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

st.subheader("Analyze Columns for Mixed Types")
if st.button("Run Mixed-Type Analysis", use_container_width=True):
    with st.spinner("Aggregating counts of NaN/inf/integer/float/string per column..."):
        mt_df = analyze_mixed_types(lf)
    st.session_state['mixed_types_df'] = mt_df

mt_df = st.session_state.get('mixed_types_df')
if mt_df is not None and not mt_df.empty:
    # Flag problematic columns (string or inf present along with numbers)
    mt_df['HasString'] = mt_df['string'] > 0
    mt_df['HasInf'] = mt_df['inf'] > 0
    mt_df['HasNumeric'] = (mt_df['integer'] + mt_df['float']) > 0
    mt_df['MixedWithString'] = mt_df['HasString'] & mt_df['HasNumeric']
    mt_df['MixedWithInf'] = mt_df['HasInf'] & mt_df['HasNumeric']

    with st.expander("Full Mixed-Type Report"):
        st.dataframe(mt_df, use_container_width=True)

    # Problem summaries
    prob_cols_string = mt_df.loc[mt_df['MixedWithString'], 'Feature'].tolist()
    prob_cols_inf = mt_df.loc[mt_df['MixedWithInf'], 'Feature'].tolist()

    st.markdown("### Columns With String + Numeric")
    if prob_cols_string:
        st.warning(f"{len(prob_cols_string)} columns contain string and numeric values.")
        st.write(prob_cols_string)
    else:
        st.info("No columns mixing string and numeric values.")

    st.markdown("### Columns With Inf + Numeric")
    if prob_cols_inf:
        st.warning(f"{len(prob_cols_inf)} columns contain inf and numeric values.")
        st.write(prob_cols_inf)
    else:
        st.info("No columns mixing inf and numeric values.")

    st.markdown("---")
    st.subheader("Fix Options")
    st.caption("Choose columns to coerce to numeric (invalid strings become null), or drop columns entirely.")

    col1, col2 = st.columns(2)

    with col1:
        to_coerce = st.multiselect("Columns to coerce to Float64:", options=list(mt_df['Feature']))
        if st.button("Apply Coerce to Numeric (Lazy)"):
            lf_new = coerce_columns_to_numeric(lf, to_coerce, dtype=pl.Float64)
            st.session_state['current_lazy_frame'] = lf_new
            st.session_state['applied_filters'].append(f"Coerce to numeric ({len(to_coerce)} cols)")
            st.success("Scheduled lazy coercion to numeric.")

    with col2:
        to_drop = st.multiselect("Columns to drop:", options=list(mt_df['Feature']), key="drop_cols_mixed")
        if st.button("Apply Drop Columns (Lazy)"):
            lf_new = drop_columns_lazy(lf, to_drop)
            st.session_state['current_lazy_frame'] = lf_new
            st.session_state['applied_filters'].append(f"Drop mixed-type cols ({len(to_drop)} cols)")
            st.success("Scheduled lazy drop of selected columns.")

    with st.expander("Save dataset to disk (after mixed-type fixes)"):
        default_path = default_output_path(file_path, suffix="mixedtype_cleaned")
        out_path = st.text_input("Output CSV path", value=default_path)
        if st.button("Save CSV (Mixed-Type Cleaned)"):
            ok = write_lazyframe_to_csv(st.session_state['current_lazy_frame'], out_path)
            if ok:
                st.success(f"Saved to {out_path}")
