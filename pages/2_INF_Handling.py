import streamlit as st
import polars as pl
from utils import ui_helpers as uih
from utils.data_quality import analyze_inf_columns, drop_inf_columns_lazy, impute_inf_with_median
from utils.io_utils import write_lazyframe_to_csv, default_output_path

st.set_page_config(page_title="INF Handling", layout="wide")
uih.initialize_state()

st.title("♾️ INF Value Analysis & Handling")

with st.sidebar:
    st.header("System Health")
    m = uih.get_resource_metrics()
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

st.subheader("Analyze INF Columns")
if st.button("Analyze", use_container_width=True):
    with st.spinner("Scanning numeric columns for INF values..."):
        total_rows_inf, inf_report_df = analyze_inf_columns(lf)
    st.session_state['inf_total_rows'] = total_rows_inf
    st.session_state['inf_report_df'] = inf_report_df

inf_report_df = st.session_state.get('inf_report_df')
if inf_report_df is not None:
    if inf_report_df.empty:
        st.success("No INF values detected in numeric columns.")
    else:
        st.dataframe(inf_report_df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            threshold = st.number_input("Drop Threshold (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            if st.button("Apply Drop Columns Above Threshold", use_container_width=True):
                lf_new, dropped = drop_inf_columns_lazy(lf, threshold_percent=threshold)
                if dropped:
                    st.session_state['current_lazy_frame'] = lf_new
                    st.session_state['applied_filters'].append(f"Drop INF > {threshold:.1f}%: {len(dropped)} cols")
                    st.success(f"Scheduled lazy drop of {len(dropped)} columns.")
                else:
                    st.info("No columns exceeded the threshold.")
        with c2:
            if st.button("Apply Impute INF with Median", use_container_width=True):
                lf_imp, medians = impute_inf_with_median(lf)
                if medians:
                    st.session_state['current_lazy_frame'] = lf_imp
                    st.session_state['applied_filters'].append(f"Impute INF medians ({len(medians)} cols)")
                    st.success(f"Prepared lazy imputation for {len(medians)} columns.")
                    with st.expander("Median Values Used"):
                        st.json(medians)
                else:
                    st.info("No INF values requiring imputation detected.")

with st.expander("Save dataset to disk (after INF handling)"):
    default_path = default_output_path(file_path, suffix="inf_handled")
    out_path = st.text_input("Output CSV path", value=default_path)
    if st.button("Save CSV (INF handled)"):
        ok = write_lazyframe_to_csv(st.session_state['current_lazy_frame'], out_path)
        if ok:
            st.success(f"Saved to {out_path}")
