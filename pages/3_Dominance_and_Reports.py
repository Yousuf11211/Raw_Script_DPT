import streamlit as st
import polars as pl
from utils.ui_helpers import initialize_state, get_resource_metrics, common_header, get_lazy_data_reader
from utils.data_analysis import get_class_distribution_report, get_dominance_report, get_value_label_breakdown

st.set_page_config(page_title="Dominance & Reports", layout="wide")
initialize_state()

# Header for dataset selection
hdr = common_header("📈 Dominance & Reports", num_inputs=1, input_specs=[{"label": "Input CSV", "kind": "file", "allowed_exts": [".csv"]}], default_output_folder="")
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
if lf is None:
    st.info("Go to Home to select or upload a CSV first.")
    st.stop()

row_count = lf.select(pl.count()).collect().item()
col_count = len(lf.columns)
st.metric("Rows", f"{row_count:,}")
st.metric("Columns", f"{col_count:,}")

st.subheader("Class Distribution")
if st.button("Compute", use_container_width=True):
    with st.spinner("Aggregating label counts..."):
        class_df, fig = get_class_distribution_report(lf)
    st.session_state['class_df'] = class_df
    st.session_state['class_fig'] = fig

if 'class_df' in st.session_state and not st.session_state['class_df'].empty:
    st.dataframe(st.session_state['class_df'], use_container_width=True)
    if st.session_state['class_fig'] is not None:
        st.pyplot(st.session_state['class_fig'])

st.divider()
st.subheader("Dominance Report")
if st.button("Generate", use_container_width=True):
    with st.spinner("Computing dominance across all columns..."):
        dom_df, label_df = get_dominance_report(lf)
    st.session_state['dominance_summary'] = dom_df
    st.session_state['dominance_label_df'] = label_df

if 'dominance_summary' in st.session_state:
    dom_df = st.session_state['dominance_summary']
    label_df = st.session_state.get('dominance_label_df')
    if not dom_df.empty:
        st.subheader("Global Label Distribution")
        st.dataframe(label_df, use_container_width=True)
        st.download_button("Download Label Distribution (CSV)", data=label_df.to_csv(index=False).encode('utf-8'), file_name="label_distribution.csv", mime="text/csv")
        st.subheader("Dominance Summary")
        st.dataframe(dom_df[['Feature','Most Common Value','Ratio','Dominance Range']], use_container_width=True)
        st.download_button("Download Dominance Summary (CSV)", data=dom_df.to_csv(index=False).encode('utf-8'), file_name="dominance_summary.csv", mime="text/csv")

        with st.expander("Per-Value Label Breakdown"):
            feature = st.selectbox("Select a feature to inspect:", options=list(lf.columns))
            topn = st.number_input("Top values to show", min_value=5, max_value=100, value=10)
            if st.button("Compute Breakdown", key="compute_breakdown_btn_dom"):
                with st.spinner("Aggregating value/label breakdown..."):
                    breakdown_df = get_value_label_breakdown(lf, feature=feature, top_n=int(topn))
                st.dataframe(breakdown_df, use_container_width=True)
                st.download_button("Download Breakdown (CSV)", data=breakdown_df.to_csv(index=False).encode('utf-8'), file_name=f"{feature}_value_label_breakdown.csv", mime="text/csv")
    else:
        st.info("Dominance report is empty or failed.")
