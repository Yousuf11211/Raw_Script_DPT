# Moved from root pages/18_Outlier_Detection.py
import streamlit as st
import os
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav, common_header, get_lazy_data_reader
from utils import (
    analyze_iqr_outliers,
    remove_outliers_lazy,
    generate_outlier_plot,
    collect_for_plots,
)

st.set_page_config(page_title="Outlier Detection (IQR)", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="pages/DataProcessing/18_Outlier_Detection.py")

hdr = common_header("Outlier Detection & Handling (IQR)", num_inputs=1, input_specs=[{"label": "Input CSV", "kind": "file", "allowed_exts": [".csv"]}], default_output_folder="outlier_plots")
if hdr['input_paths'][0]:
    path = hdr['input_paths'][0]
    st.session_state['current_file_path'] = path
    lf_loaded = get_lazy_data_reader(path)
    if lf_loaded is not None:
        st.session_state['current_lazy_frame'] = lf_loaded

st.caption("Analyze numeric columns for IQR-based outliers, inspect plots, and optionally remove or cap them.")

lf = st.session_state.get('current_lazy_frame')
file_path = st.session_state.get('current_file_path')
if lf is None:
    st.info("Select a CSV using the header above.")
    st.stop()

multiplier = st.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1)
sample_rows = st.number_input("Sample rows for analysis (0 = all)", min_value=0, max_value=2_000_000, value=0, step=10000)

if st.button("Analyze Outliers", use_container_width=True):
    with st.spinner("Computing IQR bounds and outlier counts..."):
        summary_df, bounds = analyze_iqr_outliers(lf, multiplier=float(multiplier), sample_rows=None if sample_rows == 0 else int(sample_rows))
    if summary_df.empty:
        st.warning("No numeric columns or insufficient variability.")
    else:
        st.subheader("Outlier Summary")
        st.dataframe(summary_df.head(100), use_container_width=True)
        st.download_button("Download full summary CSV", summary_df.to_csv(index=False), file_name="outlier_summary.csv")
        st.session_state['outlier_bounds'] = bounds
        st.session_state['outlier_summary_df'] = summary_df

bounds = st.session_state.get('outlier_bounds')
summary_df = st.session_state.get('outlier_summary_df')
if bounds:
    st.markdown("---")
    st.subheader("Plot & Handle Outliers")
    top_cols = []
    if summary_df is not None:
        top_cols = summary_df['Feature'].head(20).tolist()
    selected_cols = st.multiselect("Select columns for plotting/handling", options=top_cols if top_cols else list(bounds.keys()), default=top_cols[:5] if top_cols else [])
    plot_folder = st.text_input("Plot output folder", value="outlier_plots")
    mode = st.radio("Outlier handling mode", ["none","remove","keep_only","cap"], index=0, horizontal=True)

    if st.button("Generate Plots", use_container_width=True):
        pdf = collect_for_plots(lf, selected_cols)
        created = []
        for c in selected_cols:
            if c in bounds:
                lower, upper = bounds[c]
                out_path = os.path.join(plot_folder, f"{c}_plot.png")
                ok = generate_outlier_plot(pdf, c, lower, upper, out_path)
                if ok:
                    created.append(out_path)
        if created:
            st.success(f"Saved {len(created)} plot(s)")
            with st.expander("Generated plot files"):
                for p in created:
                    st.write(p)
        else:
            st.warning("No plots generated (maybe matplotlib missing or no selected columns).")

    if mode != "none" and st.button("Apply Outlier Handling", use_container_width=True):
        new_lf = remove_outliers_lazy(lf, bounds, mode=mode, columns=selected_cols if selected_cols else None)
        st.session_state['current_lazy_frame'] = new_lf
        st.session_state['applied_filters'].append(f"Outlier handling mode={mode} cols={len(selected_cols)}")
        st.success("Outlier handling applied lazily. Save from a data export page to persist.")

