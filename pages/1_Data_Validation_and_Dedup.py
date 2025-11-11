# Moved from root pages/1_Data_Validation_and_Dedup.py
import streamlit as st
import polars as pl
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav, common_header, get_lazy_data_reader
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

# --- START CSS INJECTION FOR SOFT DARK LOOK & ALIGNMENT ---
soft_dark_css = """
<style>
/* 1. Base Colors & Font Smoothing (Dark Theme) */
:root {
    --primary-bg: #1E2328; /* Dark Background */
    --secondary-bg: #293038; /* Slightly lighter card background */
    --text-color: #E6E6E6; /* Light gray text */
    --soft-blue: #8AA8F3; /* Primary Accent (Soft Lavender/Sky Blue) */
    --soft-green: #A3D9A5; /* Success Green */
    --soft-yellow: #F7E59D; /* Warning Gold */
    --soft-info: #C8D7E3; /* Info Blue-Gray */
    --border-color: rgba(255, 255, 255, 0.1);
}

body {
    background-color: var(--primary-bg);
    color: var(--text-color);
    -webkit-font-smoothing: antialiased;
}
h1, h2, h3, h4, h5, h6, label {
    color: var(--text-color);
}

/* 2. Soft Containers & Cards */
.stApp {
    background-color: var(--primary-bg);
}
.stApp > header {
    background-color: var(--secondary-bg);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); 
}
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}
.stFrame, .stContainer, .stExpander, .stDataFrame, .stTextInput, .stAlert, .stMetric {
    border-radius: 12px !important;
    background-color: var(--secondary-bg);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); 
    border: 1px solid var(--border-color);
}
/* Ensure Streamlit elements respect the dark text color */
div[data-testid="stText"], div[data-testid="stMarkdownContainer"] {
    color: var(--text-color);
}


/* 3. Soft Buttons */
.stButton>button {
    border-radius: 20px;
    border: none !important;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    transform: translateY(-1px);
}

/* Primary buttons (Run Validation, Apply Drop) */
.stButton [data-testid="baseButton-secondary"] {
    background-color: var(--soft-blue);
    color: #000000 !important; /* Darker text for contrast on soft blue */
}
.stButton [data-testid="baseButton-secondary"]:hover {
    background-color: #7A93D3;
}

/* 4. Soft Alerts */
[data-testid="stAlert"] {
    background-color: var(--secondary-bg) !important; 
    border-left: 6px solid !important; 
}

/* Softening alert colors with background and border */
[data-testid="stAlert-success"] { border-left-color: var(--soft-green) !important; color: var(--soft-green) !important; }
[data-testid="stAlert-warning"] { border-left-color: var(--soft-yellow) !important; color: var(--soft-yellow) !important; }
[data-testid="stAlert-info"] { border-left-color: var(--soft-info) !important; color: var(--soft-info) !important; }

/* 5. Metrics */
[data-testid="stMetric"] {
    background-color: var(--secondary-bg);
}

/* 6. ALIGNMENT FIX: Use Flexbox to align sub-buttons (the "Apply Drop" buttons) */
/* This targets the column structure where the buttons are placed */
div.stVerticalBlock > div.stVerticalBlock {
    /* Apply a grid or flexbox to the container holding the button blocks */
    display: flex;
    flex-direction: row; /* Layout children horizontally */
    align-items: stretch; /* Make all columns the same height */
    gap: 15px; /* Spacing between columns */
}
/* Ensure the button containers within the flexbox take up equal space */
div.stVerticalBlock > div.stVerticalBlock > div.stBlock {
    flex: 1 1 0%; /* Flex magic: grow, shrink, and basis 0 */
    min-width: 150px; /* Ensure a minimum size */
}
</style>
"""

st.markdown(soft_dark_css, unsafe_allow_html=True)
# --- END CSS INJECTION ---


# We call this again just in case it sets up other non-CSS elements
inject_global_styles()
render_top_nav(current_page="DataCleaning/1_Data_Validation_and_Dedup")

hdr = common_header("Data Validation & Deduplication", num_inputs=1,
                    input_specs=[{"label": "Input CSV", "kind": "file", "allowed_exts": [".csv"]}],
                    default_output_folder="")
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

# Use st.columns for better metric layout (recommended for Streamlit)
col1, col2 = st.columns(2)
with col1:
    row_count = lf.select(pl.count()).collect().item()
    st.metric("Rows", f"{row_count:,}")
with col2:
    col_count = len(lf.columns)
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

with st.expander("Save dataset to disk (after validation)"):
    default_path = default_output_path(file_path, suffix="validated")
    out_path = st.text_input("Output CSV path", value=default_path)
    if st.button("Save CSV (Validated)"):
        ok = write_lazyframe_to_csv(lf, out_path)
        if ok:
            st.success(f"Saved to {out_path}")

st.divider()
st.subheader("2) Duplicate Columns")

if st.button("Check Duplicate Columns", use_container_width=True, key="check_cols_btn"):  # Added key
    if not file_path:
        st.warning("Original file path unknown; header scan uses file path.")
    else:
        total_cols, duplicate_names = get_duplicate_columns(file_path)
        st.info(f"Total Columns: {total_cols}")

        # --- ALIGNMENT ZONE 2: Duplicate Columns ---
        if duplicate_names:
            st.warning(f"Found duplicates: {', '.join(duplicate_names)}")

            # Using st.columns here ensures proper visual alignment for sub-buttons
            col_drop, col_empty = st.columns([1, 3])
            with col_drop:
                if st.button("Apply Drop Duplicate Columns (Lazy)", use_container_width=True, key="drop_cols_btn"):
                    lf = drop_duplicate_columns_lazy(lf, duplicate_names)
                    st.session_state['current_lazy_frame'] = lf
                    st.session_state['applied_filters'].append(f"Drop duplicate columns ({len(duplicate_names)})")
                    st.success("Scheduled lazy drop of duplicate columns.")
            # col_empty exists to push the drop button to the left and center it visually if needed
        else:
            st.success("No duplicate columns found.")
        # --- END ALIGNMENT ZONE 2 ---

with st.expander("Save dataset to disk (after column dedup)"):
    default_path = default_output_path(file_path, suffix="col_dedup")
    out_path = st.text_input("Output CSV path", value=default_path, key="save_cols")
    if st.button("Save CSV (After Column Dedup)", key="save_cols_btn_exp"):  # Added key
        ok = write_lazyframe_to_csv(lf, out_path)
        if ok:
            st.success(f"Saved to {out_path}")

st.divider()
st.subheader("3) Duplicate Rows")

if st.button("Calculate Duplicate Rows", use_container_width=True, key="calc_rows_btn"):  # Added key
    with st.spinner("Counting duplicates..."):
        total_rows, duplicate_rows = get_row_and_duplicate_counts(lf)
    st.info(f"Initial Total Rows: {total_rows:,}")
    st.info(f"Duplicate Rows: {duplicate_rows:,}")

    # --- ALIGNMENT ZONE 3: Duplicate Rows ---
    if duplicate_rows > 0:
        # Using st.columns here ensures proper visual alignment for sub-buttons
        col_drop, col_empty = st.columns([1, 3])
        with col_drop:
            if st.button("Apply Drop Duplicate Rows (Lazy)", use_container_width=True, key="drop_rows_btn"):
                lf = drop_duplicate_rows_lazy(lf)
                st.session_state['current_lazy_frame'] = lf
                st.session_state['applied_filters'].append("Drop duplicate rows")
                st.success("Scheduled lazy drop of duplicate rows.")
        # col_empty exists to push the drop button to the left
    else:
        st.success("No duplicate rows detected.")
    # --- END ALIGNMENT ZONE 3 ---

with st.expander("Save dataset to disk (after row dedup)"):
    default_path = default_output_path(file_path, suffix="row_dedup")
    out_path = st.text_input("Output CSV path", value=default_path, key="save_rows")
    if st.button("Save CSV (After Row Dedup)", key="save_rows_btn_exp"):  # Added key
        ok = write_lazyframe_to_csv(lf, out_path)
        if ok:
            st.success(f"Saved to {out_path}")