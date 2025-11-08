import streamlit as st
import polars as pl
import os
import pandas as pd
from data_cleaning import (
    get_duplicate_columns,
    drop_duplicate_columns_lazy,
    get_row_and_duplicate_counts,
    drop_duplicate_rows_lazy
)
from io import BytesIO

# --- CONSTANTS AND INITIALIZATION ---

# Define the root path for scanning (your project directory)
SCAN_ROOT = os.getcwd()
# Exclude folders that are not data (like the temp folder or virtual environment)
EXCLUDE_DIRS = ['temp_uploads', 'venv', 'env', '.git', '__pycache__', '.idea/']
SCAN_ROOT_DISPLAY = "PROJECT ROOT"
LOCAL_DATA_PATH = os.path.join(os.getcwd(), 'data',
                               'fixed_thesis_data.csv')  # Retained for context

# Initialize state variables
if 'browser_current_path' not in st.session_state:
    st.session_state['browser_current_path'] = SCAN_ROOT

if 'current_lazy_frame' not in st.session_state:
    st.session_state['current_lazy_frame'] = None

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Thesis Data Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🚀 Thesis Data Processing and Modeling Tool")
st.markdown("---")


# --- CACHING FUNCTIONS (Essential for Performance) ---

@st.cache_resource
def get_lazy_data_reader(file_path):
    """Returns a Polars LazyFrame (lf) from a given file path."""
    try:
        lf = pl.scan_csv(file_path)
        return lf
    except Exception as e:
        st.error(f"Error reading data lazily from path: {file_path}. Details: {e}")
        return None


@st.cache_data
def save_uploaded_file_to_temp(uploaded_file):
    """Saves the uploaded file to a temporary location for Polars to read."""
    temp_dir = 'temp_uploads'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return temp_path


# --- DATA SOURCE HANDLING FUNCTION ---

def get_data_source():
    """
    Manages the UI for selecting the data source (Upload or Local Browse)
    using a dynamic folder navigation system (shallow scan).
    """

    st.header("1. Data Source")
    data_source = st.radio(
        "Select your data source:",
        ("Upload CSV", "Browse Project Files"),
        index=0,
        key="data_source_radio"
    )

    file_path = None

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose a CSV file to upload:",
            type="csv",
            key="csv_uploader"
        )
        if uploaded_file is not None:
            file_path = save_uploaded_file_to_temp(uploaded_file)
            st.session_state['current_file_path'] = file_path

    elif data_source == "Browse Project Files":

        current_dir = st.session_state['browser_current_path']

        # Display the current path for user reference
        relative_display = os.path.relpath(current_dir, SCAN_ROOT)
        if relative_display == ".":
            st.info(f"Current Location: {SCAN_ROOT_DISPLAY}")
        else:
            st.info(f"Current Location: {SCAN_ROOT_DISPLAY} / {relative_display}")

        # --- List Contents of Current Directory (Shallow Scan) ---
        contents = []
        if current_dir != SCAN_ROOT:
            contents.append(".. (Go Up)")

        # Use st.spinner for visual feedback during the quick directory scan
        with st.spinner(f"Scanning directory: {os.path.basename(current_dir)}..."):
            try:
                # Use os.listdir to get only the immediate contents (fast)
                for item in sorted(os.listdir(current_dir)):
                    item_path = os.path.join(current_dir, item)

                    if item in EXCLUDE_DIRS or item.startswith('.'):
                        continue

                    if os.path.isdir(item_path):
                        contents.append(f"📁 {item}")
                    elif item.endswith('.csv'):
                        contents.append(f"📄 {item}")

            except Exception as e:
                st.warning(f"Error accessing directory: {e}")
                contents = [".. (Go Up)"]

        # --- Dynamic Navigation Widget ---

        # FIX: Add a placeholder to stabilize the selectbox after rerun
        options_with_placeholder = ["--- Select Action ---"] + contents

        selected_item = st.selectbox(
            "Select an item to view or a CSV file to load:",
            options=options_with_placeholder,
            index=0,  # Forces placeholder to be the default selection
            key=f"file_browser_{current_dir}"
        )

        # --- Handle Selection ---
        if selected_item and selected_item != "--- Select Action ---":
            # Only proceed if the user made an explicit choice

            item_name = selected_item.split(" ", 1)[-1]
            path_changed = False

            if selected_item == ".. (Go Up)":
                parent_dir = os.path.dirname(current_dir)
                if len(parent_dir) >= len(SCAN_ROOT):
                    st.session_state['browser_current_path'] = parent_dir
                    path_changed = True

            elif selected_item.startswith("📁"):
                new_path = os.path.join(current_dir, item_name)
                st.session_state['browser_current_path'] = new_path
                path_changed = True

            elif selected_item.startswith("📄"):
                file_path = os.path.join(current_dir, item_name)
                st.session_state['current_file_path'] = file_path
                st.success(f"CSV selected: {item_name}")
                # NO st.rerun() here

            if path_changed:
                st.rerun()

    # Return the LazyFrame only if a valid file_path was determined
    if file_path:
        return get_lazy_data_reader(file_path)
    return None


# --- MAIN EXECUTION ---

# Sidebar for data source selection
with st.sidebar:
    lf_initial = get_data_source()
    if lf_initial is not None:
        st.session_state['current_lazy_frame'] = lf_initial
        st.success("Data source successfully linked in LAZY mode.")

# Assign the current working LazyFrame
lf_current = st.session_state['current_lazy_frame']

if lf_current is not None:
    # Display initial metrics
    try:
        # Eager calculation to show metrics
        row_count = lf_current.select(pl.count()).collect().item()
        col_count = len(lf_current.columns)

        st.header("Dataset Overview")
        c1, c2 = st.columns(2)
        c1.metric("Total Rows (Lazy Count)", f"{row_count:,}")
        c2.metric("Total Columns", f"{col_count:,}")

        st.subheader("Data Preview (First 5 Rows)")
        # Use .fetch() to load only the first few rows safely
        st.dataframe(lf_current.fetch(5).to_pandas(), use_container_width=True)

    except Exception as e:
        st.error(f"Could not calculate initial metrics. Error: {e}")
        st.session_state['current_lazy_frame'] = None

    st.markdown("---")

    # --- THE TABBED PIPELINE ---
    cleaning_tab, training_tab, results_tab = st.tabs([
        "🧹 Data Cleaning",
        "🧠 Model Training",
        "📈 Results & Metrics"
    ])

    # ----------------------------------------------------
    # 🧹 DATA CLEANING TAB
    # ----------------------------------------------------
    with cleaning_tab:
        st.subheader("2. Data Cleaning & Duplication Removal")

        # --- Column Duplication Check ---
        st.markdown("#### Column Duplication")
        file_path = st.session_state.get('current_file_path')
        lf_temp = lf_current

        if file_path:
            total_cols, duplicate_names = get_duplicate_columns(file_path)

            if duplicate_names:
                st.warning(f"⚠️ Found Duplicate Columns: {', '.join(duplicate_names)}. Recommended to drop.")

                if st.checkbox("Automatically drop duplicate columns", value=True, key="col_drop_check"):
                    lf_temp = drop_duplicate_columns_lazy(lf_temp, duplicate_names)
                    st.success(f"Column drop applied lazily. Remaining columns: {len(lf_temp.columns)}")

            else:
                st.success("✅ No Duplicate Columns Found in header.")

        # --- Row Duplication Check ---
        st.markdown("#### Row Duplication")

        if st.button("Calculate Row Duplicates", key="row_calc_btn"):
            with st.spinner("Counting rows and duplicates..."):
                total_rows, duplicate_rows = get_row_and_duplicate_counts(lf_temp)
                st.session_state['total_rows'] = total_rows
                st.session_state['duplicate_rows'] = duplicate_rows

        if 'duplicate_rows' in st.session_state and st.session_state['total_rows'] > 0:
            total = st.session_state['total_rows']
            removed = st.session_state['duplicate_rows']

            st.metric("Initial Total Rows", f"{total:,}")
            st.metric("Detected Duplicate Rows", f"{removed:,}", delta_color="inverse")

            if removed > 0:
                if st.checkbox("Automatically remove duplicate rows", value=True, key="row_drop_check"):
                    lf_temp = drop_duplicate_rows_lazy(lf_temp)
                    st.success(f"Row duplication removal applied lazily. Final projected rows: {total - removed:,}")

        # Update the main working LazyFrame for the next step
        st.session_state['current_lazy_frame'] = lf_temp

    # ----------------------------------------------------
    # 🧠 MODEL TRAINING TAB (Placeholder)
    # ----------------------------------------------------
    with training_tab:
        st.header("3. Model Training (Feature Engineering Next)")
        st.write("Current LazyFrame ready for feature engineering and training.")
        st.code(f"LazyFrame steps currently applied: {lf_current.explain()}")

    # ----------------------------------------------------
    # 📈 RESULTS & METRICS TAB (Placeholder)
    # ----------------------------------------------------
    with results_tab:
        st.header("4. Results and Download")
        st.write("Final metrics and download buttons will go here.")


else:
    st.info("👈 Please select or upload your data source in the sidebar to begin.")