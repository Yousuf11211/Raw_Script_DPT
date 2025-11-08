import streamlit as st
import polars as pl
import psutil
import os
from utils.data_cleaning import (
    get_duplicate_columns,
    drop_duplicate_columns_lazy,
    get_row_and_duplicate_counts,
    drop_duplicate_rows_lazy,
    get_validation_report_and_filter_plan
)
from utils.data_analysis import get_class_distribution_report, get_dominance_report, get_value_label_breakdown
from utils.data_quality import analyze_inf_columns, drop_inf_columns_lazy, impute_inf_with_median

# --- CONSTANTS AND INITIALIZATION ---

# Define the root path for scanning (your project directory)
SCAN_ROOT = os.getcwd()
# Exclude folders that are not data (like the temp folder or virtual environment)
EXCLUDE_DIRS = ['temp_uploads', 'venv', 'env', '.git', '__pycache__', '.idea/']
SCAN_ROOT_DISPLAY = "PROJECT ROOT"
LOCAL_DATA_PATH = os.path.join(os.getcwd(), 'data',
                               'fixed_thesis_data.csv')

# Initialize state variables
if 'browser_current_path' not in st.session_state:
    st.session_state['browser_current_path'] = SCAN_ROOT

if 'current_lazy_frame' not in st.session_state:
    st.session_state['current_lazy_frame'] = None

# Added initialization for applied filters tracking
if 'applied_filters' not in st.session_state:
    st.session_state['applied_filters'] = []

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

def get_resource_metrics():
    """Fetches key CPU and Memory metrics."""
    cpu_usage = psutil.cpu_percent(interval=None)  # CPU usage across all cores
    memory_info = psutil.virtual_memory()

    return {
        "CPU %": cpu_usage,
        "RAM Used (GB)": memory_info.used / (1024 ** 3),
        "RAM Total (GB)": memory_info.total / (1024 ** 3),
        "RAM %": memory_info.percent
    }


# --- INTEGRATION INTO YOUR SIDEBAR (Example) ---
# Add this section to your sidebar block:

with st.sidebar:
    # ... (Your get_data_source() function and logic) ...

    st.markdown("---")
    st.header("System Health (Live)")

    # 1. Fetch the metrics
    metrics = get_resource_metrics()

    # 2. Display them using columns and st.metric
    col_cpu, col_ram = st.columns(2)

    with col_cpu:
        st.metric("CPU Usage", f"{metrics['CPU %']:.1f}%")

    with col_ram:
        st.metric("RAM Used", f"{metrics['RAM Used (GB)']:.2f} GB")
        st.metric("RAM %", f"{metrics['RAM %']:.1f}%")

    # The use of st.rerun() or st.empty() is complex for continuous live updates.
    # For simplicity, these metrics update every time the main script runs (e.g., when a button is clicked).
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
        # FIX: Implement Polars Deprecation Fix: use .limit(5).collect() and width='stretch'
        st.dataframe(lf_current.limit(5).collect().to_pandas(), width='stretch')

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

        if st.session_state['current_lazy_frame'] is None:
            st.warning("Please load a dataset in the Data Selection tab first.")
        else:
            lf = st.session_state['current_lazy_frame']
            initial_row_count = lf.select(pl.count()).collect().item()
            st.markdown(f"**Current Rows:** {initial_row_count:,}")
            st.markdown(f"**Current Columns:** {len(lf.columns):,}")
            st.markdown("---")

            st.subheader("Row Validation (Negative Values & Ports)")
            st.markdown("This check identifies and removes rows where: "
                        "\n- A non-negative field (e.g., `duration`, `count`) is less than 0."
                        "\n- A port field (`src_port`, `dst_port`) is outside the 0-65535 range.")

            if st.button("Run Data Validation & Filter", key="run_validation_filter", use_container_width=True):
                with st.spinner("Executing EAGER query to find invalid rows..."):
                    # CALL THE NEW POLARS FUNCTION
                    lf_validated, report = get_validation_report_and_filter_plan(lf)

                st.session_state['validation_report'] = report
                st.session_state['current_lazy_frame'] = lf_validated
                st.session_state['applied_filters'].append("Row Validation & Filtering")
                st.rerun()  # Rerun to display persistent results

            # Display the results of the last validation run
            report = st.session_state.get('validation_report')
            if report and report['invalid_count'] > 0:
                st.error(f"**{report['invalid_count']:,} Rows Removed.**")
                st.markdown("###### Label Breakdown of Removed Rows:")
                st.json(report['label_breakdown'])

            elif report and report['invalid_count'] == 0:
                st.success("Validation complete. No invalid rows found.")

            st.markdown("---")

        # Initialize session state keys to control visibility and storage of cleaning results
        if 'col_check_done' not in st.session_state: st.session_state['col_check_done'] = False
        if 'row_check_done' not in st.session_state: st.session_state['row_check_done'] = False

        file_path = st.session_state.get('current_file_path')
        lf_temp = st.session_state['current_lazy_frame']  # Use the current LF

        if not file_path:
            st.warning("Please select a file in the sidebar to begin checks.")

        # --- Control Buttons for Eager Checks ---
        st.markdown("#### Run Data Checks")
        col1, col2 = st.columns(2)

        # --- Column Check Button ---
        with col1:
            if st.button("Check Column Duplicates", key="col_check_btn", use_container_width=True):
                if file_path:
                    with st.spinner("Scanning file header..."):
                        total_cols, duplicate_names = get_duplicate_columns(file_path)
                        st.session_state['total_cols'] = total_cols
                        st.session_state['duplicate_names'] = duplicate_names
                        st.session_state['col_check_done'] = True
                        st.rerun()  # Rerun to display results below

        # --- Row Check Button ---
        with col2:
            if st.button("Calculate Row Duplicates", key="row_calc_btn", use_container_width=True):
                with st.spinner("Counting rows and duplicates..."):
                    # This is the heavy EAGER calculation
                    total_rows, duplicate_rows = get_row_and_duplicate_counts(lf_temp)
                    st.session_state['total_rows'] = total_rows
                    st.session_state['duplicate_rows'] = duplicate_rows
                    st.session_state['row_check_done'] = True
                    # NO st.rerun() here

        st.markdown("---")

        # --- 1. COLUMN DUPLICATION RESULTS & ACTION ---
        st.markdown("#### Column Duplication Results")
        if st.session_state['col_check_done']:

            total_cols = st.session_state.get('total_cols', 0)
            duplicate_names = st.session_state.get('duplicate_names', None)

            st.metric("Total Columns Scanned", total_cols)

            if duplicate_names:
                st.warning(f"⚠️ Found {len(duplicate_names)} Duplicate Columns: {', '.join(duplicate_names)}.")

                if st.checkbox("Automatically drop duplicate columns (Lazy)", value=True, key="col_drop_check"):
                    lf_temp = drop_duplicate_columns_lazy(lf_temp, duplicate_names)
                    # Update the projected column count based on the current LF object (lazy operation)
                    st.success(f"Column drop rule added to plan. Projected columns: {len(lf_temp.columns)}")

            else:
                st.success("✅ No Duplicate Columns Found in header.")
        else:
            st.info("Click 'Check Column Duplicates' above to scan file header.")

        st.markdown("---")

        # --- 2. ROW DUPLICATION RESULTS & ACTION ---
        st.markdown("#### Row Duplication Results")
        if st.session_state['row_check_done']:

            total = st.session_state['total_rows']
            removed = st.session_state['duplicate_rows']

            c3, c4 = st.columns(2)
            c3.metric("Initial Total Rows", f"{total:,}")
            c4.metric("Detected Duplicate Rows", f"{removed:,}", delta_color="inverse")

            if removed > 0:
                if st.checkbox("Automatically remove duplicate rows (Lazy)", value=True, key="row_drop_check"):
                    lf_temp = drop_duplicate_rows_lazy(lf_temp)
                    st.success(f"Row duplication removal rule added to plan. Final projected rows: {total - removed:,}")

            else:
                st.success("✅ No Duplicate Rows Found.")
        else:
            st.info("Click 'Calculate Row Duplicates' above to run the full count aggregation.")

        # Update the main working LazyFrame for the next step
        st.session_state['current_lazy_frame'] = lf_temp

        # INF Columns Analysis & Handling Section
        st.markdown("---")
        st.subheader("INF Value Analysis & Handling")
        st.markdown("Identify columns containing infinite (+/-inf) values and optionally drop or impute them.")

        if st.session_state['current_lazy_frame'] is not None:
            lf_inf = st.session_state['current_lazy_frame']
            if st.button("Analyze INF Columns", key="analyze_inf_btn", use_container_width=True):
                with st.spinner("Scanning for INF values (numeric columns)..."):
                    total_rows_inf, inf_report_df = analyze_inf_columns(lf_inf)
                st.session_state['inf_total_rows'] = total_rows_inf
                st.session_state['inf_report_df'] = inf_report_df

            inf_report_df = st.session_state.get('inf_report_df')
            if inf_report_df is not None:
                if inf_report_df.empty:
                    st.success("No INF values detected in numeric columns.")
                else:
                    st.dataframe(inf_report_df, use_container_width=True)

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        threshold = st.number_input("Drop Threshold (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                                                     help="Columns with INF percentage above this threshold will be dropped lazily.")
                    with col_b:
                        if st.button("Drop Columns Above Threshold", key="drop_inf_cols", use_container_width=True):
                            lf_new, dropped = drop_inf_columns_lazy(lf_inf, threshold_percent=threshold)
                            if dropped:
                                st.session_state['current_lazy_frame'] = lf_new
                                st.session_state['applied_filters'].append(f"Drop INF > {threshold:.1f}%: {len(dropped)} cols")
                                st.success(f"Scheduled lazy drop of {len(dropped)} INF-heavy columns.")
                            else:
                                st.info("No columns exceeded the threshold.")
                    with col_c:
                        if st.button("Impute INF with Median", key="impute_inf_btn", use_container_width=True):
                            lf_imp, medians = impute_inf_with_median(lf_inf)
                            if medians:
                                st.session_state['current_lazy_frame'] = lf_imp
                                st.session_state['applied_filters'].append(f"Impute INF medians ({len(medians)} cols)")
                                st.success(f"Prepared lazy imputation for {len(medians)} columns.")
                                with st.expander("Median Values Used"):
                                    st.json(medians)
                            else:
                                st.info("No INF values requiring imputation detected.")

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
        st.header("4. Data Analysis & Final Reports")
        lf_final = st.session_state['current_lazy_frame']
        if lf_final is None:
            st.warning("Please ensure a dataset is loaded and cleaned in the previous tabs.")
        else:
            st.subheader("Applied Cleaning Steps")
            if st.session_state['applied_filters']:
                st.write(st.session_state['applied_filters'])
            else:
                st.write("No cleaning steps applied yet.")

            # Optional: Export current dataset (sample to avoid OOM)
            with st.expander("Export Current Dataset (sample)"):
                max_rows = int(lf_final.select(pl.count()).collect().item())
                sample_n = st.number_input("Rows to export (sample)", min_value=1000, max_value=max_rows, value=min(100000, max_rows), step=1000)
                if st.button("Prepare CSV", key="prepare_csv_btn"):
                    with st.spinner(f"Collecting {sample_n:,} rows and preparing CSV..."):
                        df_export = lf_final.limit(int(sample_n)).collect().to_pandas()
                        csv_bytes = df_export.to_csv(index=False).encode('utf-8')
                        st.session_state['export_csv_bytes'] = csv_bytes
                if 'export_csv_bytes' in st.session_state:
                    st.download_button("Download CSV", data=st.session_state['export_csv_bytes'], file_name="dataset_sample.csv", mime="text/csv")

            st.markdown("---")
            st.subheader("Class Distribution")
            if st.button("Compute Class Distribution", key="class_dist_btn", use_container_width=True):
                with st.spinner("Aggregating label counts..."):
                    class_df, fig = get_class_distribution_report(lf_final)
                st.session_state['class_df'] = class_df
                st.session_state['class_fig'] = fig
            if 'class_df' in st.session_state and not st.session_state['class_df'].empty:
                st.dataframe(st.session_state['class_df'], use_container_width=True)
                if st.session_state['class_fig'] is not None:
                    st.pyplot(st.session_state['class_fig'])

            st.markdown("---")
            st.subheader("Column Dominance Report")
            if st.button("Generate Dominance Report", key="dominance_btn", use_container_width=True):
                with st.spinner("Running dominance aggregation across all columns..."):
                    dominance_summary, label_df = get_dominance_report(lf_final)
                st.session_state['dominance_summary'] = dominance_summary
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
                        feature = st.selectbox("Select a feature to inspect:", options=list(lf_final.columns))
                        topn = st.number_input("Top values to show", min_value=5, max_value=100, value=10)
                        if st.button("Compute Breakdown", key="compute_breakdown_btn"):
                            with st.spinner("Aggregating value/label breakdown..."):
                                breakdown_df = get_value_label_breakdown(lf_final, feature=feature, top_n=int(topn))
                            st.dataframe(breakdown_df, use_container_width=True)
                            st.download_button("Download Breakdown (CSV)", data=breakdown_df.to_csv(index=False).encode('utf-8'), file_name=f"{feature}_value_label_breakdown.csv", mime="text/csv")
                else:
                    st.info("Dominance report is empty or failed.")


else:
    st.info("👈 Please select or upload your data source in the sidebar to begin.")