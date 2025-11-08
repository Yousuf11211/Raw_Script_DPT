import os
import streamlit as st
import psutil
import polars as pl

EXCLUDE_DIRS = ['temp_uploads', 'venv', 'env', '.git', '__pycache__', '.idea/']
SCAN_ROOT_DISPLAY = "PROJECT ROOT"


def initialize_state():
    """Initialize common session state keys used across pages."""
    if 'browser_current_path' not in st.session_state:
        st.session_state['browser_current_path'] = os.getcwd()
    if 'current_lazy_frame' not in st.session_state:
        st.session_state['current_lazy_frame'] = None
    if 'current_file_path' not in st.session_state:
        st.session_state['current_file_path'] = None
    if 'applied_filters' not in st.session_state:
        st.session_state['applied_filters'] = []


def get_resource_metrics():
    """Return current CPU and RAM metrics for sidebar display."""
    cpu_usage = psutil.cpu_percent(interval=None)
    memory_info = psutil.virtual_memory()
    return {
        "CPU %": cpu_usage,
        "RAM Used (GB)": memory_info.used / (1024 ** 3),
        "RAM Total (GB)": memory_info.total / (1024 ** 3),
        "RAM %": memory_info.percent,
    }


@st.cache_resource
def get_lazy_data_reader(file_path: str):
    """Return a Polars LazyFrame from a CSV path, or None on error."""
    try:
        return pl.scan_csv(file_path)
    except Exception as e:
        st.error(f"Error reading data lazily from path: {file_path}. Details: {e}")
        return None


@st.cache_data
def save_uploaded_file_to_temp(uploaded_file):
    """Save the uploaded file to a temporary folder and return its path."""
    temp_dir = 'temp_uploads'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def data_source_selector(label: str = "Select your data source:"):
    """Render Upload/Browse UI and return a LazyFrame when a file is selected."""
    SCAN_ROOT = os.getcwd()

    st.subheader("Data Source")
    data_source = st.radio(label, ("Upload CSV", "Browse Project Files"), index=0, key="data_source_radio")

    file_path = None

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file to upload:", type="csv", key="csv_uploader")
        if uploaded_file is not None:
            file_path = save_uploaded_file_to_temp(uploaded_file)
            st.session_state['current_file_path'] = file_path

    elif data_source == "Browse Project Files":
        current_dir = st.session_state['browser_current_path']
        relative_display = os.path.relpath(current_dir, SCAN_ROOT)
        if relative_display == ".":
            st.info(f"Current Location: {SCAN_ROOT_DISPLAY}")
        else:
            st.info(f"Current Location: {SCAN_ROOT_DISPLAY} / {relative_display}")

        contents = []
        if current_dir != SCAN_ROOT:
            contents.append(".. (Go Up)")

        with st.spinner(f"Scanning directory: {os.path.basename(current_dir)}..."):
            try:
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

        options_with_placeholder = ["--- Select Action ---"] + contents
        selected_item = st.selectbox("Select an item to view or a CSV file to load:", options=options_with_placeholder, index=0, key=f"file_browser_{current_dir}")

        if selected_item and selected_item != "--- Select Action ---":
            item_name = selected_item.split(" ", 1)[-1]
            if selected_item == ".. (Go Up)":
                parent_dir = os.path.dirname(current_dir)
                if len(parent_dir) >= len(SCAN_ROOT):
                    st.session_state['browser_current_path'] = parent_dir
                    st.rerun()
            elif selected_item.startswith("📁"):
                new_path = os.path.join(current_dir, item_name)
                st.session_state['browser_current_path'] = new_path
                st.rerun()
            elif selected_item.startswith("📄"):
                file_path = os.path.join(current_dir, item_name)
                st.session_state['current_file_path'] = file_path
                st.success(f"CSV selected: {item_name}")

    if file_path:
        lf = get_lazy_data_reader(file_path)
        if lf is not None:
            st.session_state['current_lazy_frame'] = lf
        return lf

    return None

