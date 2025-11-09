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
    """Render a project directory CSV browser and return a LazyFrame when a file is selected.

    Upload/drag-and-drop removed per requirements. This mirrors the project file browser style.
    """
    SCAN_ROOT = os.getcwd()
    st.subheader("Data Source")
    st.caption("Browse and pick a CSV from the project directory")

    chosen = _browse_path(SCAN_ROOT, state_prefix="home_browser", label=label, allowed_exts=['.csv'], allow_select_current_dir=False)

    if chosen:
        st.session_state['current_file_path'] = chosen
        lf = get_lazy_data_reader(str(chosen))
        if lf is not None:
            st.session_state['current_lazy_frame'] = lf
        return lf

    return None


def _browse_path(root_dir: str, state_prefix: str, label: str, allowed_exts=None, allow_select_current_dir: bool = False):
    """Generic project browser. When allowed_exts is provided, lists only files with those extensions.
    If allow_select_current_dir=True, user can choose the current folder (returns its path).
    Returns selected file/folder path or None.
    """
    if f'{state_prefix}_current_path' not in st.session_state:
        st.session_state[f'{state_prefix}_current_path'] = root_dir
    current_dir = st.session_state[f'{state_prefix}_current_path']
    rel = os.path.relpath(current_dir, root_dir)
    st.caption(f"{label}: {SCAN_ROOT_DISPLAY}" + ("" if rel == '.' else f" / {rel}"))

    items = []
    if current_dir != root_dir:
        items.append('.. (Go Up)')
    if allow_select_current_dir:
        items.append('📌 Use this folder')
    try:
        for item in sorted(os.listdir(current_dir)):
            item_path = os.path.join(current_dir, item)
            if item in EXCLUDE_DIRS or item.startswith('.'):  # skip hidden/system
                continue
            if os.path.isdir(item_path):
                items.append(f"📁 {item}")
            else:
                if allowed_exts is None:
                    items.append(f"📄 {item}")
                else:
                    low = item.lower()
                    if any(low.endswith(ext) for ext in allowed_exts):
                        items.append(f"📄 {item}")
    except Exception as e:
        st.warning(f"Error accessing {current_dir}: {e}")
        items = ['.. (Go Up)'] + (['📌 Use this folder'] if allow_select_current_dir else [])

    selection = st.selectbox(f"{label} (navigate and select)", ['---'] + items, key=f"{state_prefix}_select")
    chosen = None
    if selection != '---':
        if selection == '.. (Go Up)':
            parent = os.path.dirname(current_dir)
            if len(parent) >= len(root_dir):
                st.session_state[f'{state_prefix}_current_path'] = parent
            st.rerun()
        elif selection == '📌 Use this folder':
            chosen = current_dir
            st.success(f"Selected folder: {os.path.basename(current_dir) or current_dir}")
        elif selection.startswith('📁'):
            folder_name = selection.split(' ', 1)[-1]
            st.session_state[f'{state_prefix}_current_path'] = os.path.join(current_dir, folder_name)
            st.rerun()
        elif selection.startswith('📄'):
            file_name = selection.split(' ', 1)[-1]
            chosen = os.path.join(current_dir, file_name)
            st.success(f"Selected: {file_name}")
    return chosen


def common_header(page_title: str, num_inputs: int = 1, input_labels=None, default_output_folder: str = "output", input_specs=None):
    """Render a common header with N inputs (file or folder), an output folder input, and a save toggle.

    input_specs: Optional[List[dict]] where each dict can include:
        - label: str
        - kind: 'file' or 'folder'
        - allowed_exts: List[str] (for kind=='file')
    If input_specs is None, defaults to CSV file inputs.

    Returns dict with keys:
      'input_paths': [path|None,...]
      'output_folder': str
      'save': bool
    """
    root_dir = os.getcwd()
    if input_labels is None and input_specs is None:
        input_labels = [f"Input File {i+1}" for i in range(num_inputs)]
    st.title(page_title)
    st.markdown("#### Data Inputs & Output Settings")

    cols = st.columns(num_inputs + 1)
    selected_paths = []

    for i in range(num_inputs):
        with cols[i]:
            if input_specs and i < len(input_specs):
                spec = input_specs[i]
                label = spec.get('label', f"Input {i+1}")
                kind = spec.get('kind', 'file')
                allowed_exts = spec.get('allowed_exts', ['.csv']) if kind == 'file' else None
                allow_dir = (kind == 'folder')
                chosen = _browse_path(root_dir, state_prefix=f"{page_title}_input_{i}", label=label, allowed_exts=allowed_exts, allow_select_current_dir=allow_dir)
            else:
                label = input_labels[i]
                chosen = _browse_path(root_dir, state_prefix=f"{page_title}_input_{i}", label=label, allowed_exts=['.csv'], allow_select_current_dir=False)
            selected_paths.append(chosen)

    with cols[-1]:
        output_folder = st.text_input("Output folder", value=default_output_folder, key=f"common_header_out_{page_title}")
        save_results = st.checkbox("Save results", value=True, help="Uncheck to run without writing output files.", key=f"common_header_save_{page_title}")
    st.divider()
    return {
        'input_paths': selected_paths,
        'output_folder': output_folder,
        'save': save_results,
    }
