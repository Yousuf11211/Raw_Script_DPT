import os
import streamlit as st
import psutil
import polars as pl

EXCLUDE_DIRS = ['temp_uploads', 'venv', 'env', '.git', '__pycache__', '.idea/']
SCAN_ROOT_DISPLAY = "PROJECT ROOT"

# -----------------------------
# Global UI helpers (navigation, state, metrics)
# -----------------------------

def inject_global_styles(hide_builtin_sidebar_nav: bool = True):
    """Inject CSS for consistent theming. Hides Streamlit default page list if requested.
    Updated to avoid emoji usage and style custom top navigation buttons.
    """
    css = [
        """
        <style>
          .block-container { padding-top: 0.8rem; }
          /* Hide default multipage sidebar */
          %HIDE_SIDEBAR%
          /* Top nav container */
          .top-nav { display:flex; flex-wrap:wrap; gap:0.5rem; margin:0.75rem 0 1.0rem 0; }
          .top-nav .nav-group { display:flex; flex-wrap:wrap; gap:0.4rem; width:100%; }
          .nav-btn { cursor:pointer; background:#1f2937; color:#f1f5f9; padding:0.55rem 0.90rem; border-radius:6px; font-size:0.85rem; border:1px solid #374151; text-decoration:none; line-height:1.1; transition:background .15s, transform .15s, box-shadow .15s; }
          .nav-btn:hover { background:#2563eb; transform:translateY(-2px); }
          .nav-btn.active { background:#2563eb; box-shadow:0 0 0 2px #1e3a8a; }
          .nav-category { background:#111827; color:#e5e7eb; padding:0.55rem 0.85rem; border-radius:6px; border:1px solid #334155; font-size:0.80rem; cursor:pointer; transition:background .15s, transform .15s; }
          .nav-category:hover { background:#1e3a8a; }
          .nav-category.active { background:#2563eb; box-shadow:0 0 0 2px #1e3a8a; }
        </style>
        """
    ]
    if hide_builtin_sidebar_nav:
        hide_css = 'section[data-testid="stSidebarNav"] { display:none !important; }'
    else:
        hide_css = ''
    st.markdown(css[0].replace('%HIDE_SIDEBAR%', hide_css), unsafe_allow_html=True)

# Page mapping (labels without emojis)
GLOBAL_MENU = {
    "Data Cleaning": [
        ("Data Validation & Dedup", "pages/1_Data_Validation_and_Dedup.py"),
        ("INF Handling", "pages/2_INF_Handling.py"),
        ("Dominance & Reports", "pages/3_Dominance_and_Reports.py"),
        ("Constant & Low-Variance", "pages/4_Constant_and_LowVariance.py"),
        ("Mixed-Type Analysis", "pages/5_Mixed_Type_Analysis.py"),
        ("Encoding Candidates", "pages/6_Encoding_Candidates.py"),
        ("Delete Columns", "pages/10_Delete_Columns_UI.py"),
    ],
    "Data Processing": [
        ("Class Balancing", "pages/7_Class_Balancing.py"),
        ("Downscale Dataset", "pages/8_Downscale_Dataset.py"),
        ("Separate & Save Sets", "pages/12_Separate_and_Save_Sets.py"),
        ("Merge & Shuffle (Polars)", "pages/17_Merge_Shuffle_Polars.py"),
        ("Outlier Detection (IQR)", "pages/18_Outlier_Detection.py"),
    ],
    "Data Analysis": [
        ("Feature Importance", "pages/11_Feature_Importance.py"),
    ],
    "Model Training": [
        ("Hyperparameter Tuning", "pages/13_Hyperparameter_Tuning.py"),
        ("Isolation Forest Train", "pages/14_Isolation_Forest.py"),
        ("Attack Model Train & Test", "pages/15_Attack_Model_Train_Test.py"),
    ],
    "Model Testing": [
        ("Test Isolation Forest", "pages/16_Test_Isolation_Forest.py"),
        ("SHAP Explanations", "pages/19_SHAP_Explanations.py"),
    ],
    "Utilities": [
        ("Compare Raw vs Processed", "pages/9_Compare_Raw_vs_Processed.py"),
        ("Frontend Test Batch Generator", "pages/20_Frontend_Test_Batch_Generator.py"),
    ],
}

# Reverse map from path to category for active detection
PATH_TO_CATEGORY = {path: cat for cat, items in GLOBAL_MENU.items() for _label, path in items}


def render_top_nav(current_page: str | None = None):
    """Render a two-tier top navigation: categories then page buttons for the active category.
    - current_page: file path (e.g., 'pages/1_Data_Validation_and_Dedup.py') used to auto-select category & active button
    State key: 'nav_active_category'
    """
    inferred_cat = PATH_TO_CATEGORY.get(current_page)
    if 'nav_active_category' not in st.session_state:
        st.session_state['nav_active_category'] = inferred_cat or list(GLOBAL_MENU.keys())[0]
    if inferred_cat and inferred_cat != st.session_state['nav_active_category']:
        st.session_state['nav_active_category'] = inferred_cat

    active_cat = st.session_state['nav_active_category']

    # Category row (pure HTML spans without JS; fallback buttons below handle interaction)
    cat_html = '<div class="top-nav">'
    for cat in GLOBAL_MENU.keys():
        active_cls = 'nav-category active' if cat == active_cat else 'nav-category'
        cat_html += f'<span class="{active_cls}">{cat}</span>'
    cat_html += '</div>'
    st.markdown(cat_html, unsafe_allow_html=True)

    # Fallback interactive category buttons (Streamlit stateful)
    cat_cols = st.columns(len(GLOBAL_MENU))
    for i, cat in enumerate(GLOBAL_MENU.keys()):
        with cat_cols[i]:
            if st.button(cat, key=f'cat_btn_{cat}', type='secondary', help=f'Switch to {cat} tools'):
                st.session_state['nav_active_category'] = cat
                st.experimental_rerun()

    # Page buttons for active category
    page_items = GLOBAL_MENU.get(active_cat, [])
    st.markdown('<div class="top-nav nav-group">', unsafe_allow_html=True)
    for label, path in page_items:
        is_active = (current_page == path)
        btn_cols = st.columns(1)
        with btn_cols[0]:
            if st.button(label, key=f'page_btn_{path}', help=f'Open {label}', type='primary' if is_active else 'secondary'):
                try:
                    st.switch_page(path)
                except Exception:
                    st.error(f"Failed to switch to {label} ({path})")
    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics row (compact)
    m = get_resource_metrics()
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("CPU %", f"{m['CPU %']:.1f}%")
    mc2.metric("RAM %", f"{m['RAM %']:.1f}%")
    mc3.metric("RAM Used", f"{m['RAM Used (GB)']:.2f} GB")
    st.caption("Developer: Syed Yousuf Uddin")


def render_global_nav(active_page_hint: str | None = None, show_metrics: bool = True):
    """Legacy sidebar nav (kept for fallback)."""
    with st.sidebar:
        st.write("Navigation moved to top bar. (Legacy sidebar hidden)")
        if show_metrics:
            m = get_resource_metrics()
            st.metric("CPU %", f"{m['CPU %']:.1f}%")
            st.metric("RAM %", f"{m['RAM %']:.1f}%")


def initialize_state():
    if 'browser_current_path' not in st.session_state:
        st.session_state['browser_current_path'] = os.getcwd()
    if 'current_lazy_frame' not in st.session_state:
        st.session_state['current_lazy_frame'] = None
    if 'current_file_path' not in st.session_state:
        st.session_state['current_file_path'] = None
    if 'applied_filters' not in st.session_state:
        st.session_state['applied_filters'] = []


def get_resource_metrics():
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
    try:
        return pl.scan_csv(file_path)
    except Exception as e:
        st.error(f"Error reading data lazily from path: {file_path}. Details: {e}")
        return None

@st.cache_data
def save_uploaded_file_to_temp(uploaded_file):
    temp_dir = 'temp_uploads'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def data_source_selector(label: str = "Select your data source:"):
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
    if f'{state_prefix}_current_path' not in st.session_state:
        st.session_state[f'{state_prefix}_current_path'] = root_dir
    current_dir = st.session_state[f'{state_prefix}_current_path']
    rel = os.path.relpath(current_dir, root_dir)
    st.caption(f"{label}: {SCAN_ROOT_DISPLAY}" + ("" if rel == '.' else f" / {rel}"))
    items = []
    if current_dir != root_dir:
        items.append('.. (Go Up)')
    if allow_select_current_dir:
        items.append('[Select] Use this folder')
    try:
        for item in sorted(os.listdir(current_dir)):
            item_path = os.path.join(current_dir, item)
            if item in EXCLUDE_DIRS or item.startswith('.'):
                continue
            if os.path.isdir(item_path):
                items.append(f"[DIR] {item}")
            else:
                if allowed_exts is None:
                    items.append(f"[FILE] {item}")
                else:
                    low = item.lower()
                    if any(low.endswith(ext) for ext in allowed_exts):
                        items.append(f"[FILE] {item}")
    except Exception as e:
        st.warning(f"Error accessing {current_dir}: {e}")
        items = ['.. (Go Up)'] + (['[Select] Use this folder'] if allow_select_current_dir else [])
    selection = st.selectbox(f"{label} (navigate and select)", ['---'] + items, key=f"{state_prefix}_select")
    chosen = None
    if selection != '---':
        if selection == '.. (Go Up)':
            parent = os.path.dirname(current_dir)
            if len(parent) >= len(root_dir):
                st.session_state[f'{state_prefix}_current_path'] = parent
            st.rerun()
        elif selection.startswith('[Select]'):
            chosen = current_dir
            st.success(f"Selected folder: {os.path.basename(current_dir) or current_dir}")
        elif selection.startswith('[DIR]'):
            folder_name = selection.split(' ', 1)[-1]
            st.session_state[f'{state_prefix}_current_path'] = os.path.join(current_dir, folder_name)
            st.rerun()
        elif selection.startswith('[FILE]'):
            file_name = selection.split(' ', 1)[-1]
            chosen = os.path.join(current_dir, file_name)
            st.success(f"Selected: {file_name}")
    return chosen

def common_header(page_title: str, num_inputs: int = 1, input_labels=None, default_output_folder: str = "output", input_specs=None):
    root_dir = os.getcwd()
    if input_labels is None and input_specs is None:
        input_labels = [f"Input File {i+1}" for i in range(num_inputs)]
    st.title(page_title)  # Title now expected to have no emojis
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
