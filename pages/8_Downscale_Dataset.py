# Moved from root pages/8_Downscale_Dataset.py
import streamlit as st
import os
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav, common_header
from utils.downscale import (
    downscale_from_folder,
    downscale_from_file,
    downscale_from_lazyframe,
    DEFAULT_INPUT_FOLDER,
    DEFAULT_OUTPUT_FOLDER,
)

st.set_page_config(page_title="Downscale Dataset", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="pages/8_Downscale_Dataset.py")

hdr = common_header(
    "Downscale Dataset (Create Small Benign/Attack Sets)",
    num_inputs=1,
    input_specs=[{"label": "Input folder (optional)", "kind": "folder"}],
    default_output_folder=DEFAULT_OUTPUT_FOLDER
)
sel_input_folder = hdr['input_paths'][0]
sel_output_folder = hdr['output_folder'] or DEFAULT_OUTPUT_FOLDER

mode = st.radio("Source Mode", ["Current Loaded File", "Specific File Path", "Scan Folder"], index=0, horizontal=True)

st.subheader("Settings")
col1, col2 = st.columns(2)
with col1:
    if mode == "Scan Folder":
        input_folder = st.text_input("Input folder (recursive)", value=sel_input_folder or DEFAULT_INPUT_FOLDER)
    elif mode == "Specific File Path":
        input_file = st.text_input("Input CSV file", value="")
with col2:
    output_folder = st.text_input("Output folder (two files will be written)", value=sel_output_folder)

col3, col4 = st.columns(2)
with col3:
    benign_frac = st.slider("Benign sampling fraction", 0.01, 1.0, 0.10, 0.01)
with col4:
    rand_state = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

st.markdown("### Attack Selection (Optional)")
colA, colB = st.columns(2)
with colA:
    attack_frac = st.slider("Attack sampling fraction", 0.01, 1.0, 1.00, 0.01, help="Set < 1.0 to sample attacks as well.")
with colB:
    attack_labels_text = st.text_input("Attack labels to include (comma-separated)", value="", help="Leave empty to include all attack labels.")

selected_attack_labels = [s.strip() for s in attack_labels_text.split(',') if s.strip()] or None

current_lf = st.session_state.get('current_lazy_frame')

if st.button("Run Downscale", use_container_width=True):
    result = {}
    if mode == "Current Loaded File":
        if current_lf is None:
            st.error("No file currently loaded. Go to any data page and select a CSV first.")
        else:
            with st.spinner("Collecting current lazyframe and downscaling..."):
                result = downscale_from_lazyframe(
                    lf=current_lf,
                    output_folder=output_folder,
                    benign_sampling_fraction=float(benign_frac),
                    random_state=int(rand_state),
                    attack_sampling_fraction=float(attack_frac),
                    selected_attack_labels=selected_attack_labels,
                )
    elif mode == "Specific File Path":
        if not input_file or not os.path.isfile(input_file):
            st.error("Please provide a valid CSV file path.")
        else:
            with st.spinner("Reading file and downscaling..."):
                result = downscale_from_file(
                    input_file=input_file,
                    output_folder=output_folder,
                    benign_sampling_fraction=float(benign_frac),
                    random_state=int(rand_state),
                    attack_sampling_fraction=float(attack_frac),
                    selected_attack_labels=selected_attack_labels,
                )
    else:  # Scan Folder
        if not os.path.isdir(input_folder):
            st.error(f"Input folder does not exist: {input_folder}")
        else:
            with st.spinner("Walking folder and processing CSV files..."):
                result = downscale_from_folder(
                    input_folder=input_folder,
                    output_folder=output_folder,
                    benign_sampling_fraction=float(benign_frac),
                    random_state=int(rand_state),
                    attack_sampling_fraction=float(attack_frac),
                    selected_attack_labels=selected_attack_labels,
                )

    if result:
        if result.get("error"):
            st.error(f"Downscale failed: {result['error']}")
        else:
            st.success("Downscale complete.")
            st.markdown("---")
            st.subheader("Results")
            st.write(f"Benign rows: {result.get('benign_rows', 0):,}")
            st.write(f"Attack rows: {result.get('attacks_rows', 0):,}")
            st.caption(f"Benign fraction: {float(benign_frac):.2f} | Attack fraction: {float(attack_frac):.2f} | Attack labels: {selected_attack_labels or 'ALL'}")
            if "benign_label_counts" in result:
                st.markdown("#### Benign label counts")
                st.json(result["benign_label_counts"])
            if "attacks_label_counts" in result:
                st.markdown("#### Attack label counts")
                st.json(result["attacks_label_counts"])
            benign_path = result.get('output_benign_path')
            attacks_path = result.get('output_attacks_path')
            if benign_path and os.path.isfile(benign_path):
                with open(benign_path, 'rb') as f:
                    st.download_button("Download benign.csv", f, file_name=os.path.basename(benign_path))
            if attacks_path and os.path.isfile(attacks_path):
                with open(attacks_path, 'rb') as f:
                    st.download_button("Download attacks.csv", f, file_name=os.path.basename(attacks_path))
            with st.expander("Per-file summary"):
                st.json(result.get('per_file', []))
