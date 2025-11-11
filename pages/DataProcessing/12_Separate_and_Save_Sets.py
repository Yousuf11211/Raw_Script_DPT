# Moved from root pages/12_Separate_and_Save_Sets.py
import streamlit as st
import os
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav, common_header
from utils import (
    sep_analyze_and_classify,
    sep_process_combined,
    sep_process_proportional,
    SEP_DEFAULT_INPUT_FOLDER,
    SEP_DEFAULT_OUTPUT_FOLDER,
)

st.set_page_config(page_title="Separate and Save Benign/Attack Sets", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="pages/DataProcessing/12_Separate_and_Save_Sets.py")

hdr = common_header(
    "Separate and Save Benign/Attack Sets",
    num_inputs=1,
    input_specs=[{"label": "Input folder", "kind": "folder"}],
    default_output_folder=SEP_DEFAULT_OUTPUT_FOLDER
)
sel_input_folder = hdr['input_paths'][0]
sel_output_folder = hdr['output_folder'] or SEP_DEFAULT_OUTPUT_FOLDER

st.markdown("Split large folders into Benign and Attack output files with your chosen size and strategy.")

col1, col2 = st.columns(2)
with col1:
    input_folder = st.text_input("Input folder", value=sel_input_folder or SEP_DEFAULT_INPUT_FOLDER)
with col2:
    output_folder = st.text_input("Output folder", value=sel_output_folder)

mode = st.radio("Processing mode", ["benign", "attacks", "both"], index=2, horizontal=True)
rows_per_file = st.number_input("Rows per output file", min_value=10_000, max_value=5_000_000, value=500_000, step=10_000)
shuffle = st.checkbox("Shuffle outputs", value=True)

if st.button("Analyze Folder", use_container_width=True):
    if not os.path.isdir(input_folder):
        st.error(f"Input folder not found: {input_folder}")
    else:
        with st.spinner("Analyzing folder..."):
            all_files = [os.path.join(root, f) for root, _, files in os.walk(input_folder) for f in files if f.lower().endswith('.csv')]
            total_counts, files_by_label, actual_label_col, skipped = sep_analyze_and_classify(all_files, mode)
        if not actual_label_col:
            st.error("No label column detected. Ensure files contain a 'label' column (case-insensitive).")
        else:
            st.success("Analysis done.")
            colA, colB = st.columns(2)
            with colA:
                st.subheader("Total Row Counts by Label")
                for lbl, cnt in sorted(total_counts.items()):
                    st.write(f"{lbl}: {cnt:,}")
            with colB:
                st.subheader("Files by Label (counts)")
                for lbl, paths in files_by_label.items():
                    st.write(f"{lbl}: {len(paths)} files")
            if skipped:
                with st.expander("Skipped files"):
                    st.write(skipped)
            st.session_state['sep_analysis'] = {
                'files_by_label': files_by_label,
                'actual_label_col': actual_label_col,
                'total_counts': total_counts,
                'input_folder': input_folder,
                'output_folder': output_folder,
            }

state = st.session_state.get('sep_analysis')
if state:
    files_by_label = state['files_by_label']
    actual_label_col = state['actual_label_col']
    benign_label = 'benign' if 'benign' in files_by_label else next((k for k in files_by_label.keys() if str(k).lower()== 'benign'), None)
    attack_labels = [k for k in files_by_label.keys() if str(k).lower() != 'benign']

    st.markdown("---")
    st.subheader("Run Separation")

    colB1, colB2 = st.columns(2)
    with colB1:
        st.markdown("#### Benign (Memory Efficient)")
        if benign_label and files_by_label.get(benign_label):
            if st.button("Create Benign files", use_container_width=True):
                out_dir = os.path.join(state['output_folder'], 'Benign')
                res = sep_process_combined(
                    file_list=files_by_label[benign_label],
                    rows_per_output_file=int(rows_per_file),
                    labels_to_keep=[benign_label],
                    output_group_name='Benign',
                    output_base_path=out_dir,
                    should_shuffle=shuffle,
                    actual_label_col_name=actual_label_col
                )
                st.success(f"Created {len(res['created_files'])} Benign files ({res['total_rows']:,} rows)")
                with st.expander("Created Benign files"):
                    for p in res['created_files']:
                        st.write(p)
        else:
            st.info("No Benign label found in analysis.")

    with colB2:
        st.markdown("#### Attacks (Proportional Mix)")
        if attack_labels:
            if st.button("Create Attack files", use_container_width=True):
                out_dir = os.path.join(state['output_folder'], 'Attacks')
                files = sorted(list({f for lbl in attack_labels for f in files_by_label.get(lbl, [])}))
                res = sep_process_proportional(
                    file_list=files,
                    rows_per_output_file=int(rows_per_file),
                    labels_to_keep=attack_labels,
                    output_group_name='Attacks',
                    output_base_path=out_dir,
                    should_shuffle=shuffle,
                    actual_label_col_name=actual_label_col
                )
                st.success(f"Created {len(res['created_files'])} Attack files ({res['total_rows']:,} rows)")
                with st.expander("Created Attack files"):
                    for p in res['created_files']:
                        st.write(p)
        else:
            st.info("No attack labels found in analysis.")

