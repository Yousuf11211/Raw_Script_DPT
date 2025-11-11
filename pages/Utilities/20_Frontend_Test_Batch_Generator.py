# Moved from root pages/20_Frontend_Test_Batch_Generator.py
import streamlit as st
import os
import pandas as pd
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav, common_header
from utils.frontend_test_generator import generate_testing_batches

st.set_page_config(page_title="Frontend Test Batch Generator", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="pages/Utilities/20_Frontend_Test_Batch_Generator.py")

header = common_header("Frontend Test Batch Generator", num_inputs=2, input_labels=["Benign CSV", "Attack CSV"], default_output_folder="frontend_testing")

colA, colB, colC, colD = st.columns(4)
with colA:
    benign_limit = st.number_input("Benign load limit", min_value=100, max_value=200000, value=5000, step=100)
with colB:
    num_batches = st.number_input("Number of batches", min_value=1, max_value=200, value=20, step=1)
with colC:
    benign_ratio = st.slider("Benign ratio per batch", 0.5, 0.9, 0.70, 0.01)
with colD:
    rows_min_max = st.text_input("Rows range (min,max)", value="50,100")

benign_label = st.text_input("Benign label value", value="BENIGN")
attack_label = st.text_input("Attack label value", value="ATTACK")

run_btn = st.button("Generate Batches", use_container_width=True)

if run_btn:
    benign_input_path, attack_input_path = header['input_paths']
    save_results = header['save']
    output_folder = header['output_folder']
    if benign_input_path is None or attack_input_path is None:
        st.error("Please select both benign and attack CSV files.")
    else:
        try:
            low_high = [int(x.strip()) for x in rows_min_max.split(',') if x.strip()]
            if len(low_high) != 2:
                raise ValueError("Rows range must be two integers, e.g. 50,100")
            min_rows, max_rows = low_high
            with st.spinner("Reading CSVs and preparing batches..."):
                df_benign = pd.read_csv(benign_input_path)
                df_attack = pd.read_csv(attack_input_path)
                if 'label' not in df_benign.columns:
                    df_benign['label'] = benign_label
                else:
                    df_benign['label'] = benign_label
                if 'label' not in df_attack.columns:
                    df_attack['label'] = attack_label
                else:
                    df_attack['label'] = attack_label
                if len(df_benign) > benign_limit:
                    df_benign = df_benign.sample(n=int(benign_limit), random_state=42)
                meta = generate_testing_batches(
                    df_benign,
                    df_attack,
                    output_folder=output_folder,
                    num_batches=int(num_batches),
                    benign_ratio=float(benign_ratio),
                    min_rows=int(min_rows),
                    max_rows=int(max_rows),
                    dry_run=not save_results
                )
            if save_results:
                st.success(f"Created {meta['batches_created']} batch file(s) in {output_folder}")
                with st.expander("Output files"):
                    for f in sorted(os.listdir(output_folder)):
                        if f.endswith('.csv'):
                            st.write(os.path.join(output_folder, f))
            else:
                st.info("Save disabled. Preview only — no files written.")
                avg_rows = (min_rows + max_rows) // 2
                est_total_rows = avg_rows * int(num_batches)
                st.write({
                    'batches_planned': int(num_batches),
                    'avg_rows_estimate': avg_rows,
                    'estimated_total_rows': est_total_rows,
                    'benign_ratio': float(benign_ratio),
                    'benign_rows_available': len(df_benign),
                    'attack_rows_available': len(df_attack)
                })
                st.subheader("Benign Head")
                st.dataframe(df_benign.head(), use_container_width=True)
                st.subheader("Attack Head")
                st.dataframe(df_attack.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Batch generation failed: {e}")

