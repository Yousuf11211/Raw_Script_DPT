import streamlit as st
import os
from utils.ui_helpers import initialize_state, get_resource_metrics
from utils import load_benign_attack, generate_testing_batches

st.set_page_config(page_title="Frontend Test Batch Generator", layout="wide")
initialize_state()

st.title("🧪 Frontend Test Batch Generator")
st.caption("Generate mixed benign/attack testing CSV batches with unique timestamps and src_ip values.")

with st.sidebar:
    st.header("System Health")
    m = get_resource_metrics()
    st.metric("CPU %", f"{m['CPU %']:.1f}%")
    st.metric("RAM %", f"{m['RAM %']:.1f}%")

col1, col2, col3 = st.columns(3)
with col1:
    benign_folder = st.text_input("Benign folder", value="Testing_isolation_model_cleaned")
with col2:
    attack_folder = st.text_input("Attack folder", value="Testing_Attack")
with col3:
    output_folder = st.text_input("Output folder", value="frontend_testing")

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

if st.button("Generate Batches", use_container_width=True):
    if not os.path.isdir(benign_folder) or not os.path.isdir(attack_folder):
        st.error("Benign or Attack folder not found.")
    else:
        try:
            low_high = [int(x.strip()) for x in rows_min_max.split(',') if x.strip()]
            if len(low_high) != 2:
                raise ValueError("Rows range must be two integers, e.g. 50,100")
            min_rows, max_rows = low_high
            with st.spinner("Loading data and generating batches..."):
                df_benign, df_attack = load_benign_attack(benign_folder, attack_folder, int(benign_limit), benign_label, attack_label)
                meta = generate_testing_batches(
                    df_benign, df_attack, output_folder=output_folder,
                    num_batches=int(num_batches), benign_ratio=float(benign_ratio),
                    min_rows=int(min_rows), max_rows=int(max_rows)
                )
            st.success(f"Created {meta['batches_created']} batch file(s) in {output_folder}")
            with st.expander("Output files"):
                for f in sorted(os.listdir(output_folder)):
                    if f.endswith('.csv'):
                        st.write(os.path.join(output_folder, f))
        except Exception as e:
            st.error(f"Batch generation failed: {e}")

