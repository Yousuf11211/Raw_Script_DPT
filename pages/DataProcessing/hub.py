import streamlit as st
from utils.ui_helpers import inject_global_styles, initialize_state, render_top_nav, render_category_hub

st.set_page_config(page_title="Data Processing Hub", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="pages/DataProcessing/hub.py")

render_category_hub(
    category="Data Processing",
    heading="Data Processing Tools",
    description="Balance classes, downscale datasets, separate benign/attack sets, merge & shuffle large corpora, and manage outliers."
)

