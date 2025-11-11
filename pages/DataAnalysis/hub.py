import streamlit as st
from utils.ui_helpers import inject_global_styles, initialize_state, render_top_nav, render_category_hub

st.set_page_config(page_title="Data Analysis Hub", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="pages/DataAnalysis/hub.py")

render_category_hub(
    category="Data Analysis",
    heading="Data Analysis Tools",
    description="Run feature importance and related analyses on your dataset."
)

