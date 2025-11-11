import streamlit as st
from utils.ui_helpers import inject_global_styles, initialize_state, render_top_nav, render_category_hub

st.set_page_config(page_title="Data Cleaning Hub", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="pages/DataCleaning/hub.py")

render_category_hub(
    category="Data Cleaning",
    heading="Data Cleaning Tools",
    description="Validate, deduplicate, handle INF & mixed types, encoding candidates, low variance, dominance reports, and column deletion."
)

