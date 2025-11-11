import streamlit as st
from utils.ui_helpers import inject_global_styles, initialize_state, render_top_nav, render_category_hub

st.set_page_config(page_title="Utilities Hub", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="pages/Utilities/hub.py")

render_category_hub(
    category="Utilities",
    heading="Utility Tools",
    description="Folder comparison and frontend test batch generation utilities."
)

