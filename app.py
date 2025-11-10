import streamlit as st
from utils.ui_helpers import inject_global_styles, render_global_nav, initialize_state

st.set_page_config(page_title="Thesis Data Tool", layout="wide", initial_sidebar_state="expanded", page_icon="🚀")
initialize_state()

# Apply global styles & show nav even during redirect fallback
inject_global_styles()
render_global_nav(active_page_hint="Landing")

# Redirect to the Landing page under pages/
try:
    st.switch_page("pages/0_Landing.py")
except Exception:
    st.title("Thesis Data Tool")
    st.info("Use the hierarchical sidebar to navigate or click below.")
    try:
        st.page_link("pages/0_Landing.py", label="Go to Landing Page", icon="🚀")
    except Exception:
        st.write("Open the ‘0_Landing’ page from the sidebar.")
