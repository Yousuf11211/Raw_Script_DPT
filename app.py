import streamlit as st
from utils.ui_helpers import inject_global_styles, render_global_nav, initialize_state

# Configure root launcher page (acts as redirect to landing); remove emoji icon per requirements
st.set_page_config(page_title="Thesis Data Tool", layout="wide", initial_sidebar_state="collapsed")
initialize_state()

# Apply global styles (hides default sidebar page list) then redirect to landing
inject_global_styles()

try:
    st.switch_page("pages/0_Landing.py")
except Exception:
    # Fallback minimal content if switch_page not available
    st.title("Thesis Data Tool")
    st.info("Unable to auto-redirect. Open the ‘0_Landing’ page from the page selector if visible.")
