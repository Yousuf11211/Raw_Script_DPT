import streamlit as st
from utils.landing import render_landing_page
from utils.ui_helpers import initialize_state, inject_global_styles, render_global_nav

st.set_page_config(page_title="Thesis Data Tool", layout="wide", initial_sidebar_state="expanded", page_icon="🚀")
initialize_state()

# Inject styles + navigation (landing sets active hint broadly)
inject_global_styles()
render_global_nav(active_page_hint="Landing")

render_landing_page()

# Footer note (navigation already shows metrics & developer)
