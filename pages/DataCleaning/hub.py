import streamlit as st
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav

st.set_page_config(page_title="Data Cleaning", layout="wide", initial_sidebar_state="collapsed")
initialize_state()
inject_global_styles()
render_top_nav(current_page="DataCleaning/hub")

# Minimal hub page: the submenu below the top nav lists Data Cleaning tools.
st.title("Data Cleaning")
st.caption("Select a tool below.")
