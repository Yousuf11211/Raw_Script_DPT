import streamlit as st
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav

st.set_page_config(page_title="Model Testing", layout="wide", initial_sidebar_state="collapsed")
initialize_state()
inject_global_styles()
render_top_nav(current_page="ModelTesting/hub")

st.title("Model Testing")
st.caption("Select a tool below.")
