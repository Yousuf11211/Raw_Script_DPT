import streamlit as st
from utils.landing import render_landing_page
from utils.ui_helpers import initialize_state, get_resource_metrics

st.set_page_config(page_title="Thesis Data Tool", layout="wide", initial_sidebar_state="expanded", page_icon="🚀")
initialize_state()

render_landing_page()

with st.sidebar:
    st.header("System Health")
    metrics = get_resource_metrics()
    col_cpu, col_ram = st.columns(2)
    with col_cpu:
        st.metric("CPU Usage", f"{metrics['CPU %']:.1f}%")
    with col_ram:
        st.metric("RAM Used", f"{metrics['RAM Used (GB)']:.2f} GB")
        st.metric("RAM %", f"{metrics['RAM %']:.1f}%")
    st.divider()
    st.caption("Navigate using the sidebar pages. Each page has its own file selection header.")

