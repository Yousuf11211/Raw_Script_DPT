import streamlit as st

st.set_page_config(page_title="Thesis Data Tool", layout="wide", initial_sidebar_state="expanded", page_icon="🚀")

# Redirect to the Landing page under pages/ if supported
try:
    st.switch_page("pages/0_Landing.py")
except Exception:
    st.title("Thesis Data Tool")
    st.info("Use the sidebar to open ‘0_Landing’ or click below.")
    try:
        st.page_link("pages/0_Landing.py", label="Go to Landing Page", icon="🚀")
    except Exception:
        st.write("Open the ‘0_Landing’ page from the sidebar.")
