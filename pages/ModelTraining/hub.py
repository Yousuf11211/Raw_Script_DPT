import streamlit as st
from utils.ui_helpers import inject_global_styles, initialize_state, render_top_nav, render_category_hub

st.set_page_config(page_title="Model Training Hub", layout="wide")
initialize_state()
inject_global_styles()
render_top_nav(current_page="pages/ModelTraining/hub.py")

render_category_hub(
    category="Model Training",
    heading="Model Training Tools",
    description="Hyperparameter tuning, isolation forest training, and attack model training & testing."
)

