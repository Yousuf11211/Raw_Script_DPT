import streamlit as st
from utils.ui_helpers import initialize_state, get_resource_metrics
from utils.feature_importance import (
    prepare_feature_matrix,
    compute_random_forest_importance,
    compute_xgboost_importance,
    compute_xgb_per_label_importance,
    merge_importances,
    get_near_zero_features,
)

# Page setup
st.set_page_config(page_title="Feature Importance", layout="wide")
initialize_state()

st.title("📊 Feature Importance (RandomForest vs XGBoost)")

with st.sidebar:
    st.header("System Health")
    m = get_resource_metrics()
    st.metric("CPU %", f"{m['CPU %']:.1f}%")
    st.metric("RAM %", f"{m['RAM %']:.1f}%")

# Current dataset from session
lf = st.session_state.get('current_lazy_frame')
file_path = st.session_state.get('current_file_path')
if lf is None:
    st.info("Load a dataset on the Home page first.")
    st.stop()

# Controls
st.subheader("Run Feature Importance Analysis")
method_choice = st.multiselect(
    "Choose methods",
    ["RandomForest", "XGBoost", "Compare Both"],
    default=["RandomForest", "XGBoost"]
)
per_label = st.checkbox("Include per-label (One-vs-Rest) XGBoost analysis", value=False)
rf_estimators = st.number_input("RandomForest n_estimators", min_value=50, max_value=1000, value=100, step=10)
xgb_estimators = st.number_input("XGBoost n_estimators", min_value=50, max_value=1000, value=100, step=10)
near_zero_threshold = st.number_input("Near-zero importance threshold (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05)

# Sampling and optional drop
sample_frac = st.slider("Row sampling fraction (for speed)", 0.05, 1.0, 1.0, 0.05)
apply_drop = st.checkbox("Allow dropping near-zero features from current dataset", value=False)

if st.button("Run Importance", use_container_width=True):
    try:
        with st.spinner("Preparing data matrix..."):
            X, y, diag = prepare_feature_matrix(lf, sample_frac=float(sample_frac))
        st.caption(f"Rows: {diag['rows']} | Sampled: {diag['sampled_rows']} | Features: {diag['cols']-1} | Object encoded: {diag['object_cols']}")
        rf_df = xgb_df = None
        if "RandomForest" in method_choice or "Compare Both" in method_choice:
            with st.spinner("Training RandomForest..."):
                rf_df = compute_random_forest_importance(X, y, n_estimators=int(rf_estimators))
                st.success("RandomForest importance computed.")
        if "XGBoost" in method_choice or "Compare Both" in method_choice:
            with st.spinner("Training XGBoost..."):
                xgb_df = compute_xgboost_importance(X, y, n_estimators=int(xgb_estimators))
                if xgb_df is None:
                    st.warning("XGBoost not installed. Run 'pip install xgboost' to enable XGBoost analysis.")
                else:
                    st.success("XGBoost importance computed.")
        if "Compare Both" in method_choice:
            merged = merge_importances(rf_df, xgb_df)
            st.subheader("Merged Importance Results")
            st.dataframe(merged.head(100), use_container_width=True)
            st.download_button("Download merged CSV", merged.to_csv(index=False), file_name="merged_feature_importance.csv")
        else:
            if rf_df is not None:
                st.subheader("RandomForest Importance")
                st.dataframe(rf_df.head(50), use_container_width=True)
                st.download_button("Download RF importance CSV", rf_df.to_csv(index=False), file_name="rf_feature_importance.csv")
            if xgb_df is not None:
                st.subheader("XGBoost Importance")
                st.dataframe(xgb_df.head(50), use_container_width=True)
                st.download_button("Download XGB importance CSV", xgb_df.to_csv(index=False), file_name="xgb_feature_importance.csv")
        # Near-zero features (report and optionally drop)
        to_drop = []
        if rf_df is not None:
            rf_zero = get_near_zero_features(rf_df, near_zero_threshold, 'rf_importance_pct')
            if rf_zero:
                st.warning(f"RandomForest near-zero features (< {near_zero_threshold}%): {len(rf_zero)}")
                with st.expander("RF near-zero list"):
                    st.write(rf_zero)
                to_drop.extend(rf_zero)
        if xgb_df is not None:
            xgb_zero = get_near_zero_features(xgb_df, near_zero_threshold, 'xgb_importance_pct')
            if xgb_zero:
                st.warning(f"XGBoost near-zero features (< {near_zero_threshold}%): {len(xgb_zero)}")
                with st.expander("XGB near-zero list"):
                    st.write(xgb_zero)
                # union with existing
                to_drop.extend([f for f in xgb_zero if f not in to_drop])
        if apply_drop and to_drop:
            if st.button(f"Drop {len(to_drop)} near-zero features from current dataset"):
                new_lf = lf.drop(to_drop)
                st.session_state['current_lazy_frame'] = new_lf
                st.session_state['applied_filters'].append(f"Drop near-zero importance ({len(to_drop)} features)")
                st.success("Scheduled lazy drop of near-zero features.")
        # Per-label analysis
        if per_label and xgb_df is not None:
            with st.spinner("Running per-label XGBoost One-vs-Rest analysis..."):
                per_label_maps = compute_xgb_per_label_importance(X, y)
            if per_label_maps:
                st.subheader("Per-Label XGBoost Importance (Top 15 each)")
                for label_name, df_imp in per_label_maps.items():
                    st.markdown(f"**{label_name}**")
                    st.dataframe(df_imp.head(15), use_container_width=True)
            else:
                st.info("Per-label analysis skipped (XGBoost unavailable).")
    except Exception as e:
        st.error(f"Failed to compute importance: {e}")
