import streamlit as st
import pandas as pd
import os
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from utils.ui_helpers import initialize_state, get_resource_metrics, common_header, get_lazy_data_reader
from utils.feature_importance import (
    prepare_feature_matrix,
    compute_random_forest_importance,
    compute_xgboost_importance,
    compute_xgb_per_label_importance,
    merge_importances,
    get_near_zero_features,
)
from utils import (
    refit_and_evaluate,
    build_heatmap_table,
)

st.set_page_config(page_title="Hyperparameter Tuning", layout="wide")
initialize_state()

header_train = common_header(
    "⚙️ Hyperparameter Tuning (RandomForest & XGBoost)",
    num_inputs=1,
    input_specs=[{"label": "Training CSV", "kind": "file", "allowed_exts": [".csv"]}],
    default_output_folder=""
)
train_csv_selected = header_train['input_paths'][0]

with st.sidebar:
    st.header("System Health")
    m = get_resource_metrics()
    st.metric("CPU %", f"{m['CPU %']:.1f}%")
    st.metric("RAM %", f"{m['RAM %']:.1f}%")

lf = st.session_state.get('current_lazy_frame')
file_path = st.session_state.get('current_file_path')

if train_csv_selected:
    try:
        lf = pl.scan_csv(train_csv_selected)
        file_path = train_csv_selected
        st.session_state['current_lazy_frame'] = lf
        st.session_state['current_file_path'] = file_path
    except Exception as e:
        st.error(f"Failed to lazy-load training CSV: {e}")

if lf is None:
    st.info("Select a Training CSV using the header above.")
    st.stop()

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
                    st.warning("XGBoost not installed. Run 'pip install xgboost'.")
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
                to_drop.extend([f for f in xgb_zero if f not in to_drop])
        if apply_drop and to_drop:
            if st.button(f"Drop {len(to_drop)} near-zero features from current dataset"):
                new_lf = lf.drop(to_drop)
                st.session_state['current_lazy_frame'] = new_lf
                st.session_state['applied_filters'].append(f"Drop near-zero importance ({len(to_drop)} features)")
                st.success("Scheduled lazy drop of near-zero features.")
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

st.markdown("---")
st.subheader("Refit Best Model & Evaluate")
header_eval = common_header(
    "Test Data Selection",
    num_inputs=1,
    input_specs=[{"label": "Test CSV", "kind": "file", "allowed_exts": [".csv"]}],
    default_output_folder=""
)

test_file_path = header_eval['input_paths'][0]
model_choice = st.selectbox("Model to refit", ["RandomForest","XGBoost"], index=0)
refit_button = st.button("Refit & Evaluate", use_container_width=True)

if refit_button:
    if not test_file_path or not os.path.isfile(test_file_path):
        st.error("Provide valid test CSV path.")
    else:
        try:
            test_df = pd.read_csv(test_file_path, low_memory=False)
            test_df.columns = test_df.columns.str.lower()
            if 'label' not in test_df.columns:
                st.error("Test data must have a 'label' column.")
            else:
                if 'X' not in locals() or 'y' not in locals():
                    with st.spinner("Preparing training matrix from selected Training CSV..."):
                        X, y, diag = prepare_feature_matrix(st.session_state['current_lazy_frame'], sample_frac=1.0)
                X_test = test_df.drop(columns=['label'])
                y_test = test_df['label']
                if model_choice == 'RandomForest':
                    rf_state = st.session_state.get('tuning_rf')
                    if not rf_state:
                        st.error("Run RF tuning first.")
                    else:
                        best_params = rf_state['best_params']
                        model = RandomForestClassifier(random_state=42, **best_params)
                        report = refit_and_evaluate(model, X, y, X_test, y_test)
                        st.success("Refit complete.")
                        st.text(report)
                else:
                    xgb_state = st.session_state.get('tuning_xgb')
                    if not xgb_state:
                        st.error("Run XGB tuning first.")
                    else:
                        from xgboost import XGBClassifier
                        best_params = xgb_state['best_params']
                        label_map = xgb_state['label_map']
                        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', **best_params)
                        report = refit_and_evaluate(model, X, y, X_test, y_test, label_map=label_map)
                        st.success("Refit complete.")
                        st.text(report)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

st.markdown("---")
st.subheader("Tuning Heatmap Preview")
heat_cols = st.columns(2)
with heat_cols[0]:
    show_rf_heat = st.button("Show RF Heatmap Table", use_container_width=True)
with heat_cols[1]:
    show_xgb_heat = st.button("Show XGB Heatmap Table", use_container_width=True)

if show_rf_heat and st.session_state.get('tuning_rf'):
    rf_df = st.session_state['tuning_rf']['results']
    pivot = build_heatmap_table(rf_df, 'param_max_depth', 'param_n_estimators')
    if pivot is not None:
        st.dataframe(pivot, use_container_width=True)
    else:
        st.info("RF heatmap not available (missing required columns).")

if show_xgb_heat and st.session_state.get('tuning_xgb'):
    xgb_df = st.session_state['tuning_xgb']['results']
    pivot = build_heatmap_table(xgb_df, 'param_max_depth', 'param_n_estimators')
    if pivot is not None:
        st.dataframe(pivot, use_container_width=True)
    else:
        st.info("XGB heatmap not available (missing required columns).")

st.markdown("---")
st.caption("If XGBoost missing, install with: pip install xgboost")
