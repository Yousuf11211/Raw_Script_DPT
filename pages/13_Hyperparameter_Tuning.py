import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from utils.ui_helpers import initialize_state, get_resource_metrics
from utils.feature_importance import prepare_feature_matrix
from utils import (
    parse_param_list,
    tune_xgboost,
    tune_random_forest,
    refit_and_evaluate,
    build_heatmap_table,
)

st.set_page_config(page_title="Hyperparameter Tuning", layout="wide")
initialize_state()

st.title("⚙️ Hyperparameter Tuning (RandomForest & XGBoost)")

with st.sidebar:
    st.header("System Health")
    m = get_resource_metrics()
    st.metric("CPU %", f"{m['CPU %']:.1f}%")
    st.metric("RAM %", f"{m['RAM %']:.1f}%")

lf = st.session_state.get('current_lazy_frame')
file_path = st.session_state.get('current_file_path')
if lf is None:
    st.info("Load a dataset with a 'label' column on Home first.")
    st.stop()

st.subheader("Data Sampling & CV Settings")
colA, colB, colC = st.columns(3)
with colA:
    sample_frac = st.slider("Row sampling fraction", 0.05, 1.0, 1.0, 0.05)
with colB:
    cv_folds = st.selectbox("CV folds", [3,5,7], index=0)
with colC:
    scoring = st.selectbox("Scoring metric", ["f1_macro","accuracy","precision_macro","recall_macro"], index=0)

st.markdown("---")
st.subheader("RandomForest Parameter Grid")
rf_cols = st.columns(4)
with rf_cols[0]:
    rf_n_estimators_raw = st.text_area("n_estimators", "100,200,300", key="rf_n_estimators")
with rf_cols[1]:
    rf_max_depth_raw = st.text_area("max_depth", "10,20,30", key="rf_max_depth")
with rf_cols[2]:
    rf_min_samples_split_raw = st.text_area("min_samples_split", "2,5", key="rf_min_samples_split")
with rf_cols[3]:
    rf_max_features_raw = st.text_area("max_features", "sqrt,log2", key="rf_max_features")

st.markdown("---")
st.subheader("XGBoost Parameter Grid")
xgb_cols = st.columns(4)
with xgb_cols[0]:
    xgb_n_estimators_raw = st.text_area("n_estimators", "100,200,300", key="xgb_n_estimators")
with xgb_cols[1]:
    xgb_max_depth_raw = st.text_area("max_depth", "5,10,20", key="xgb_max_depth")
with xgb_cols[2]:
    xgb_learning_rate_raw = st.text_area("learning_rate", "0.01,0.1", key="xgb_learning_rate")
with xgb_cols[3]:
    xgb_subsample_raw = st.text_area("subsample", "0.8,1.0", key="xgb_subsample")

run_cols = st.columns(3)
with run_cols[0]:
    run_rf = st.button("Run RandomForest Tuning", use_container_width=True)
with run_cols[1]:
    run_xgb = st.button("Run XGBoost Tuning", use_container_width=True)
with run_cols[2]:
    run_both = st.button("Run Both", use_container_width=True)

results_rf = st.session_state.get('tuning_rf')
results_xgb = st.session_state.get('tuning_xgb')

if run_rf or run_both or run_xgb:
    with st.spinner("Preparing data matrix..."):
        X, y, diag = prepare_feature_matrix(lf, sample_frac=float(sample_frac))
    st.caption(f"Rows: {diag['rows']} | Sampled: {diag['sampled_rows']} | Features: {diag['cols']-1} | Object encoded: {diag['object_cols']}")

if run_rf or run_both:
    rf_grid = {
        'n_estimators': parse_param_list(rf_n_estimators_raw, int),
        'max_depth': parse_param_list(rf_max_depth_raw, int),
        'min_samples_split': parse_param_list(rf_min_samples_split_raw, int),
        'max_features': parse_param_list(rf_max_features_raw, str),
    }
    if any(len(v)==0 for v in rf_grid.values()):
        st.error("RandomForest grid has an empty dimension.")
    else:
        prog = st.progress(0)
        status = st.empty()
        def rf_cb(i,total,row):
            prog.progress(i/total)
            status.text(f"RF Progress: {i}/{total} | last score {row['mean_test_score']:.4f}")
        try:
            rf_df, rf_best_params, rf_best_score = tune_random_forest(
                X, y, rf_grid, cv=cv_folds, scoring=scoring, progress_callback=rf_cb
            )
            st.session_state['tuning_rf'] = {
                'results': rf_df,
                'best_params': rf_best_params,
                'best_score': rf_best_score,
            }
            st.success(f"RF tuning done. Best {scoring}: {rf_best_score:.4f}")
            st.dataframe(rf_df.sort_values('mean_test_score', ascending=False).head(50), use_container_width=True)
            st.download_button("Download RF results CSV", rf_df.to_csv(index=False), file_name="rf_tuning_results.csv")
        except Exception as e:
            st.error(f"RF tuning failed: {e}")

if run_xgb or run_both:
    if not run_rf and 'X' not in locals():
        with st.spinner("Preparing data matrix..."):
            X, y, diag = prepare_feature_matrix(lf, sample_frac=float(sample_frac))
        st.caption(f"Rows: {diag['rows']} | Sampled: {diag['sampled_rows']} | Features: {diag['cols']-1} | Object encoded: {diag['object_cols']}")
    xgb_grid = {
        'n_estimators': parse_param_list(xgb_n_estimators_raw, int),
        'max_depth': parse_param_list(xgb_max_depth_raw, int),
        'learning_rate': parse_param_list(xgb_learning_rate_raw, float),
        'subsample': parse_param_list(xgb_subsample_raw, float),
    }
    if any(len(v)==0 for v in xgb_grid.values()):
        st.error("XGBoost grid has an empty dimension.")
    else:
        if not st.session_state.get('HAS_XGB_LIB', False):
            try:
                import xgboost  # noqa: F401
                st.session_state['HAS_XGB_LIB'] = True
            except Exception:
                st.warning("XGBoost library not installed. Run: pip install xgboost")
        if st.session_state.get('HAS_XGB_LIB'):
            prog_x = st.progress(0)
            status_x = st.empty()
            def xgb_cb(i,total,row):
                prog_x.progress(i/total)
                status_x.text(f"XGB Progress: {i}/{total} | last score {row['mean_test_score']:.4f}")
            try:
                xgb_df, xgb_best_params, xgb_best_score, label_map = tune_xgboost(
                    X, y, xgb_grid, cv=cv_folds, scoring=scoring, progress_callback=xgb_cb
                )
                st.session_state['tuning_xgb'] = {
                    'results': xgb_df,
                    'best_params': xgb_best_params,
                    'best_score': xgb_best_score,
                    'label_map': label_map,
                }
                st.success(f"XGB tuning done. Best {scoring}: {xgb_best_score:.4f}")
                st.dataframe(xgb_df.sort_values('mean_test_score', ascending=False).head(50), use_container_width=True)
                st.download_button("Download XGB results CSV", xgb_df.to_csv(index=False), file_name="xgb_tuning_results.csv")
            except Exception as e:
                st.error(f"XGB tuning failed: {e}")

# Refit & Evaluate section
st.markdown("---")
st.subheader("Refit Best Model & Evaluate")
colEval = st.columns(3)
with colEval[0]:
    test_file_path = st.text_input("Test CSV path", value="")
with colEval[1]:
    model_choice = st.selectbox("Model to refit", ["RandomForest","XGBoost"], index=0)
with colEval[2]:
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

# Heatmap preview (if available)
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
