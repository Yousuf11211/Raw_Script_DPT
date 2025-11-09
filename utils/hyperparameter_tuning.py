import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from sklearn.model_selection import ParameterGrid, cross_validate
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ---------------- Parameter Parsing -----------------

def parse_param_list(raw: str, cast_type) -> List[Any]:
    if not raw.strip():
        return []
    vals = []
    for part in raw.replace('\n',',').split(','):
        p = part.strip()
        if not p:
            continue
        try:
            vals.append(cast_type(p))
        except Exception:
            continue
    return vals

# ---------------- Label Mapping for XGBoost ---------

def build_label_map(y: pd.Series) -> Dict[Any, int]:
    unique = sorted(y.unique())
    return {label: idx for idx, label in enumerate(unique)}

# ---------------- XGBoost Tuning --------------------

def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Dict[str, List[Any]],
    cv: int = 3,
    scoring: str = 'f1_macro',
    n_jobs: int = 2,
    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], float, Dict[Any,int]]:
    if not HAS_XGB:
        raise ImportError("xgboost not installed. Install with 'pip install xgboost'.")
    label_map = build_label_map(y)
    y_mapped = y.map(label_map)
    combos = list(ParameterGrid(param_grid))
    rows: List[Dict[str, Any]] = []
    best_score = -1.0
    best_params: Dict[str, Any] = {}
    total = len(combos)
    for idx, params in enumerate(combos, start=1):
        model = xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=1,
            **params
        )
        cv_res = cross_validate(
            model, X, y_mapped,
            scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=0,
            return_train_score=False
        )
        mean_score = float(cv_res['test_score'].mean())
        std_score = float(cv_res['test_score'].std())
        row = { 'mean_test_score': mean_score, 'std_test_score': std_score }
        for k,v in params.items():
            row[f'param_{k}'] = v
        rows.append(row)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
        if progress_callback:
            progress_callback(idx, total, row)
    return pd.DataFrame(rows), best_params, best_score, label_map

# ---------------- RandomForest Tuning ---------------

def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Dict[str, List[Any]],
    cv: int = 3,
    scoring: str = 'f1_macro',
    n_jobs: int = -1,
    progress_callback: Optional[Callable[[int,int,Dict[str,Any]], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], float]:
    combos = list(ParameterGrid(param_grid))
    rows: List[Dict[str, Any]] = []
    best_score = -1.0
    best_params: Dict[str, Any] = {}
    total = len(combos)
    for idx, params in enumerate(combos, start=1):
        model = RandomForestClassifier(random_state=42, **params)
        cv_res = cross_validate(
            model, X, y,
            scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=0,
            return_train_score=False
        )
        mean_score = float(cv_res['test_score'].mean())
        std_score = float(cv_res['test_score'].std())
        row = { 'mean_test_score': mean_score, 'std_test_score': std_score }
        for k,v in params.items():
            row[f'param_{k}'] = v
        rows.append(row)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
        if progress_callback:
            progress_callback(idx, total, row)
    return pd.DataFrame(rows), best_params, best_score

# ---------------- Evaluation & Heatmap --------------

def refit_and_evaluate(model, X_train, y_train, X_test, y_test, label_map: Optional[Dict[Any,int]] = None) -> str:
    if label_map:
        y_train_mapped = y_train.map(label_map)
        y_test_mapped = y_test.map(label_map)
    else:
        y_train_mapped = y_train
        y_test_mapped = y_test
    model.fit(X_train, y_train_mapped)
    preds = model.predict(X_test)
    return classification_report(y_test_mapped, preds)


def build_heatmap_table(results_df: pd.DataFrame, depth_key: str, estimators_key: str) -> Optional[pd.DataFrame]:
    if depth_key not in results_df.columns or estimators_key not in results_df.columns:
        return None
    pivot = results_df.pivot_table(index=depth_key, columns=estimators_key, values='mean_test_score')
    return pivot

