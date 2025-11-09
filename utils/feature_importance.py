import polars as pl
import pandas as pd
from typing import Tuple, Dict, List, Optional

# ----------------------------- Dataset Prep -----------------------------

def prepare_feature_matrix(lf: pl.LazyFrame, label_col: str = 'label', sample_frac: float = 1.0, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, Dict[str, any]]:
    """
    Collect lazyframe to pandas, optional row sampling (fraction 0<..<=1), separate features and label, encode object columns.
    Returns (X_df, y_series, diagnostics). Raises ValueError if label column missing.
    diagnostics: {'rows': int, 'cols': int, 'sampled_rows': int, 'object_cols': int}
    """
    df_full = lf.collect().to_pandas()
    if sample_frac < 1.0 and sample_frac > 0:
        df = df_full.sample(frac=sample_frac, random_state=random_state)
    else:
        df = df_full
    cols_lower = {c.lower(): c for c in df.columns}
    if label_col not in cols_lower and label_col.lower() not in cols_lower:
        raise ValueError(f"Label column '{label_col}' not present.")
    actual_label = cols_lower.get(label_col, cols_lower.get(label_col.lower()))
    df.rename(columns={actual_label: 'label'}, inplace=True)
    y = df['label']
    X = df.drop(columns=['label'])
    obj_cols = X.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        from sklearn.preprocessing import LabelEncoder
        for c in obj_cols:
            try:
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
            except Exception:
                # fallback: hash encoding
                X[c] = X[c].astype(str).apply(lambda v: hash(v) % 10_000_000)
    diagnostics = {
        'rows': len(df_full),
        'cols': len(df_full.columns),
        'sampled_rows': len(df),
        'object_cols': len(obj_cols)
    }
    return X, y, diagnostics

# ----------------------------- RandomForest Importance -----------------------------

def compute_random_forest_importance(X: pd.DataFrame, y: pd.Series, n_estimators: int = 100, random_state: int = 42) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_ * 100.0
    df_imp = pd.DataFrame({'feature': X.columns, 'rf_importance_pct': importances}).sort_values('rf_importance_pct', ascending=False).reset_index(drop=True)
    return df_imp

# ----------------------------- XGBoost Importance -----------------------------

def compute_xgboost_importance(X: pd.DataFrame, y: pd.Series, n_estimators: int = 100, random_state: int = 42) -> Optional[pd.DataFrame]:
    try:
        import xgboost as xgb
    except ImportError:
        return None
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X, y)
    importances = model.feature_importances_
    pct = (importances / importances.sum()) * 100.0 if importances.sum() > 0 else importances
    df_imp = pd.DataFrame({'feature': X.columns, 'xgb_importance_pct': pct}).sort_values('xgb_importance_pct', ascending=False).reset_index(drop=True)
    return df_imp

# ----------------------------- XGBoost Per-Label (One-vs-Rest) -----------------------------

def compute_xgb_per_label_importance(X: pd.DataFrame, y: pd.Series) -> Dict[str, pd.DataFrame]:
    try:
        import xgboost as xgb
    except ImportError:
        return {}
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    unique_vals = sorted(set(y_enc))
    results: Dict[str, pd.DataFrame] = {}
    for label_val in unique_vals:
        y_binary = np.where(y_enc == label_val, 1, 0)
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y_binary)
        importances = model.feature_importances_
        df_imp = pd.DataFrame({'feature': X.columns,'importance': importances}).sort_values('importance', ascending=False).reset_index(drop=True)
        results[le.classes_[label_val]] = df_imp
    return results

# ----------------------------- Comparison Merge -----------------------------

def merge_importances(rf_df: Optional[pd.DataFrame], xgb_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if rf_df is None and xgb_df is None:
        return pd.DataFrame(columns=['feature'])
    if rf_df is None:
        return xgb_df.copy()
    if xgb_df is None:
        return rf_df.copy()
    merged = pd.merge(rf_df, xgb_df, on='feature', how='outer')
    # Rank calculations
    if 'rf_importance_pct' in merged.columns:
        merged['rf_rank'] = merged['rf_importance_pct'].rank(ascending=False, method='dense')
    if 'xgb_importance_pct' in merged.columns:
        merged['xgb_rank'] = merged['xgb_importance_pct'].rank(ascending=False, method='dense')
    if 'rf_rank' in merged.columns and 'xgb_rank' in merged.columns:
        merged['rank_diff'] = merged['rf_rank'] - merged['xgb_rank']
    return merged.sort_values(by=[col for col in ['rf_importance_pct','xgb_importance_pct'] if col in merged.columns], ascending=False)

# ----------------------------- Near-Zero Filtering -----------------------------

def get_near_zero_features(df_imp: pd.DataFrame, threshold: float, col_name: str) -> List[str]:
    if col_name not in df_imp.columns:
        return []
    return df_imp.loc[df_imp[col_name] < threshold, 'feature'].tolist()
