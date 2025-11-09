import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def sample_numeric_from_csv(file_path: str, chunk_size: int, sample_fraction: float) -> pd.DataFrame:
    """Stream a large CSV in chunks and sample fraction from each, returning numeric-only combined frame."""
    frames: List[pd.DataFrame] = []
    with pd.read_csv(file_path, chunksize=chunk_size, low_memory=False) as reader:
        for chunk in reader:
            if sample_fraction < 1.0:
                chunk = chunk.sample(frac=sample_fraction)
            frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df


def train_isolation_forest_on_csv(file_path: str,
                                  chunk_size: int = 2_000_000,
                                  sample_fraction: float = 0.1,
                                  n_estimators: int = 100,
                                  contamination: str | float = 'auto',
                                  max_samples: str | int | float = 'auto',
                                  max_features: int | float = 1.0,
                                  random_state: int = 42,
                                  n_jobs: int = -1) -> Tuple[IsolationForest, Dict[str, Any]]:
    X = sample_numeric_from_csv(file_path, chunk_size, sample_fraction)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X)
    stats = {
        'rows_used': len(X),
        'features': list(X.columns),
        'n_features': X.shape[1]
    }
    return model, stats


def train_xgb_on_csv(file_path: str,
                     params: Dict[str, Any],
                     train_full: bool = True,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Tuple[Any, Dict[str, int], Optional[Dict[str, Any]]]:
    """Train an XGBoost classifier on a single CSV, label-encoding 'label'.
    Returns (model, label_map, eval_dict or None)."""
    if not HAS_XGB:
        raise ImportError("xgboost not installed. Install with 'pip install xgboost'.")
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.lower()
    if 'label' not in df.columns:
        raise ValueError("No 'label' column found.")
    X = df.drop(columns=['label'])
    y_raw = df['label']
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    # encode object columns in X
    obj_cols = X.select_dtypes(include='object').columns
    for c in obj_cols:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    model = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, **params)
    eval_out: Optional[Dict[str, Any]] = None
    if train_full:
        model.fit(X, y)
    else:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        cm = confusion_matrix(y_test, y_pred)
        eval_out = {
            'report': report,
            'confusion_matrix': pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        }
    label_map = {cls: idx for idx, cls in enumerate(le.classes_)}
    return model, label_map, eval_out


def test_sklearn_model_on_csv(model_path: str,
                               test_csv_path: str,
                               label_mapping_path: Optional[str] = None) -> Dict[str, Any]:
    """Load a joblib sklearn model and test it on a CSV. If label_mapping provided, compute reports.
    Returns dict with counts, predictions, optional report and confusion matrix."""
    import joblib
    from sklearn.metrics import classification_report, confusion_matrix
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv_path, low_memory=False)
    df.columns = df.columns.str.lower()
    if 'label' in df.columns:
        y_raw = df['label']
        X = df.drop(columns=['label'])
    else:
        y_raw = None
        X = df
    # Align features
    if hasattr(model, 'feature_names_in_'):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    y_pred = model.predict(X)
    result: Dict[str, Any] = {}
    if label_mapping_path and y_raw is not None:
        # mapping file has lines "class: idx"; build maps
        mapping: Dict[int,str] = {}
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[2:]
            for line in lines:
                if ':' not in line:
                    continue
                cls, num = line.strip().split(':')
                cls, num = cls.strip(), num.strip()
                if num.isdigit():
                    mapping[int(num)] = cls
        inv = {v:k for k,v in mapping.items()}
        y_true = np.array([inv.get(lbl, -1) for lbl in y_raw])
        report = classification_report(y_true, y_pred, target_names=[mapping[i] for i in sorted(mapping.keys())], zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        result['report'] = report
        result['confusion_matrix'] = pd.DataFrame(cm, index=[mapping[i] for i in sorted(mapping.keys())], columns=[mapping[i] for i in sorted(mapping.keys())])
    # predicted labels mapping if available
    if label_mapping_path:
        # If we have mapping, convert indices to names
        mapping2: Dict[int,str] = {}
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[2:]
            for line in lines:
                if ':' not in line:
                    continue
                cls, num = line.strip().split(':')
                cls, num = cls.strip(), num.strip()
                if num.isdigit():
                    mapping2[int(num)] = cls
        pred_labels = [mapping2.get(int(idx), str(idx)) for idx in y_pred]
    else:
        pred_labels = [str(idx) for idx in y_pred]
    from collections import Counter
    counts = Counter(pred_labels)
    result['prediction_counts'] = dict(counts)
    result['predicted_labels'] = pred_labels
    return result

