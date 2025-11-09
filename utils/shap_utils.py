import os
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from sklearn.preprocessing import LabelEncoder
import joblib


def _load_label_mapping_txt(path: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if not path or not os.path.isfile(path):
        return mapping
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[2:]
        for line in lines:
            if ':' not in line:
                continue
            cls, num = line.strip().split(':')
            cls, num = cls.strip(), num.strip()
            if num.isdigit():
                mapping[int(num)] = cls
    return mapping


def shap_explain_tree_model(
    model_path: str,
    test_csv_path: str,
    label_mapping_path: Optional[str] = None,
    sample_rows: Optional[int] = None,
    top_k: int = 3,
    max_per_class: int = 1,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Compute per-class explanation strings and return a dataframe of per-row top contributions.
    Returns (per_class_explanations, per_row_summary_df)
    """
    if not HAS_SHAP:
        raise ImportError("shap not installed. Install with 'pip install shap'.")

    model = joblib.load(model_path)
    df = pd.read_csv(test_csv_path, low_memory=False)
    cols = [c for c in df.columns if c != 'label']
    X = df[cols]
    # encode object columns
    for c in X.select_dtypes(include='object').columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    if sample_rows is not None and sample_rows > 0 and len(X) > sample_rows:
        X = X.sample(n=int(sample_rows), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    feature_names = list(X.columns)

    # Prepare mapping
    idx_to_label = _load_label_mapping_txt(label_mapping_path) if label_mapping_path else {}

    explanations: Dict[str, str] = {}
    per_row_records: List[Dict[str, Any]] = []

    # Model.predict to get predicted class index per row
    y_pred = model.predict(X)

    for i, pred_idx in enumerate(y_pred):
        # shap_values can be list per class for multi-class
        row_sv = shap_values[pred_idx][i] if isinstance(shap_values, list) else shap_values[i]
        contrib = sorted(zip(feature_names, row_sv), key=lambda x: abs(x[1]), reverse=True)
        top = contrib[:top_k]
        pred_label = idx_to_label.get(int(pred_idx), str(pred_idx))
        if pred_label not in explanations and len(top) > 0:
            explanation = f"{pred_label} predicted because: " + \
                ", ".join([f"{f} was {'high' if v > 0 else 'low'}" for f, v in top])
            explanations[pred_label] = explanation
        per_row_records.append({
            'row_index': int(i),
            'pred_class': int(pred_idx),
            'pred_label': pred_label,
            'top_features': "; ".join([f for f,_ in top]),
            'top_values': "; ".join([str(v) for _,v in top])
        })

    # Keep at most max_per_class in explanations
    if max_per_class == 1:
        # already first occurrence only
        pass
    else:
        # Extension: could aggregate more per class if needed
        pass

    return explanations, pd.DataFrame(per_row_records)

