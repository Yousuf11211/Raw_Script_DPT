import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, List


def calculate_target_strategy(y: List[int], ratio: float) -> Dict[int, int]:
    """Calculate target class counts based on desired minority/majority ratio."""
    counts = Counter(y)
    if not counts:
        return {}
    majority_class_key = max(counts, key=counts.get)
    majority_count = counts[majority_class_key]
    target_strategy: Dict[int, int] = {}
    target_minority_count = int(majority_count * ratio)
    for cls, count in counts.items():
        if cls == majority_class_key:
            target_strategy[cls] = count
        else:
            target_strategy[cls] = max(count, target_minority_count)
    return target_strategy


def apply_resampling(X: pd.DataFrame, y: List[int], target_strategy: Dict[int, int], oversampler_name: str) -> Tuple[pd.DataFrame, List[int]]:
    """Apply undersampling then oversampling to reach target counts for each class.
    oversampler_name: one of 'SMOTE', 'BorderlineSMOTE', 'ADASYN'"""
    from imblearn.under_sampling import RandomUnderSampler  # local import to avoid hard dependency at import time
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

    current_counts = Counter(y)
    undersample = {c: t for c, t in target_strategy.items() if c in current_counts and current_counts[c] > t}
    oversample = {c: t for c, t in target_strategy.items() if c in current_counts and current_counts[c] < t}

    X_res, y_res = X.copy(), list(y)

    if undersample:
        rus = RandomUnderSampler(sampling_strategy=undersample, random_state=42)
        X_res, y_res = rus.fit_resample(X_res, y_res)

    if oversample:
        min_samples_for_smote = min(count for cls, count in Counter(y_res).items() if cls in oversample)
        num_neighbors = max(1, min(min_samples_for_smote - 1, 5))
        sampler_params = {
            'sampling_strategy': oversample,
            'random_state': 42
        }
        if oversampler_name == 'ADASYN':
            sampler_params['n_neighbors'] = num_neighbors
            sampler = ADASYN(**sampler_params)
        elif oversampler_name == 'BorderlineSMOTE':
            sampler_params['k_neighbors'] = num_neighbors
            sampler = BorderlineSMOTE(**sampler_params)
        else:  # default SMOTE
            sampler_params['k_neighbors'] = num_neighbors
            sampler = SMOTE(**sampler_params)
        X_res, y_res = sampler.fit_resample(X_res, y_res)

    return X_res, y_res


def balance_dataframe(df: pd.DataFrame, label_col: str, ratio: float, oversampler_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Balance a pandas DataFrame w.r.t. the label column using undersampling + oversampling.
    Returns: (balanced_df, label_distribution_df)
    oversampler_name: 'SMOTE', 'BorderlineSMOTE', or 'ADASYN'
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(df[label_col])
    X = df.drop(columns=[label_col])

    strategy = calculate_target_strategy(y_enc, ratio)
    X_bal, y_bal = apply_resampling(X, list(y_enc), strategy, oversampler_name)

    df_bal = pd.DataFrame(X_bal, columns=X.columns)
    df_bal[label_col] = le.inverse_transform(y_bal)

    dist = pd.DataFrame(Counter(y_bal).items(), columns=['EncodedLabel', 'Count'])
    dist['Label'] = dist['EncodedLabel'].map({i: lbl for i, lbl in enumerate(le.classes_)})
    dist = dist[['Label', 'Count']].sort_values('Count', ascending=False)

    return df_bal, dist


def label_distribution(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    counts = Counter(df[label_col])
    dist = pd.DataFrame(list(counts.items()), columns=['Label', 'Count']).sort_values('Count', ascending=False)
    dist['Percentage'] = (dist['Count'] / dist['Count'].sum()) * 100.0
    return dist
