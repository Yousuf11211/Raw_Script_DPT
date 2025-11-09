import os
import pandas as pd
from typing import Dict, Any, List, Optional
import polars as pl

DEFAULT_INPUT_FOLDER = "2018_Separated_Nomissing"
DEFAULT_OUTPUT_FOLDER = "Downscale_Csv_2018"


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if col.lower() == "label":
            return col
    return None


def _downscale_dataframe(df: pd.DataFrame, output_folder: str, benign_sampling_fraction: float, random_state: int, source_name: str) -> Dict[str, Any]:
    os.makedirs(output_folder, exist_ok=True)
    output_benign = os.path.join(output_folder, "benign.csv")
    output_attacks = os.path.join(output_folder, "attacks.csv")

    result: Dict[str, Any] = {
        "output_benign_path": output_benign,
        "output_attacks_path": output_attacks,
        "benign_rows": 0,
        "attacks_rows": 0,
        "per_file": []
    }

    label_col = find_label_column(df)
    if not label_col:
        return {**result, "error": "No label column found"}
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    is_benign = df["label"].astype(str).str.lower().eq("benign")
    benign_df = df.loc[is_benign]
    attack_df = df.loc[~is_benign]

    sampled_benign = benign_df.sample(frac=benign_sampling_fraction, random_state=random_state) if not benign_df.empty else benign_df
    final_benign = sampled_benign.sample(frac=1, random_state=random_state).reset_index(drop=True)
    final_attacks = attack_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if not final_benign.empty:
        final_benign.to_csv(output_benign, index=False)
        result["benign_rows"] = len(final_benign)
        result["benign_label_counts"] = final_benign['label'].value_counts().to_dict()

    if not final_attacks.empty:
        final_attacks.to_csv(output_attacks, index=False)
        result["attacks_rows"] = len(final_attacks)
        result["attacks_label_counts"] = final_attacks['label'].value_counts().to_dict()

    result["per_file"].append({
        "file": source_name,
        "type": "mixed",
        "rows_in": len(df),
        "rows_kept_benign": len(final_benign),
        "rows_kept_attacks": len(final_attacks),
    })

    return result


def downscale_from_file(
    input_file: str,
    output_folder: str,
    benign_sampling_fraction: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Any]:
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        return {"error": str(e)}
    return _downscale_dataframe(
        df=df,
        output_folder=output_folder,
        benign_sampling_fraction=benign_sampling_fraction,
        random_state=random_state,
        source_name=input_file,
    )


def downscale_from_lazyframe(
    lf: pl.LazyFrame,
    output_folder: str,
    benign_sampling_fraction: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Any]:
    df = lf.collect().to_pandas()
    return _downscale_dataframe(
        df=df,
        output_folder=output_folder,
        benign_sampling_fraction=benign_sampling_fraction,
        random_state=random_state,
        source_name="current_lazy_frame",
    )


def downscale_from_folder(
    input_folder: str,
    output_folder: str,
    benign_sampling_fraction: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Walk input_folder, read CSVs, sample benign fraction per file (if any benign present), keep all attack rows.
    Save two output CSVs in output_folder: benign.csv and attacks.csv.

    Returns dict with paths, counts and a per-file summary.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_benign = os.path.join(output_folder, "benign.csv")
    output_attacks = os.path.join(output_folder, "attacks.csv")

    benign_dfs: List[pd.DataFrame] = []
    attack_dfs: List[pd.DataFrame] = []
    per_file: List[Dict[str, Any]] = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if not file.endswith(".csv"):
                continue
            path = os.path.join(root, file)
            try:
                df = pd.read_csv(path, low_memory=False)
            except Exception as e:
                per_file.append({"file": path, "error": str(e)})
                continue
            label_col = find_label_column(df)
            if not label_col:
                per_file.append({"file": path, "skipped": "no label column"})
                continue
            if label_col != "label":
                df = df.rename(columns={label_col: "label"})

            is_benign_file = df["label"].astype(str).str.lower().eq("benign").any()
            if is_benign_file:
                sample_df = df.sample(frac=benign_sampling_fraction, random_state=random_state)
                benign_dfs.append(sample_df)
                per_file.append({
                    "file": path,
                    "type": "benign",
                    "rows_in": len(df),
                    "rows_kept": len(sample_df)
                })
            else:
                attack_dfs.append(df)
                per_file.append({
                    "file": path,
                    "type": "attack",
                    "rows_in": len(df),
                    "rows_kept": len(df)
                })

    result: Dict[str, Any] = {
        "output_benign_path": output_benign,
        "output_attacks_path": output_attacks,
        "benign_rows": 0,
        "attacks_rows": 0,
        "per_file": per_file
    }

    if benign_dfs:
        final_benign = pd.concat(benign_dfs, ignore_index=True)
        final_benign = final_benign.sample(frac=1, random_state=random_state).reset_index(drop=True)
        final_benign.to_csv(output_benign, index=False)
        result["benign_rows"] = len(final_benign)
        result["benign_label_counts"] = final_benign['label'].value_counts().to_dict()

    if attack_dfs:
        final_attacks = pd.concat(attack_dfs, ignore_index=True)
        final_attacks = final_attacks.sample(frac=1, random_state=random_state).reset_index(drop=True)
        final_attacks.to_csv(output_attacks, index=False)
        result["attacks_rows"] = len(final_attacks)
        result["attacks_label_counts"] = final_attacks['label'].value_counts().to_dict()

    return result
