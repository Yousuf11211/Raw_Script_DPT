# To_Chnage Scripts

Standalone Python scripts for data cleaning, analysis, and model training/testing.
Each script runs independently and mirrors the Django behavior for device prompts, chunking,
output handling, and progress reporting.

## Quick Start
1) Install dependencies:
```
pip install -r requirement.txt
```
2) Run a script:
```
python <script_name>.py
```
3) Edit the config variables at the top of each script to point at your CSVs.

## Common Behavior
- Device selection:
  - Each script prints GPU detected or GPU not detected. Using CPU.
  - If GPU is detected, you can choose CPU or GPU; most scripts still use CPU-only pandas.
  - XGBoost scripts use GPU only when selected and available.
- Chunk size prompt:
  - Choose 25/100/500/1000 MB and the script estimates rows per chunk from a sample.
  - All large CSVs are processed in chunks to avoid loading full files into memory.
- Optional max rows to save:
  - If a script writes CSV output, it prompts to limit saved rows.

## Outputs
- All outputs are written to `To_Chnage/outputs/<script_name>/`.
- Output files are never overwritten; new runs append `_run2`, `_run3`, etc.

## Script Catalog

### Feature_selection_XGBoost.py
Purpose: global and per-label feature importance using XGBoost.
Inputs: `INPUT_FILE_PATH` (CSV with `label` column).
Outputs: feature report `.txt`, top-50 plot `.png`, top-feature violin plot `.png`, optional cleaned CSV, optional per-label plots.
Notes: training is sampled up to 500k rows to cap memory.

### Feature_Importance_RandomForest.py
Purpose: RandomForest feature importance across a folder or single CSV.
Inputs: `PROCESS_FOLDER`, `FOLDER_PATH` or `SINGLE_FILE_PATH` (CSV with `label` column).
Outputs: report `.txt`, report `.csv`, top-20 plot `.png`, optional cleaned CSV.
Notes: training is sampled up to 500k rows.

### Downscale_Data.py
Purpose: sample benign rows and keep all attack rows.
Inputs: `INPUT_FOLDER` containing CSVs with `label` column.
Outputs: `benign.csv`, `attacks.csv`, printed label counts.
Notes: benign rows are sampled by `BENIGN_SAMPLING_FRACTION`.

### Deletes_Mentioned_Columns.py
Purpose: batch delete specified columns from CSVs.
Inputs: CLI args for input/output folder; `BASE_COLUMNS_TO_REMOVE`; optional columns file.
Outputs: cleaned CSVs plus `deletion_summary.csv`.
Notes: supports dry-run mode and optional max rows to save.

### Count_Remove_Duplicate_Rows_Columns.py
Purpose: counts rows/columns, detects duplicate column names and duplicate rows, optional cleanup.
Inputs: `INPUT_FOLDER` with CSVs.
Outputs: optional cleaned CSVs and a report `duplicate_check_report.txt`.
Notes: duplicate rows are detected via row hashes for memory safety.

### Constant_LowVarience_Dominance_INF_Impossible.py
Purpose: multi-tool for dominance reports, validation, inf handling, low-variance cleanup, and comparison.
Inputs: `INPUT_FOLDER` with CSVs.
Outputs: dominance reports, cleaned CSVs, and optional imputation outputs.
Notes: interactive task menu (1-5) with per-task prompts.

### Compare_Colums_and_Rows_2CSV.py
Purpose: compare column consistency and row set differences between raw and processed folders.
Inputs: `raw_folder` and `processed_folder`.
Outputs: `comparison_report.txt` with column mismatches and row hash counts.
Notes: uses streaming row hashes; does not load full data into memory.

### Column_Wise_Missing_Percentage.py
Purpose: report missing and inf values per column across a folder.
Inputs: `MAIN_FOLDER` with CSVs.
Outputs: optional per-file report `.txt`.
Notes: processes each file in chunks.

### Class_Balancing.py
Purpose: balance class distributions using SMOTE/BorderlineSMOTE/ADASYN.
Inputs: `INPUT_FOLDER` with labeled CSVs.
Outputs: balanced CSVs under `outputs/Attack_Balanced/<method>/`.
Notes: sampling cap is 500k rows to reduce memory usage.

### Check_which_column_has_Mixed_Types.py
Purpose: detect columns with mixed types (string/inf).
Inputs: `CSV_FILE`.
Outputs: `mixed_type_report.txt`.
Notes: processes in chunks; focuses on string and inf issues.

### Check_Handle_Dominance_Impossible_INF_values.py
Purpose: dominance reporting, data validation, and inf handling.
Inputs: `INPUT_FOLDER` with CSVs.
Outputs: dominance reports, cleaned CSVs, imputed CSVs.
Notes: interactive task menu (1-3).

### Checks_Which_columns_need_encoding.py
Purpose: detect invalid handshake rows based on `delta_start` and `handshake_duration`.
Inputs: `INPUT_FOLDER` with CSVs containing those columns.
Outputs: cleaned CSVs with invalid rows removed.
Notes: prompts before deleting rows and saving output.

### Label_Detection_and_Splitting_for_Training_Testing.py
Purpose: label counts and train/test split.
Inputs: `PARENT_FOLDER` with CSVs containing `label` column.
Outputs: per-file report `.txt`, train/test CSVs in output folders.
Notes: split is approximately 60/40 with chunked writes and optional max rows.

### Isolation_Forest_Model.py
Purpose: train Isolation Forest on sampled benign data.
Inputs: `LARGE_BENIGN_FILE`.
Outputs: `isolation.joblib` model file.
Notes: only numeric columns are used for training.

### Hyperparameter_Tuning_XGBoost.py
Purpose: manual grid search for XGBoost hyperparameters.
Inputs: `TRAIN_FILE_PATH`, `TEST_FILE_PATH` with `label` column.
Outputs: tuning results CSV, summary report, tuned model, heatmap.
Notes: training and test sets are sampled up to 500k rows.

### Hyperparameter_Tuning_rf.py
Purpose: GridSearchCV for RandomForest.
Inputs: `TRAIN_FILE_PATH`, `TEST_FILE_PATH` with `label` column.
Outputs: tuning results CSV, summary report, tuned model, heatmap.
Notes: training and test sets are sampled up to 500k rows.

### Graph_To_Compare_Various_Attack_Number.py
Purpose: aggregate label counts across CSVs.
Inputs: `PARENT_FOLDER` with labeled CSVs.
Outputs: `Overall_Label_Distribution.csv`, `Overall_Class_Distribution.png`.

### Find_Missing_and_Separate_Bening_Attack.py
Purpose: separate benign and attack data into chunked outputs.
Inputs: `INPUT_FOLDER` with labeled CSVs.
Outputs: `Benign/` and/or `Attacks/` output files under `outputs/Separated_Model_Data/`.
Notes: interactive selection for benign/attacks/both and rows per output file.

### Model_XGBooost.py
Purpose: train XGBoost models per CSV.
Inputs: `INPUT_FOLDER` with labeled CSVs.
Outputs: model `.pkl`, label mapping `.txt`, optional report and confusion matrix.
Notes: training data is sampled up to 500k rows.

### Model_Testing.py
Purpose: run inference on a test CSV and save reports.
Inputs: `MODEL_PATH`, `LABEL_MAPPING_PATH`, `TEST_CSV_PATH`.
Outputs: optional classification report, confusion matrix, prediction counts, and predictions CSV.
Notes: predictions are streamed; max rows limit applies to saved predictions.

### Model_Random_Forest.py
Purpose: train RandomForest models per CSV.
Inputs: `INPUT_FOLDER` with labeled CSVs.
Outputs: model `.pkl`, label mapping `.txt`, optional report and confusion matrix.
Notes: training data is sampled up to 500k rows.

### Outliner_Detection.py
Purpose: IQR-based outlier detection for numeric columns.
Inputs: `FOLDER` + `FILENAME`.
Outputs: per-column plots under `outputs/Outliner_Detection/`.
Notes: analysis uses sampled data up to 500k rows.

### testing_isolation_forest.py
Purpose: evaluate Isolation Forest model on test data.
Inputs: `MODEL_FILENAME`, `TEST_DATA_FILE`, `LABEL_COLUMN`.
Outputs: evaluation report `.txt` and confusion matrix `.csv`.
Notes: predictions are streamed by chunks.

## Optional GPU Packages
- GPU detection works with `torch` or `tensorflow` if installed.
- `requirement.txt` includes these packages; remove them if you only want CPU execution.
