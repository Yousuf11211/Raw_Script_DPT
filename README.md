# Thesis Data Tool

A Streamlit-based workbench for end-to-end data cleaning, processing, analysis, model training, testing, and utility operations on large tabular / network security datasets.

## Structure
```
pages/
  0_Landing.py                # Landing page (overview + quick hub links)
  DataCleaning/               # Data Cleaning tools
    hub.py                    # Category hub
    1_Data_Validation_and_Dedup.py
    2_INF_Handling.py
    3_Dominance_and_Reports.py
    4_Constant_and_LowVariance.py
    5_Mixed_Type_Analysis.py
    6_Encoding_Candidates.py
    10_Delete_Columns_UI.py
  DataProcessing/
    hub.py
    7_Class_Balancing.py
    8_Downscale_Dataset.py
    12_Separate_and_Save_Sets.py
    17_Merge_Shuffle_Polars.py
    18_Outlier_Detection.py
  DataAnalysis/
    hub.py
    11_Feature_Importance.py
  ModelTraining/
    hub.py
    13_Hyperparameter_Tuning.py
    14_Isolation_Forest.py
    15_Attack_Model_Train_Test.py
  ModelTesting/
    hub.py
    16_Test_Isolation_Forest.py
    19_SHAP_Explanations.py
  Utilities/
    hub.py
    9_Compare_Raw_vs_Processed.py
    20_Frontend_Test_Batch_Generator.py

  # Legacy numbered hub files (21–26) now contain redirect stubs only.
```
Redirect stubs remain for backwards compatibility (e.g. `pages/1_Data_Validation_and_Dedup.py` switches to `pages/DataCleaning/1_Data_Validation_and_Dedup.py`).

## Navigation
- Top bar: high-level category buttons.
- Clicking a category takes you to its hub page showing only relevant tool buttons.
- Landing page provides quick hub access and session summary.

## Common Header Pattern
Each tool page calls `common_header(...)` to:
- Select one or more input files / folders from project tree (no drag-and-drop upload used).
- Specify output folder and optional save toggle.
- Normalize dataset loading (lazy Polars scanning stored in `st.session_state['current_lazy_frame']`).

## Session State Keys
- `current_file_path`: Active file path selected via header/browser.
- `current_lazy_frame`: Polars LazyFrame for the active dataset.
- `applied_filters`: List of descriptive strings logging transformation steps.

## Adding a New Tool
1. Place new page inside appropriate category subfolder.
2. Import utilities and call:
   ```python
   initialize_state()
   inject_global_styles()
   render_top_nav(current_page="pages/<Category>/<File>.py")
   hdr = common_header("Your Page Title", num_inputs=1, input_specs=[{"label": "Input CSV", "kind": "file", "allowed_exts": [".csv"]}], default_output_folder="output")
   ```
3. Use `hdr['input_paths']` and `hdr['output_folder']` as needed.
4. Optionally append actions to `st.session_state['applied_filters']`.

## Developer
Syed Yousuf Uddin

## Running
```bash
streamlit run app.py
```

## Notes
- All processing is lazy until you explicitly save (write) results.
- Redirect stubs can be removed once all legacy links are updated externally.
- No emojis are used in UI titles or navigation to maintain a professional look.

