import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ======================
# CONFIGURATION - CHANGE THESE SETTINGS
# ======================

# Set to True if you want to process ALL CSV files in a folder
# Set to False if you want to process just ONE specific CSV file
PROCESS_FOLDER = True  # Change this to False to process a single file

# If PROCESS_FOLDER is True, set the folder path:
FOLDER_PATH = "Raw_Data_2017"

# If PROCESS_FOLDER is False, set the single CSV file path:
SINGLE_FILE_PATH = "Training_2018/training_2_validated.csv"

# Threshold for "near-zero" importance (in percentage)
# Features with importance below this will be flagged for removal
IMPORTANCE_THRESHOLD = 0.1  # 0.1% - you can change this

# ======================
# MAIN CODE - DON'T CHANGE UNLESS YOU KNOW WHAT YOU'RE DOING
# ======================

print("=== Feature Importance Analysis Tool ===")
print(f"Processing mode: {'Folder' if PROCESS_FOLDER else 'Single file'}")
print(f"Importance threshold: {IMPORTANCE_THRESHOLD}%")
print("-" * 50)

# Step 1: Load data
dfs = []

if PROCESS_FOLDER:
    print(f"Loading all CSV files from folder: {FOLDER_PATH}")
    for root, dirs, files in os.walk(FOLDER_PATH):
        for file in files:
            if not file.endswith(".csv"):
                continue
            file_path = os.path.join(root, file)
            print(f"  Loading {file_path}...")

            try:
                df = pd.read_csv(file_path, low_memory=False)
            except Exception as e:
                print(f"  ERROR reading {file_path}: {e}")
                continue

            # Check for label column
            label_cols = [col for col in df.columns if col.lower() == 'label']
            if not label_cols:
                print(f"  WARNING: No 'label' column found in {file_path}, skipping.")
                continue

            # Standardize label column name
            df.rename(columns={label_cols[0]: 'label'}, inplace=True)
            dfs.append(df)
            print(f"  ✓ Loaded {len(df)} rows from {os.path.basename(file_path)}")
else:
    print(f"Loading single file: {SINGLE_FILE_PATH}")
    try:
        df = pd.read_csv(SINGLE_FILE_PATH, low_memory=False)

        # Check for label column
        label_cols = [col for col in df.columns if col.lower() == 'label']
        if not label_cols:
            print(f"ERROR: No 'label' column found in {SINGLE_FILE_PATH}")
            exit(1)

        # Standardize label column name
        df.rename(columns={label_cols[0]: 'label'}, inplace=True)
        dfs.append(df)
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"ERROR reading {SINGLE_FILE_PATH}: {e}")
        exit(1)

if not dfs:
    print("ERROR: No valid CSV files were loaded!")
    exit(1)

# Step 2: Combine data
print(f"\nCombining {len(dfs)} dataset(s)...")
data = pd.concat(dfs, ignore_index=True)
print(f"✓ Combined dataset shape: {data.shape}")

# Step 3: Prepare data for machine learning
print("\nPreparing data...")
y = LabelEncoder().fit_transform(data['label'])
X = data.drop(columns=['label'])

# Encode categorical features
categorical_cols = X.select_dtypes(include='object').columns.tolist()
if categorical_cols:
    print(f"  Encoding {len(categorical_cols)} categorical columns...")
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

print(f"✓ Final feature matrix: {X.shape}")
print(f"✓ Number of unique labels: {len(set(y))}")

# Step 4: Train Random Forest
print("\nTraining Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X, y)
print("✓ Model training completed!")

# Step 5: Calculate feature importance
print("\nCalculating feature importance...")
importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance_pct': 100 * importances
}).sort_values(by='Importance_pct', ascending=False)

# Step 6: Save reports
print("\nSaving reports...")

# Text report
with open("Feature_Importance_Report.txt", "w", encoding="utf-8") as f:
    f.write("Feature Importance Report\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total features: {len(feat_imp_df)}\n")
    f.write(f"Dataset shape: {data.shape}\n")
    f.write("-" * 50 + "\n")
    for idx, row in feat_imp_df.iterrows():
        f.write(f"{row['Feature']:<30}: {row['Importance_pct']:.4f}%\n")

# CSV report
feat_imp_df.to_csv("Feature_Importance.csv", index=False)

# Plot top 20
plt.figure(figsize=(12, 8))
top_20 = feat_imp_df.head(20)
plt.barh(range(len(top_20)), top_20['Importance_pct'])
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel("Importance (%)")
plt.title("Top 20 Feature Importances")
plt.gca().invert_yaxis()  # Highest importance at top
plt.tight_layout()
plt.savefig("Top20_Feature_Importance.png", dpi=300, bbox_inches='tight')
plt.show()

print("✓ Feature_Importance_Report.txt saved")
print("✓ Feature_Importance.csv saved")
print("✓ Top20_Feature_Importance.png saved")

# Step 7: Handle low-importance features
print(f"\nAnalyzing features with importance < {IMPORTANCE_THRESHOLD}%...")
near_zero = feat_imp_df[feat_imp_df['Importance_pct'] < IMPORTANCE_THRESHOLD]['Feature'].tolist()

print(f"Found {len(near_zero)} features with very low importance:")
if near_zero:
    print("Low-importance features:")
    for i, feature in enumerate(near_zero, 1):
        importance = feat_imp_df[feat_imp_df['Feature'] == feature]['Importance_pct'].iloc[0]
        print(f"  {i}. {feature}: {importance:.4f}%")

    print(f"\nRemoving these {len(near_zero)} features could:")
    print("  - Speed up model training")
    print("  - Reduce overfitting")
    print("  - Simplify the model")

    response = input(f"\nDo you want to remove these {len(near_zero)} low-importance features? (y/n): ")

    if response.lower() == 'y':
        print("Removing low-importance features...")
        X_filtered = X.drop(columns=near_zero)
        data_filtered = X_filtered.copy()
        data_filtered['label'] = data['label']

        # Determine output filename
        if PROCESS_FOLDER:
            base_name = os.path.basename(FOLDER_PATH.rstrip(os.sep))
            output_filename = f"{base_name}_lessfeatures.csv"
            output_path = os.path.join(FOLDER_PATH, output_filename)
        else:
            base_name = os.path.splitext(os.path.basename(SINGLE_FILE_PATH))[0]
            output_dir = os.path.dirname(SINGLE_FILE_PATH)
            output_filename = f"{base_name}_lessfeatures.csv"
            output_path = os.path.join(output_dir, output_filename)

        data_filtered.to_csv(output_path, index=False)
        print(f"✓ Saved filtered dataset: {output_path}")
        print(f"  Original features: {X.shape[1]}")
        print(f"  Remaining features: {X_filtered.shape[1]}")
        print(f"  Features removed: {len(near_zero)}")
    else:
        print("No features were removed.")
else:
    print("✓ No low-importance features found!")

print("\n=== Analysis Complete ===")
print("Check the generated files for detailed results!")
