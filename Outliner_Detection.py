import pandas as pd
import os

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

FOLDER = "Training_2018"
FILENAME = "training_2_validated.csv"
OUT_FOLDER = os.path.join(FOLDER, "outlier_plots")

def find_iqr_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    return mask, lower, upper

def main():
    file_path = os.path.join(FOLDER, FILENAME)
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from '{FILENAME}'")

    os.makedirs(OUT_FOLDER, exist_ok=True)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found for outlier detection.")
        return

    print(f"\nFound {len(numeric_cols)} numeric columns to analyze.")
    cols_with_outliers = []

    for col in numeric_cols:
        print(f"\nProcessing column: {col}")

        df_col = df.dropna(subset=[col]).copy()
        if df_col[col].nunique() <= 1:
            print(f"  Skipping column '{col}' (not enough unique values).")
            continue

        outlier_mask, lower, upper = find_iqr_outliers(df_col, col)
        n_outliers = outlier_mask.sum()
        print(f"  IQR thresholds -> Lower: {lower:.2f}, Upper: {upper:.2f}")
        print(f"  Found {n_outliers} outliers in '{col}'.")

        if n_outliers > 0:
            cols_with_outliers.append(col)
            # Print label breakdown for outliers
            if 'label' in df_col.columns:
                label_counts = df_col.loc[outlier_mask, 'label'].value_counts()
                print(f"  Outlier label counts:")
                for label, count in label_counts.items():
                    print(f"    {label}: {count}")
        else:
            print(f"  No outliers found for '{col}'.")

        # Generate and save ONLY the plot (PNG)
        if HAS_PLOT:
            plt.figure(figsize=(10, 6))
            if df_col[col].nunique() > 50:
                df_col[col].hist(bins=50, color="steelblue", edgecolor="black")
                plt.title(f"Histogram of '{col}'")
            else:
                counts = df_col[col].value_counts().sort_index()
                plt.bar(counts.index, counts.values, color="steelblue")
                plt.title(f"Bar Chart of '{col}' Value Counts")

            plt.axvline(lower, color='red', linestyle='--', label='Lower IQR Bound')
            plt.axvline(upper, color='green', linestyle='--', label='Upper IQR Bound')
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.legend(loc='upper right')
            plt.tight_layout()

            out_plot = os.path.join(OUT_FOLDER, f"{col}_plot.png")
            plt.savefig(out_plot)
            plt.close()
            print(f"  Saved plot to: {out_plot}")
        else:
            print("  Plot skipped (matplotlib not installed).")

    print("\n===========================")
    print(f"\nPlots saved in folder: {OUT_FOLDER}")
    print(f"\nColumns with outlier values: {cols_with_outliers}")
    print(f"Number of columns with outliers: {len(cols_with_outliers)}")

if __name__ == "__main__":
    main()
