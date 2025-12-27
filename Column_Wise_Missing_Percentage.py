import os
import pandas as pd
import numpy as np

# Main folder with all raw datasets
main_folder = "Attacks_Removed_Constant"

# Ask user if reports should be saved
save_reports = input("Do you want to save reports to files? (y/n): ").strip().lower() == "y"

if save_reports:
    output_folder = "Downscale_Csv_20181"
    os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(main_folder):
    for file in files:
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(root, file)
        print(f"\nScanning {file_path} ...")

        try:
            # Load CSV
            df = pd.read_csv(file_path, low_memory=False)

            total_rows = len(df)
            total_cols = len(df.columns)

            # Missing values
            missing_counts = df.isna().sum()
            missing_perc = (df.isna().mean() * 100).round(2)
            missing_report = pd.DataFrame({
                "Missing Count": missing_counts,
                "Missing %": missing_perc
            })
            missing_report = missing_report[missing_report["Missing Count"] > 0]

            # Infinite values
            inf_counts = df.isin([np.inf, -np.inf]).sum()
            inf_report = inf_counts[inf_counts > 0]

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Create report lines
        report_lines = []
        report_lines.append(f"Report for {file_path}")
        report_lines.append("=" * 50)
        report_lines.append(f"Total rows: {total_rows}")
        report_lines.append(f"Total columns: {total_cols}")
        report_lines.append("")

        # Missing values
        if missing_report.empty:
            report_lines.append("No missing values found in any column.")
        else:
            report_lines.append("Columns with missing values:")
            for col, row in missing_report.iterrows():
                report_lines.append(f"{col:<25}: {row['Missing Count']} missing ({row['Missing %']}%)")
        report_lines.append("")

        # Infinite values
        if inf_report.empty:
            report_lines.append("No infinite values found in any column.")
        else:
            report_lines.append("Columns with infinite values:")
            for col, cnt in inf_report.items():
                report_lines.append(f"{col:<25}: {cnt} infinite values")
        report_lines.append("")

        # Always print
        print("\n".join(report_lines))

        # Save if user chose to
        if save_reports:
            relative_path = os.path.relpath(root, main_folder)
            report_subfolder = os.path.join(output_folder, relative_path)
            os.makedirs(report_subfolder, exist_ok=True)

            output_file = os.path.join(report_subfolder, f"{os.path.splitext(file)[0]}_report.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))

            print(f"Report saved to {output_file}")
