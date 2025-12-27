#This script is used to downscale the data
#Generate 2 new csv files (bening and attacks)
#show count for each label in each file
#The output csv files are saved in the output folder
#Data is used for testing the models




import os
import pandas as pd

#Input folder with all csv files
INPUT_FOLDER = "2018_Separated_Nomissing"

#output folder to save the new csv files
OUTPUT_FOLDER = "Downscale_Csv_2018"

#combine 10% of the bening data from each file
BENIGN_SAMPLING_FRACTION = 0.1

# A fixed number to ensure that the "random" shuffle is the same every time you
# run the script. This makes your experiments repeatable.
RANDOM_STATE = 42


def main():
    """
    The main function that runs the entire data downscaling process.
    """
    # ==========================================================================
    # --- 2. Initialization ---
    # We set up our environment and create empty containers for the data.
    # ==========================================================================

    print("Starting the downscaling and separation process...")

    # Create the output folder if it doesn't already exist.
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Define the full paths for the two final output files.
    output_benign_file = os.path.join(OUTPUT_FOLDER, "benign.csv")
    output_attacks_file = os.path.join(OUTPUT_FOLDER, "attacks.csv")

    # These empty lists will act as "buckets" to hold the data pieces
    # before we combine them at the end.
    benign_dfs = []
    attack_dfs = []

    # ==========================================================================
    # --- 3. Main Processing Loop ---
    # We walk through all subfolders and files to find and process every CSV.
    # ==========================================================================

    # os.walk is a Python tool that explores a directory and all its subdirectories.
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            # We only care about files that end with .csv
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(root, file)
            print(f"\nProcessing {file_path} ...")

            try:
                # Read the entire CSV file into a pandas DataFrame.
                df = pd.read_csv(file_path, low_memory=False)
            except Exception as e:
                print(f"  -> Error reading file: {e}. Skipping.")
                continue

            # Find the 'label' column, ignoring if it's 'Label', 'label', etc.
            label_col_found = None
            for col in df.columns:
                if col.lower() == "label":
                    label_col_found = col
                    break

            if not label_col_found:
                print(f"  -> No label column found in this file. Skipping.")
                continue

            # Standardize the column name to lowercase 'label' for consistency.
            df.rename(columns={label_col_found: "label"}, inplace=True)

            # Check if the file contains benign data. We use .any() in case
            # a file accidentally contains a mix of labels.
            if df["label"].str.lower().eq("benign").any():
                print(f"  -> Identified as Benign. Sampling {BENIGN_SAMPLING_FRACTION:.0%} of its rows.")
                sample_df = df.sample(frac=BENIGN_SAMPLING_FRACTION, random_state=RANDOM_STATE)
                benign_dfs.append(sample_df)
                print(f"  -> Kept {len(sample_df):,} rows.")
            else:
                print("  -> Identified as Attack. Keeping 100% of its rows.")
                attack_dfs.append(df)
                print(f"  -> Kept {len(df):,} rows.")

    # ==========================================================================
    # --- 4. Final Combination, Shuffling, and Saving ---
    # After processing all files, we combine the data pieces and save them.
    # ==========================================================================

    final_benign_df = None
    final_attacks_df = None

    # Process and save the Benign data bucket.
    if benign_dfs:
        print("\nStep 4a: Combining, shuffling, and saving all Benign data...")
        # pd.concat merges all the small DataFrame pieces into one large one.
        final_benign_df = pd.concat(benign_dfs, ignore_index=True)
        # .sample(frac=1) shuffles all the rows randomly.
        final_benign_df = final_benign_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        final_benign_df.to_csv(output_benign_file, index=False)
        print(f"-> Benign data saved to: {output_benign_file}")
    else:
        print("\nNo benign data was processed.")

    # Process and save the Attack data bucket.
    if attack_dfs:
        print("\nStep 4b: Combining, shuffling, and saving all Attack data...")
        final_attacks_df = pd.concat(attack_dfs, ignore_index=True)
        final_attacks_df = final_attacks_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        final_attacks_df.to_csv(output_attacks_file, index=False)
        print(f"-> Attack data saved to: {output_attacks_file}")
    else:
        print("\nNo attack data was processed.")

    # ==========================================================================
    # --- 5. Final Report ---
    # Display the final counts for each of the new datasets.
    # ==========================================================================

    print("\n" + "=" * 60)
    print(" " * 20 + "FINAL DATASET REPORT")
    print("=" * 60)

    if final_benign_df is not None:
        print("\n--- Counts for benign.csv ---")
        print(f"Total Rows: {len(final_benign_df):,}")
        print(final_benign_df['label'].value_counts())
    else:
        print("\n--- No benign.csv was created ---")

    if final_attacks_df is not None:
        print("\n--- Counts for attacks.csv ---")
        print(f"Total Rows: {len(final_attacks_df):,}")
        print(final_attacks_df['label'].value_counts())
    else:
        print("\n--- No attacks.csv was created ---")

    print("\n" + "=" * 60)
    print("Process finished successfully! 🎉")


# ==============================================================================
# --- 6. Script Execution ---
# This line ensures that the main() function is called only when the script
# is run directly.
# ==============================================================================
if __name__ == "__main__":
    main()