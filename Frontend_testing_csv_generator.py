import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

# --- CONFIGURATION ---
# IMPORTANT: Adjust these paths to match your folder structure
BENIGN_FOLDER = 'Testing_isolation_model_cleaned'  # Folder containing benign CSV files
ATTACK_FOLDER = 'Testing_Attack'  # Folder containing attack CSV files
OUTPUT_FOLDER = 'frontend_testing'  # The script will create this folder
BENIGN_LABEL = 'BENIGN'
ATTACK_LABEL = 'ATTACK'
BENIGN_LOAD_LIMIT = 5000  # Max number of benign rows to load


# --- SETUP AND FOLDER CREATION ---

def setup_folders():
    """Creates input and output folders if they don't exist."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Directory structure prepared.")
    print(f"Please place your benign CSVs in: '{BENIGN_FOLDER}'")
    print(f"Please place your attack CSVs in: '{ATTACK_FOLDER}'")


# --- UNIQUE DATA GENERATION FUNCTIONS ---

# Start time for the sequence of unique timestamps
# Note: The requested format 2018-03-02 07:46:53.346213 is a datetime object
START_TIME = datetime(2018, 3, 2, 7, 46, 53, 346213)
IP_BASE = (172, 31, 64)  # Using the requested 172.31.64.xxx structure


def generate_unique_timestamp(start_dt, index):
    """Generates a unique timestamp by adding a specific microsecond offset."""
    # We add 100 microseconds per index to ensure uniqueness across 50-100 rows
    return start_dt + timedelta(microseconds=index * 100)


def generate_unique_ip(index):
    """Generates a unique source IP address (e.g., 172.31.64.111) based on index."""
    # Simple strategy: use the index modulo 254 to create a unique-looking IP segment
    segment = (index % 254) + 1  # Ensure it is not 0 or 255
    return f"{IP_BASE[0]}.{IP_BASE[1]}.{IP_BASE[2]}.{segment}"


# --- DATA LOADING FUNCTION ---

def load_data():
    """Loads all attack data and a limited sample of benign data."""

    # 1. Load Benign Data (max 1000 rows)
    benign_files = [os.path.join(BENIGN_FOLDER, f) for f in os.listdir(BENIGN_FOLDER) if f.endswith('.csv')]
    benign_data_list = []

    for f in benign_files:
        if sum(len(df) for df in benign_data_list) >= BENIGN_LOAD_LIMIT:
            break
        try:
            df = pd.read_csv(f)
            rows_to_load = BENIGN_LOAD_LIMIT - sum(len(df) for df in benign_data_list)
            benign_data_list.append(df.head(rows_to_load))
        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty file {f}")
        except Exception as e:
            print(f"Error loading {f}: {e}")

    full_benign_data = pd.concat(benign_data_list, ignore_index=True)

    # *** FIX 1: Use 'label' (lowercase) for benign data ***
    # Add default label if missing (required for segmentation)
    if 'label' not in full_benign_data.columns:
        full_benign_data['label'] = BENIGN_LABEL
    else:
        # Ensure loaded data is correctly labeled as benign (overwrite)
        full_benign_data['label'] = BENIGN_LABEL

    print(f"Loaded {len(full_benign_data)} total rows of Benign data (max {BENIGN_LOAD_LIMIT}).")

    # 2. Load Full Attack Data
    attack_files = [os.path.join(ATTACK_FOLDER, f) for f in os.listdir(ATTACK_FOLDER) if f.endswith('.csv')]

    try:
        full_attack_data = pd.concat([pd.read_csv(f) for f in attack_files], ignore_index=True)
    except pd.errors.EmptyDataError:
        print("Warning: No attack data loaded or files are empty.")
        full_attack_data = pd.DataFrame()
    except Exception as e:
        print(f"Error loading attack data: {e}")
        full_attack_data = pd.DataFrame()

    # *** FIX 2 & 3: Use 'label' (lowercase) and simplify logic to avoid KeyError ***
    # Ensure the 'label' column exists and correctly label all rows as ATTACK.
    full_attack_data['label'] = ATTACK_LABEL

    print(f"Loaded {len(full_attack_data)} total rows of Attack data.")

    return full_benign_data, full_attack_data


# --- BATCH GENERATION LOGIC ---

def generate_batches(df_benign, df_attack, num_batches=20):
    """Generates 20 unique testing CSV files with mixed data."""

    # Use sets to track available row indices (crucial for unique sampling)
    benign_indices = set(df_benign.index)
    attack_indices = set(df_attack.index)

    # Check minimum required data for a single batch (50 rows, 35 benign, 15 attack)
    if len(df_benign) < 35 * num_batches:
        print(f"FATAL: Need at least {35 * num_batches} unique benign rows, but only have {len(df_benign)}.")
        return
    if len(df_attack) < 15 * num_batches:
        print(f"FATAL: Need at least {15 * num_batches} unique attack rows, but only have {len(df_attack)}.")
        return

    # Initialize counter for unique IP/Timestamp generation
    global_index_counter = 0

    for i in range(1, num_batches + 1):
        # 1. Determine batch size (50-100 rows)
        rows_per_csv = random.randint(50, 100)

        # 2. Calculate row counts (70% Benign, 30% Attack)
        benign_count = int(rows_per_csv * 0.70)
        attack_count = rows_per_csv - benign_count

        # --- Sample Benign Data ---
        available_benign = list(benign_indices)
        if len(available_benign) < benign_count:
            print(f"WARN: Not enough unique benign data for file {i}. Stopping at batch {i - 1}.")
            break

        sampled_benign_indices = random.sample(available_benign, benign_count)
        benign_indices.difference_update(sampled_benign_indices)  # Remove used indices
        df_benign_sample = df_benign.loc[sampled_benign_indices].copy()

        # --- Sample Attack Data ---
        available_attack = list(attack_indices)
        if len(available_attack) < attack_count:
            print(f"WARN: Not enough unique attack data for file {i}. Stopping at batch {i - 1}.")
            break

        sampled_attack_indices = random.sample(available_attack, attack_count)
        attack_indices.difference_update(sampled_attack_indices)  # Remove used indices
        df_attack_sample = df_attack.loc[sampled_attack_indices].copy()

        # --- Combine, Process, and Save ---
        final_df = pd.concat([df_benign_sample, df_attack_sample], ignore_index=True)
        final_df = final_df.sample(frac=1).reset_index(drop=True)  # Shuffle the rows

        # Add unique Timestamp and Src_IP columns
        timestamps = [generate_unique_timestamp(START_TIME, global_index_counter + j) for j in range(len(final_df))]
        src_ips = [generate_unique_ip(global_index_counter + j) for j in range(len(final_df))]

        final_df['timestamp'] = [dt.strftime('%Y-%m-%d %H:%M:%S.%f') for dt in timestamps]
        final_df['src_ip'] = src_ips

        # Increment the global counter for the next batch
        global_index_counter += len(final_df)

        # Print label counts (Using 'label' lowercase)
        label_counts = final_df['label'].value_counts().to_dict()
        print(f"\n--- Batch {i}: final_testing_{i}.csv ---")
        print(f"  Total Rows: {len(final_df)}")
        print(f"  Label Counts: {label_counts}")

        # Save the new CSV
        output_path = os.path.join(OUTPUT_FOLDER, f'final_testing_{i}.csv')
        final_df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")

    print("\n--- Script Finished ---")


if __name__ == '__main__':
    setup_folders()

    # Proceed only if the folders are set up and data exists
    df_benign, df_attack = load_data()

    if not df_benign.empty and not df_attack.empty:
        generate_batches(df_benign, df_attack)
    else:
        print("\nFATAL: Insufficient data loaded. Please check that your input folders contain CSV files with data.")