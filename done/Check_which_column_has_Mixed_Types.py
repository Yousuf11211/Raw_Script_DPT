import os
import pandas as pd
from collections import Counter, defaultdict
import math

# CSV file path
csv_file = "../2017/final_testing_1.csv"

# Chunk size for large files
chunk_size = 100_000


# Helper function to classify a value's data type
def classify_value(val):
    """
    Classifies a single value into 'NaN', 'inf', 'integer', 'float', or 'string'.
    """
    # Check for null/NaN values first
    if pd.isna(val):
        return "NaN"

    # Try converting to a number
    try:
        num = float(val)

        # Check for special float types like infinity
        if math.isinf(num):
            return "inf"

        # Check if the number is a whole number
        if num.is_integer():
            return "integer"

        # If not special or an integer, it's a standard float
        return "float"

    # If the float() conversion fails, it must be a string
    except (ValueError, TypeError):
        return "string"


# Use defaultdict(Counter) to store the counts of each data type per column
col_type_counts = defaultdict(Counter)

print("Starting to process the CSV file to find critical errors...")

# Read the large CSV file in chunks, forcing all data to be read as strings initially
for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False, dtype=str)):
    print(f"Processing chunk {i + 1}...")
    # For each column in the current chunk...
    for col in chunk.columns:
        # ...map the classify_value function to every item and update the counts
        col_type_counts[col].update(chunk[col].map(classify_value))

print("\n--- Analysis Complete ---")

# Print a summary of columns that contain the critical 'string' or 'inf' types
print("\nSummary of Columns with Critical Errors:\n")
for col, counts in col_type_counts.items():
    # MODIFIED LINE: Only report on columns containing 'string' or 'inf'
    if 'string' in counts or 'inf' in counts:
        print(f"Column: {col} (CRITICAL - mixed types found)")
        for val_type, cnt in counts.items():
            print(f"  {val_type} --- {cnt}")
        print("-" * 40)