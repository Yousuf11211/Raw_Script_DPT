import pandas as pd

# Load the live lab traffic
df = pd.read_csv("Created_Data/ssh_bruteforce.csv", low_memory=False)

# Filter out the epoch timestamp bug (keep rows where IAT mean is less than 1 million)
clean_df = df[df["packets_IAT_mean"] < 1000000]

print(f"Original flows: {len(df)}")
print(f"Clean flows remaining: {len(clean_df)}")

# Save the clean data
clean_df.to_csv("ssh_bruteforce_clean.csv", index=False)