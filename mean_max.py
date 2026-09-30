import pandas as pd

# Load your live lab traffic
live_df = pd.read_csv("Created_Data/ssh_bruteforce.csv", low_memory=False)

# Define a few critical features that usually change between networks
features_to_check = [
    "duration",
    "packets_IAT_mean",
    "fwd_packets_count",
    "bwd_packets_count",
    "fwd_init_win_bytes",
    "bwd_init_win_bytes"
]

print("LIVE LAB TRAFFIC STATS:")
print("-" * 50)
print(live_df[features_to_check].describe().round(2))