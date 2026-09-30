import pandas as pd

df = pd.read_csv("Live_Traffic_Predictions.csv")

print(df["IDS_Confidence"].describe())

print("\nLowest confidence:")
print(df.nsmallest(10, "IDS_Confidence")[
    ["src_ip", "dst_ip", "dst_port", "IDS_Prediction", "IDS_Confidence"]
])