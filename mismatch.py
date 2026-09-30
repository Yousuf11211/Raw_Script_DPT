import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = (
    r"C:\Users\Yousuf\Desktop\Raw_Script_DPT"
    r"\Bening1\Model_1_Ready_to_train.csv"
)

IAT_ABNORMAL_THRESHOLD = 1000


top_features = [
    "mean_header_bytes",
    "fwd_mean_header_bytes",
    "bwd_init_win_bytes",
    "dst_port",
    "bwd_mean_header_bytes",
    "bwd_payload_bytes_max",
    "fwd_init_win_bytes",
    "fwd_payload_bytes_max",
    "payload_bytes_max",
    "bwd_packets_iat_total",
    "packet_iat_total",
    "bwd_packets_iat_mean",
    "bwd_packets_iat_max",
    "std_header_bytes",
    "fwd_payload_bytes_std",
    "bwd_packets_iat_min",
    "bwd_payload_bytes_std",
    "packet_iat_min",
    "fwd_packets_rate",
    "payload_bytes_variance",
]


iat_columns = [
    "bwd_packets_iat_total",
    "bwd_packets_iat_mean",
    "bwd_packets_iat_max",
    "bwd_packets_iat_min",
    "packet_iat_total",
    "packet_iat_min",
]


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 90)
print("TRAINING DATA FEATURE DISTRIBUTION ANALYSIS")
print("=" * 90)

print(f"\nLoading training CSV:\n{TRAIN_CSV}")

df = pd.read_csv(
    TRAIN_CSV,
    low_memory=False
)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

print(f"\nRows loaded: {len(df):,}")
print(f"Columns found: {len(df.columns):,}")


# ============================================================
# CHECK LABEL COLUMN
# ============================================================

if "label" not in df.columns:
    raise ValueError(
        "The training CSV does not contain a 'label' column."
    )

print("\nLabels:")
print(
    df["label"]
    .value_counts(dropna=False)
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = list(
    dict.fromkeys(
        top_features + iat_columns
    )
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print("\n[ERROR] Missing required columns:")

    for col in missing_columns:
        print(f"   - {col}")

    raise SystemExit(1)

print(
    "\n[+] All required diagnostic features "
    "are available."
)


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

for col in required_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# CLASS-1 FEATURE STATISTICS
# ============================================================

attack = df[
    df["label"] == 1
].copy()

print("\n" + "=" * 90)
print("TRAINING CLASS-1 FEATURE STATISTICS")
print("=" * 90)

print(
    f"\nClass-1 rows found: "
    f"{len(attack):,}"
)

if attack.empty:

    print(
        "\n[ERROR] No rows with label == 1 "
        "were found."
    )

    raise SystemExit(1)


stats = attack[
    top_features
].agg(
    [
        "min",
        "median",
        "max",
    ]
).T

stats["mean"] = (
    attack[
        top_features
    ].mean()
)

stats["zero_percent"] = (
    (
        attack[
            top_features
        ] == 0
    )
    .mean()
    * 100
)

print()
print(
    stats.to_string()
)


# ============================================================
# ABNORMAL IAT FREQUENCY BY LABEL
# ============================================================

print("\n" + "=" * 90)
print("ABNORMAL IAT FREQUENCY BY LABEL")
print("=" * 90)

labels = sorted(
    df["label"]
    .dropna()
    .unique()
)

for label_value in labels:

    subset = df[
        df["label"] == label_value
    ]

    print(
        f"\nLabel {label_value}: "
        f"{len(subset):,} rows"
    )

    print("-" * 90)

    for col in iat_columns:

        values = (
            subset[col]
            .dropna()
        )

        if values.empty:

            print(
                f"  {col:<25} "
                f"No valid numeric values"
            )

            continue

        abnormal = (
            values
            > IAT_ABNORMAL_THRESHOLD
        )

        abnormal_count = int(
            abnormal.sum()
        )

        abnormal_percent = (
            abnormal.mean()
            * 100
        )

        print(
            f"  {col:<25} "
            f"{abnormal_count:>8,} abnormal "
            f"({abnormal_percent:6.2f}%)"
        )


# ============================================================
# IAT DISTRIBUTION SUMMARY BY LABEL
# ============================================================

print("\n" + "=" * 90)
print("IAT DISTRIBUTION SUMMARY BY LABEL")
print("=" * 90)

for label_value in labels:

    subset = df[
        df["label"] == label_value
    ]

    print(
        f"\nLabel {label_value}:"
    )

    summary_rows = []

    for col in iat_columns:

        values = (
            subset[col]
            .dropna()
        )

        if values.empty:
            continue

        summary_rows.append(
            {
                "feature": col,
                "min": values.min(),
                "median": values.median(),
                "mean": values.mean(),
                "max": values.max(),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if not summary_df.empty:

        summary_df = (
            summary_df
            .set_index("feature")
        )

        print(
            summary_df.to_string()
        )


# ============================================================
# PORT-22 TRAINING TRAFFIC
# ============================================================

print("\n" + "=" * 90)
print("PORT-22 TRAINING TRAFFIC COMPARISON")
print("=" * 90)


attack_ssh = df[
    (df["label"] == 1)
    & (df["dst_port"] == 22)
].copy()


benign_ssh = df[
    (df["label"] == 0)
    & (df["dst_port"] == 22)
].copy()


print(
    f"\nClass-1 port-22 rows: "
    f"{len(attack_ssh):,}"
)

print(
    f"Class-0 port-22 rows: "
    f"{len(benign_ssh):,}"
)


# ============================================================
# HELPER FOR PORT-22 STATISTICS
# ============================================================

def show_stats(
    name,
    data
):

    print(
        "\n" + "-" * 90
    )

    print(name)

    print(
        "-" * 90
    )

    if data.empty:

        print(
            "No rows found."
        )

        return

    result = data[
        top_features
    ].agg(
        [
            "min",
            "median",
            "max",
        ]
    ).T

    result["mean"] = (
        data[
            top_features
        ].mean()
    )

    result["zero_percent"] = (
        (
            data[
                top_features
            ] == 0
        )
        .mean()
        * 100
    )

    print(
        result.to_string()
    )


# ============================================================
# SHOW CLASS-1 / CLASS-0 PORT-22 STATS
# ============================================================

show_stats(
    "CLASS-1 PORT-22 TRAINING FLOWS",
    attack_ssh
)

show_stats(
    "CLASS-0 PORT-22 TRAINING FLOWS",
    benign_ssh
)


# ============================================================
# PORT-22 ABNORMAL IAT COMPARISON
# ============================================================

print("\n" + "=" * 90)
print("PORT-22 ABNORMAL IAT COMPARISON")
print("=" * 90)


port22_groups = [
    (
        "Class-1 port 22",
        attack_ssh
    ),
    (
        "Class-0 port 22",
        benign_ssh
    ),
]


for name, subset in port22_groups:

    print(
        f"\n{name}: "
        f"{len(subset):,} rows"
    )

    print(
        "-" * 90
    )

    if subset.empty:

        print(
            "No rows found."
        )

        continue

    for col in iat_columns:

        values = (
            subset[col]
            .dropna()
        )

        if values.empty:

            print(
                f"  {col:<25} "
                f"No valid values"
            )

            continue

        abnormal = (
            values
            > IAT_ABNORMAL_THRESHOLD
        )

        abnormal_count = int(
            abnormal.sum()
        )

        abnormal_percent = (
            abnormal.mean()
            * 100
        )

        print(
            f"  {col:<25} "
            f"{abnormal_count:>8,} abnormal "
            f"({abnormal_percent:6.2f}%)"
        )


# ============================================================
# PORT-22 DIRECT MEDIAN COMPARISON
# ============================================================

print("\n" + "=" * 90)
print("PORT-22 MEDIAN COMPARISON")
print("=" * 90)


comparison_rows = []


for feature in top_features:

    class1_median = (
        attack_ssh[feature].median()
        if not attack_ssh.empty
        else np.nan
    )

    class0_median = (
        benign_ssh[feature].median()
        if not benign_ssh.empty
        else np.nan
    )

    comparison_rows.append(
        {
            "feature": feature,
            "class1_port22_median": class1_median,
            "class0_port22_median": class0_median,
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df = (
    comparison_df
    .set_index("feature")
)

print(
    comparison_df.to_string()
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)