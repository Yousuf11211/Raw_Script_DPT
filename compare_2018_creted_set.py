import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

CIC_FILE = "Beningtest/Benign_part_1.csv"
CREATED_FILE = "Created_Data/benign_2hour_baseline.csv"

OUTPUT_FOLDER = "benign_feature_comparison"


# ============================================================
# COLUMNS TO COMPARE
# ============================================================

COLUMNS_TO_COMPARE = [

    "packets_count",
    "subflow_bwd_packets",
    "fwd_payload_bytes_max",
    "bytes_rate",
    "bwd_packets_iat_min",
    "packets_iat_mean",
    "packets_rate",
    "payload_bytes_max",
    "bwd_bulk_duration",
    "ece_flag_counts",
    "fwd_fin_flag_counts",
    "bwd_packets_iat_mean",
    "bwd_packets_iat_max",
    "ack_flag_counts",
    "fwd_psh_flag_counts",
    "std_header_bytes",
    "bwd_syn_flag_counts",
    "bwd_packets_rate",
    "fwd_mean_header_bytes",
    "total_payload_bytes",
    "fwd_payload_bytes_std",
    "dst_port",
    "bwd_rst_flag_counts",
    "fin_flag_counts",
    "duration",
    "avg_bwd_bulk_rate",
    "fwd_syn_flag_counts",
    "fwd_total_payload_bytes",
    "fwd_packets_rate",
    "fwd_packets_iat_total",
    "fwd_packets_count",
    "bwd_payload_bytes_std",
    "bwd_total_payload_bytes",
    "fwd_ack_flag_counts",
    "packet_iat_total",
    "subflow_bwd_bytes",
    "payload_bytes_variance",
    "bwd_ack_flag_counts",
    "fwd_packets_iat_max",
    "bwd_fin_flag_counts",
    "packet_iat_std",
    "syn_flag_counts",
    "bwd_bulk_state_count",
    "subflow_fwd_packets",
    "bwd_init_win_bytes",
    "fwd_packets_iat_mean",
    "fwd_bytes_rate",
    "bwd_payload_bytes_max",
    "fwd_packets_iat_std",
    "mean_header_bytes",
    "fwd_rst_flag_counts",
    "bwd_packets_iat_total",
    "packet_iat_min",
    "bwd_total_header_bytes",
    "rst_flag_counts",
    "subflow_fwd_bytes",
    "fwd_init_win_bytes",
    "payload_bytes_std",
    "cwr_flag_counts",
    "avg_segment_size",
    "bwd_mean_header_bytes",
    "bwd_bytes_rate",
    "fwd_packets_iat_min",
    "bwd_packets_count",
    "down_up_rate",
    "fwd_avg_segment_size",
    "packet_iat_max",
    "total_header_bytes",
    "payload_bytes_mean",
    "psh_flag_counts",
    "bwd_packets_iat_std",
    "fwd_total_header_bytes"
]


# ============================================================
# SAMPLE SETTINGS
# ============================================================

# so the visual comparison is point-for-point fair.
MAX_SAMPLE_ROWS = 5000

# Large 2018 CSV is read in chunks instead of all at once.
CHUNK_SIZE = 100000


# ============================================================
# GRAPH RANGE SETTINGS
# ============================================================

# Extreme outliers can make every normal point appear at zero.
# Use the central 99% for the graph's shared X-axis.
LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5


# ============================================================
# CIC LABEL SETTINGS
# ============================================================

FILTER_CIC_BENIGN = False

BENIGN_LABEL = "benign"


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# NORMALIZE COLUMN NAME
def normalize_column_name(column_name):

    return str(
        column_name
    ).strip().lower()


# NORMALIZE REQUESTED COLUMN LIST
COLUMNS_TO_COMPARE = [

    normalize_column_name(column)

    for column in COLUMNS_TO_COMPARE

]


# Remove duplicates while keeping the original order.
COLUMNS_TO_COMPARE = list(
    dict.fromkeys(
        COLUMNS_TO_COMPARE
    )
)


# ============================================================
# GET COLUMN MAPPING
#
# Creates:
#
# {
#     "duration": "Duration",
#     "packets_count": "Packets_Count"
# }
#
# Left side  = normalized name used by our script
# Right side = actual name inside CSV
# ============================================================

def get_column_mapping(file_path):

    original_columns = list(

        pd.read_csv(
            file_path,
            nrows=0
        ).columns

    )


    mapping = {}


    for original_column in original_columns:

        normalized_column = normalize_column_name(
            original_column
        )


        # Detect unusual case where a CSV has two headers
        # that become identical after lowercasing.
        if normalized_column in mapping:

            print(
                f"[WARNING] Duplicate normalized header "
                f"'{normalized_column}' found in {file_path}"
            )

            print(
                f"          Existing: "
                f"{mapping[normalized_column]}"
            )

            print(
                f"          Duplicate: "
                f"{original_column}"
            )


        mapping[normalized_column] = original_column


    return mapping


# ============================================================
# READ COLUMN MAPPINGS
# ============================================================

cic_column_mapping = get_column_mapping(
    CIC_FILE
)

created_column_mapping = get_column_mapping(
    CREATED_FILE
)


# Lowercase/normalized versions of all CSV headers.
cic_columns = list(
    cic_column_mapping.keys()
)

created_columns = list(
    created_column_mapping.keys()
)


# ============================================================
# FIND LABEL COLUMN
# ============================================================

cic_label_column = (

    "label"

    if "label" in cic_columns

    else None

)


# ============================================================
# PRINT DATASET INFORMATION
# ============================================================

print(
    "\n=========================================="
)

print(
    "DATASET INFORMATION"
)

print(
    "=========================================="
)


print(
    f"\n2018 CSV:"
    f"\n{CIC_FILE}"
)

print(
    f"2018 total columns: "
    f"{len(cic_columns)}"
)


print(
    f"\nCreated CSV:"
    f"\n{CREATED_FILE}"
)

print(
    f"Created total columns: "
    f"{len(created_columns)}"
)


if cic_label_column is not None:

    print(
        f"\n2018 label column found:"
        f" {cic_column_mapping['label']}"
    )

else:

    print(
        "\nNo label column found in the 2018 CSV."
    )


# ============================================================
# CHECK REQUESTED COLUMNS
#
# Both lists are lowercase at this point.
# ============================================================

print(
    "\n=========================================="
)

print(
    "COLUMN CHECK"
)

print(
    "==========================================\n"
)


available_columns = []


for column in COLUMNS_TO_COMPARE:

    in_cic = column in cic_columns

    in_created = column in created_columns


    if in_cic and in_created:

        print(
            f"[OK]              {column}"
        )

        available_columns.append(
            column
        )


    elif not in_cic and not in_created:

        print(
            f"[MISSING BOTH]    {column}"
        )


    elif not in_cic:

        print(
            f"[MISSING 2018]    {column}"
        )


    elif not in_created:

        print(
            f"[MISSING CREATED] {column}"
        )


# ============================================================
# STOP IF NOTHING MATCHES
# ============================================================

if not available_columns:

    raise ValueError(
        "None of the requested columns exist in both datasets."
    )


print(
    f"\nTotal features that will be compared: "
    f"{len(available_columns)}"
)


# ============================================================
# RANDOM SAMPLE FROM LARGE CSV
#
# Important:
#
# The requested feature names are lowercase.
#
# However, pd.read_csv(usecols=...) needs the REAL header
# names from the CSV.
#
# We therefore:
#
# lowercase feature name
#       ↓
# mapping
#       ↓
# actual CSV header
#       ↓
# read column
#       ↓
# rename it to lowercase inside Python
# ============================================================

def sample_large_csv(
    file_path,
    columns,
    max_rows,
    column_mapping,
    filter_benign=False,
    label_column=None
):

    # --------------------------------------------------------
    # GET REAL CSV HEADER NAMES
    # --------------------------------------------------------

    original_use_columns = []


    for column in columns:

        normalized_column = normalize_column_name(
            column
        )

        original_column = column_mapping[
            normalized_column
        ]

        original_use_columns.append(
            original_column
        )


    # --------------------------------------------------------
    # ADD LABEL COLUMN IF NEEDED
    # --------------------------------------------------------

    if (
        filter_benign
        and label_column is not None
        and label_column in column_mapping
    ):

        original_label_column = column_mapping[
            label_column
        ]


        if (
            original_label_column
            not in original_use_columns
        ):

            original_use_columns.append(
                original_label_column
            )


    reservoir = None

    total_rows_seen = 0

    total_rows_after_filter = 0


    print(
        "\n=========================================="
    )

    print(
        f"READING: {file_path}"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # READ FILE IN CHUNKS
    # --------------------------------------------------------

    for chunk_number, chunk in enumerate(

        pd.read_csv(

            file_path,

            usecols=original_use_columns,

            chunksize=CHUNK_SIZE,

            low_memory=False

        ),

        start=1

    ):


        # ----------------------------------------------------
        # LOWERCASE ALL LOADED COLUMN HEADERS
        # ----------------------------------------------------

        chunk.columns = [

            normalize_column_name(column)

            for column in chunk.columns

        ]


        # ----------------------------------------------------
        # COUNT ROWS
        # ----------------------------------------------------

        original_chunk_size = len(
            chunk
        )

        total_rows_seen += original_chunk_size


        # ----------------------------------------------------
        # KEEP ONLY BENIGN 2018 FLOWS
        # ----------------------------------------------------

        if (
            filter_benign
            and label_column is not None
            and label_column in chunk.columns
        ):

            chunk = chunk[

                chunk[label_column]

                .astype(str)

                .str.strip()

                .str.lower()

                .eq(
                    BENIGN_LABEL
                )

            ]


        # ----------------------------------------------------
        # REMOVE LABEL AFTER FILTERING
        # ----------------------------------------------------

        if (
            label_column is not None
            and label_column in chunk.columns
        ):

            chunk = chunk.drop(
                columns=[
                    label_column
                ]
            )


        # ----------------------------------------------------
        # SKIP EMPTY CHUNK
        # ----------------------------------------------------

        if len(chunk) == 0:

            continue


        total_rows_after_filter += len(
            chunk
        )


        # ----------------------------------------------------
        # CONVERT FEATURES TO NUMERIC
        # ----------------------------------------------------

        for column in columns:

            column = normalize_column_name(
                column
            )

            chunk[column] = pd.to_numeric(

                chunk[column],

                errors="coerce"

            )


        # ----------------------------------------------------
        # REMOVE ROWS WHERE EVERY SELECTED FEATURE IS MISSING
        # ----------------------------------------------------

        chunk = chunk.dropna(

            how="all",

            subset=columns

        )


        if len(chunk) == 0:

            continue


        # ----------------------------------------------------
        # RANDOM PRIORITY
        #
        # This allows us to randomly sample across the ENTIRE
        # huge CSV instead of just taking the first 150 rows.
        # ----------------------------------------------------

        chunk["_random_priority"] = rng.random(
            len(chunk)
        )


        # ----------------------------------------------------
        # ADD TO SAMPLE RESERVOIR
        # ----------------------------------------------------

        if reservoir is None:

            reservoir = chunk


        else:

            reservoir = pd.concat(

                [
                    reservoir,
                    chunk
                ],

                ignore_index=True

            )


        # ----------------------------------------------------
        # KEEP ONLY MAX_SAMPLE_ROWS
        # ----------------------------------------------------

        if len(reservoir) > max_rows:

            reservoir = reservoir.nsmallest(

                max_rows,

                "_random_priority"

            )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        print(

            f"\rChunk: {chunk_number} | "

            f"Rows read: "
            f"{total_rows_seen:,} | "

            f"Rows after filter: "
            f"{total_rows_after_filter:,} | "

            f"Sample kept: "
            f"{len(reservoir):,}",

            end=""

        )


    print()


    # ========================================================
    # VERIFY SAMPLE
    # ========================================================

    if reservoir is None:

        raise ValueError(
            f"No usable data found in {file_path}"
        )


    # ========================================================
    # REMOVE INTERNAL RANDOM COLUMN
    # ========================================================

    reservoir = reservoir.drop(

        columns=[
            "_random_priority"
        ]

    )


    reservoir = reservoir.reset_index(
        drop=True
    )


    print(

        f"\nFinal random sample: "
        f"{len(reservoir):,} flows"

    )


    return reservoir


# ============================================================
# SAMPLE 2018 DATASET
# ============================================================

cic_sample = sample_large_csv(

    file_path=CIC_FILE,

    columns=available_columns,

    max_rows=MAX_SAMPLE_ROWS,

    column_mapping=cic_column_mapping,

    filter_benign=FILTER_CIC_BENIGN,

    label_column=cic_label_column

)


# ============================================================
# SAMPLE CREATED DATASET
# ============================================================

created_sample = sample_large_csv(

    file_path=CREATED_FILE,

    columns=available_columns,

    max_rows=MAX_SAMPLE_ROWS,

    column_mapping=created_column_mapping,

    filter_benign=False,

    label_column=None

)


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_filename(name):

    name = re.sub(

        r"[^A-Za-z0-9_-]+",

        "_",

        name

    )


    return name.strip(
        "_"
    )


# ============================================================
# CREATE POINT DISTRIBUTION GRAPH
# ============================================================

def create_point_graph(
    values,
    feature,
    dataset_name,
    output_file,
    x_min,
    x_max,
    dot_color
):

    # --------------------------------------------------------
    # KEEP ONLY VALUES INSIDE SHARED GRAPH RANGE
    # --------------------------------------------------------

    values = values[

        (values >= x_min)

        &

        (values <= x_max)

    ]


    if len(values) == 0:

        print(
            f"[WARNING] No values left for {feature}"
        )

        return


    # --------------------------------------------------------
    # RANDOM VERTICAL JITTER
    #
    # Y has NO feature meaning.
    #
    # It simply makes overlapping values easier to see.
    # --------------------------------------------------------

    y = rng.normal(

        loc=0,

        scale=0.05,

        size=len(values)

    )


    # --------------------------------------------------------
    # CREATE GRAPH
    # --------------------------------------------------------

    plt.figure(

        figsize=(
            14,
            4
        )

    )


    # --------------------------------------------------------
    # DRAW INDIVIDUAL VALUES
    # --------------------------------------------------------

    plt.scatter(

        values,

        y,

        s=12,

        alpha=0.55,

        color=dot_color

    )


    # --------------------------------------------------------
    # CENTER LINE
    # --------------------------------------------------------

    plt.axhline(

        0,

        linewidth=0.8,

        alpha=0.4

    )


    # --------------------------------------------------------
    # SAME X RANGE FOR BOTH DATASETS
    # --------------------------------------------------------

    plt.xlim(
        x_min,
        x_max
    )


    # --------------------------------------------------------
    # AXIS
    # --------------------------------------------------------

    plt.xlabel(
        feature
    )

    plt.ylabel(
        ""
    )


    # Hide Y numbers because Y is only jitter.
    plt.yticks(
        []
    )


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    plt.title(

        f"{dataset_name}\n"

        f"{feature} — "

        f"{len(values):,} sampled flows"

    )


    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    plt.grid(

        axis="x",

        alpha=0.25

    )


    plt.tight_layout()


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    plt.savefig(

        output_file,

        dpi=160

    )


    plt.close()


# ============================================================
# CREATE TWO GRAPHS FOR EACH FEATURE
# ============================================================

print(
    "\n=========================================="
)

print(
    "CREATING GRAPHS"
)

print(
    "==========================================\n"
)


for index, feature in enumerate(

    available_columns,

    start=1

):


    # ========================================================
    # 2018 VALUES
    # ========================================================

    cic_values = (

        cic_sample[feature]

        .replace(

            [
                np.inf,
                -np.inf
            ],

            np.nan

        )

        .dropna()

    )


    # ========================================================
    # CREATED DATA VALUES
    # ========================================================

    created_values = (

        created_sample[feature]

        .replace(

            [
                np.inf,
                -np.inf
            ],

            np.nan

        )

        .dropna()

    )


    # ========================================================
    # VERIFY VALUES EXIST
    # ========================================================

    if len(cic_values) == 0:

        print(

            f"[SKIPPED] {feature}: "

            f"2018 contains no usable values."

        )

        continue


    if len(created_values) == 0:

        print(

            f"[SKIPPED] {feature}: "

            f"Created dataset contains no usable values."

        )

        continue


    # ========================================================
    # COMBINE ONLY TO CALCULATE SHARED X RANGE
    #
    # The two datasets are still plotted separately.
    # ========================================================

    combined = pd.concat(

        [
            cic_values,
            created_values
        ],

        ignore_index=True

    )


    # ========================================================
    # SHARED REASONABLE RANGE
    # ========================================================

    x_min = np.percentile(

        combined,

        LOW_PERCENTILE

    )


    x_max = np.percentile(

        combined,

        HIGH_PERCENTILE

    )


    # ========================================================
    # HANDLE CONSTANT FEATURES
    # ========================================================

    if x_min == x_max:

        if x_min != 0:

            padding = abs(
                x_min
            ) * 0.05

        else:

            padding = 1


        x_min -= padding

        x_max += padding


    # ========================================================
    # SAFE FILE NAME
    # ========================================================

    file_name = safe_filename(
        feature
    )


    # ========================================================
    # NUMBER
    #
    # 01
    # 02
    # 03
    #
    # Makes Windows Explorer group each pair together.
    # ========================================================

    number = f"{index:02d}"


    # ========================================================
    # 2018 GRAPH
    # ========================================================

    cic_output = os.path.join(

        OUTPUT_FOLDER,

        f"{number}_{file_name}_2018.png"

    )


    create_point_graph(

        values=cic_values,

        feature=feature,

        dataset_name="CIC-IDS2018 Benign",

        output_file=cic_output,

        x_min=x_min,

        x_max=x_max,

        dot_color="navy"

    )


    # ========================================================
    # CREATED GRAPH
    # ========================================================

    created_output = os.path.join(

        OUTPUT_FOLDER,

        f"{number}_{file_name}_created.png"

    )


    create_point_graph(

        values=created_values,

        feature=feature,

        dataset_name="Created Benign Traffic",

        output_file=created_output,

        x_min=x_min,

        x_max=x_max,

        dot_color="darkred"

    )


    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print(
        f"[FEATURE {number}] {feature}"
    )


    print(
        f"   2018 image:   "
        f"{number}_{file_name}_2018.png"
    )


    print(
        f"   Created image:"
        f" {number}_{file_name}_created.png"
    )


    print(
        f"   2018 points:  "
        f"{len(cic_values):,}"
    )


    print(
        f"   Created points:"
        f" {len(created_values):,}"
    )


    print(
        f"   2018 median:  "
        f"{cic_values.median():.6f}"
    )


    print(
        f"   Created median:"
        f" {created_values.median():.6f}"
    )


    print(
        f"   2018 mean:    "
        f"{cic_values.mean():.6f}"
    )


    print(
        f"   Created mean: "
        f"{created_values.mean():.6f}"
    )


    print(
        f"   Graph range:  "
        f"{x_min:.6f} → {x_max:.6f}"
    )


    print()


# ============================================================
# FINISHED
# ============================================================

print(
    "\n=========================================="
)

print(
    "FINISHED"
)

print(
    "=========================================="
)


print(

    f"\nGraphs saved inside:\n"

    f"{OUTPUT_FOLDER}"

)


print(

    f"\nTotal feature pairs attempted: "

    f"{len(available_columns)}"

)


print(
    "\n2018 graphs  = NAVY"
)

print(
    "Created data = DARK RED"
)


print(
    f"\nMaximum sampled flows from each dataset: "
    f"{MAX_SAMPLE_ROWS}"
)