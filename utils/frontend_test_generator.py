import os
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Tuple, List, Dict

START_TIME = datetime(2018, 3, 2, 7, 46, 53, 346213)
IP_BASE = (172, 31, 64)


def load_benign_attack(benign_folder: str, attack_folder: str, benign_limit: int, benign_label: str, attack_label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    benign_files = [os.path.join(benign_folder, f) for f in os.listdir(benign_folder) if f.endswith('.csv')]
    benign_data_list: List[pd.DataFrame] = []
    for f in benign_files:
        if sum(len(df) for df in benign_data_list) >= benign_limit:
            break
        try:
            df = pd.read_csv(f)
            rows_to_load = benign_limit - sum(len(df) for df in benign_data_list)
            benign_data_list.append(df.head(rows_to_load))
        except Exception:
            continue
    full_benign = pd.concat(benign_data_list, ignore_index=True) if benign_data_list else pd.DataFrame()
    if 'label' not in full_benign.columns:
        full_benign['label'] = benign_label
    else:
        full_benign['label'] = benign_label

    attack_files = [os.path.join(attack_folder, f) for f in os.listdir(attack_folder) if f.endswith('.csv')]
    attack_list: List[pd.DataFrame] = []
    for f in attack_files:
        try:
            attack_list.append(pd.read_csv(f))
        except Exception:
            continue
    full_attack = pd.concat(attack_list, ignore_index=True) if attack_list else pd.DataFrame()
    full_attack['label'] = attack_label
    return full_benign, full_attack


def generate_unique_timestamp(start_dt: datetime, index: int) -> datetime:
    return start_dt + timedelta(microseconds=index * 100)


def generate_unique_ip(index: int) -> str:
    seg = (index % 254) + 1
    return f"{IP_BASE[0]}.{IP_BASE[1]}.{IP_BASE[2]}.{seg}"


def generate_testing_batches(df_benign: pd.DataFrame,
                              df_attack: pd.DataFrame,
                              output_folder: str,
                              num_batches: int = 20,
                              benign_ratio: float = 0.70,
                              min_rows: int = 50,
                              max_rows: int = 100) -> Dict[str, int]:
    os.makedirs(output_folder, exist_ok=True)
    benign_indices = set(df_benign.index)
    attack_indices = set(df_attack.index)
    generated = 0
    global_counter = 0
    for i in range(1, num_batches + 1):
        rows_per = random.randint(min_rows, max_rows)
        benign_count = int(rows_per * benign_ratio)
        attack_count = rows_per - benign_count
        if len(benign_indices) < benign_count or len(attack_indices) < attack_count:
            break
        sampled_benign = random.sample(list(benign_indices), benign_count)
        sampled_attack = random.sample(list(attack_indices), attack_count)
        benign_indices.difference_update(sampled_benign)
        attack_indices.difference_update(sampled_attack)
        df_b_sample = df_benign.loc[sampled_benign].copy()
        df_a_sample = df_attack.loc[sampled_attack].copy()
        final_df = pd.concat([df_b_sample, df_a_sample], ignore_index=True).sample(frac=1).reset_index(drop=True)
        timestamps = [generate_unique_timestamp(START_TIME, global_counter + j) for j in range(len(final_df))]
        ips = [generate_unique_ip(global_counter + j) for j in range(len(final_df))]
        final_df['timestamp'] = [dt.strftime('%Y-%m-%d %H:%M:%S.%f') for dt in timestamps]
        final_df['src_ip'] = ips
        out_path = os.path.join(output_folder, f"final_testing_{i}.csv")
        final_df.to_csv(out_path, index=False)
        global_counter += len(final_df)
        generated += 1
    return {"batches_created": generated}

