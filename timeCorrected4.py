import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD AND PREPROCESS RAW DATA
# ==============================================================================
print("=" * 70)
print("FRAUD DETECTION - COMPLETE FEATURE PIPELINE (DATE FIX)")
print("=" * 70)
print("\n--- 1. Loading and Preprocessing Raw Data ---")

# --- Load Raw Data ---
filename_trans = "testData/firstTestData/acct_transaction.csv"
filename_alert = "testData/firstTestData/acct_alert.csv"
df_trans_raw = pd.read_csv(filename_trans)
df_alert_raw = pd.read_csv(filename_alert)

print(f"Loaded {len(df_trans_raw):,} transactions")
print(f"Loaded {len(df_alert_raw):,} raw alert entries")

# --- Preprocessing ---
self_map = {"Y": 1, "N": 0, "UNK": -1}
df_trans_raw['is_self_txn_enc'] = df_trans_raw['is_self_txn'].map(self_map)

exchange_rates = {
    'TWD': 1.0, 'USD': 32.42, 'JPY': 0.20, 'AUD': 21.55, 'CNY': 4.46, 'EUR': 34.71,
    'SEK': 2.91, 'GBP': 41.23, 'HKD': 4.15, 'THB': 0.94, 'CAD': 23.51, 'NZD': 20.15,
    'CHF': 35.60, 'SGD': 24.08, 'ZAR': 1.78, 'MXN': 1.85
}
df_trans_raw['txn_amt_twd'] = df_trans_raw['txn_amt'] * df_trans_raw['currency_type'].map(exchange_rates)

time_parts = df_trans_raw['txn_time'].str.split(':', expand=True).astype(int)
df_trans_raw['txn_time_seconds'] = time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
df_trans_raw['txn_hour'] = time_parts[0]

df_trans_raw['channel_type'] = df_trans_raw['channel_type'].replace('UNK', -1).astype(int)

# ==============================================================================
# 1B. DATE AND TIMESTAMP FIX
# ==============================================================================
# DO NOT CONVERT txn_date or event_date to datetime. Treat them as integers.
# Ensure they are loaded as integers (they should be, but just in case)
df_trans_raw['txn_date'] = df_trans_raw['txn_date'].astype(int)
df_alert_raw['event_date'] = df_alert_raw['event_date'].astype(int)

# FIX txn_timestamp: Create a total_seconds_from_start integer
# (Day * 86400 seconds/day) + seconds_into_day
df_trans_raw['txn_timestamp'] = (df_trans_raw['txn_date'] * 86400) + df_trans_raw['txn_time_seconds']
print("✓ Correctly parsing dates as integers (days).")
# ==============================================================================

# --- Currency Dummies (for advanced features) ---
currency_dummies = pd.get_dummies(df_trans_raw['currency_type'], prefix='currency', dtype=int)
df_trans_raw = pd.concat([df_trans_raw, currency_dummies], axis=1)
currency_cols = [col for col in currency_dummies.columns]
print(f"✓ Created {len(currency_cols)} currency dummy columns")

print("✓ Preprocessing complete.")

# ==============================================================================
# 2. ANALYZE DATE RANGES (FIXED)
# ==============================================================================
print("\n--- 2. Analyzing Date Ranges ---")

txn_min_date = df_trans_raw['txn_date'].min()
txn_max_date = df_trans_raw['txn_date'].max()
alert_min_date = df_alert_raw['event_date'].min()
alert_max_date = df_alert_raw['event_date'].max()

print(f"  • Transaction dates (days): {txn_min_date} to {txn_max_date}")
print(f"  • Alert dates (days): {alert_min_date} to {alert_max_date}")
print(f"  • Total alerts in raw data: {len(df_alert_raw)}")
print(f"  • Unique accounts with alerts: {df_alert_raw['acct'].nunique()}")

# ==============================================================================
# 3. COMPETITION TIME-SLICING LOGIC (FIXED)
# ==============================================================================
print("\n--- 3. Applying Competition Time-Slicing Logic ---")

# T is now an integer (e.g., 121)
T = df_trans_raw['txn_date'].max() 

# Use simple integer math, not Timedelta
T_minus_7 = T - 7
T_minus_30 = T - 30

print(f"  • T (Present/Cutoff Day): {T}")
print(f"  • Recent 7-day Window (Days): {T_minus_7 + 1} to {T}")
print(f"  • Recent 30-day Window (Days): {T_minus_30 + 1} to {T}")

df_features = df_trans_raw[df_trans_raw['txn_date'] <= T].copy()
df_recent_7d = df_features[df_features['txn_date'] > T_minus_7].copy()
df_recent_30d = df_features[df_features['txn_date'] > T_minus_30].copy()

# ==============================================================================
# 4. DETERMINE LABEL STRATEGY (FIXED)
# ==============================================================================
print("\n--- 4. Determining Label Strategy ---")

df_labels_future = df_alert_raw[df_alert_raw['event_date'] > T].copy()

if len(df_labels_future) > 0:
    print("  • SCENARIO: Test/Prediction mode")
    # Use integer math
    T_plus_31 = T + 31 
    df_labels = df_alert_raw[
        (df_alert_raw['event_date'] > T) & 
        (df_alert_raw['event_date'] <= T_plus_31)
    ].copy()
    print(f"  • Future window (Days): {T + 1} to {T_plus_31}")
else:
    print("  • SCENARIO: Training mode")
    df_labels = df_alert_raw.copy()

alerted_accounts_set = set(df_labels['acct'])

print(f"  • Total transactions for features: {len(df_features):,}")
print(f"  • Unique accounts to be flagged: {len(alerted_accounts_set):,}")

# ==============================================================================
# 5. FEATURE ENGINEERING
# ==============================================================================
print("\n--- 5. Building Account-Level Feature Set ---")

all_accts = pd.concat([df_features['from_acct'], df_features['to_acct']]).unique()
account_df = pd.DataFrame(index=all_accts)
account_df.index.name = 'acct'
print(f"  • Total unique accounts: {len(all_accts):,}")

# This duration is now in seconds
dataset_duration_seconds = (T * 86400) - (df_features['txn_date'].min() * 86400)
if dataset_duration_seconds == 0:
    dataset_duration_seconds = 365 * 24 * 3600

# ==============================================================================
# SECTION A: BASIC LIFETIME FEATURES
# ==============================================================================
print("  → Creating basic lifetime features...")
# (No changes needed in this section)
account_df['sent_count'] = df_features['from_acct'].value_counts()
account_df['received_count'] = df_features['to_acct'].value_counts()
account_df['total_txn_count'] = account_df['sent_count'].fillna(0) + account_df['received_count'].fillna(0)
account_df['total_sent_twd'] = df_features.groupby('from_acct')['txn_amt_twd'].sum()
account_df['total_received_twd'] = df_features.groupby('to_acct')['txn_amt_twd'].sum()
account_df['in_out_ratio'] = account_df['total_sent_twd'] / (account_df['total_received_twd'] + 1e-6)
account_df['avg_sent'] = df_features.groupby('from_acct')['txn_amt_twd'].mean()
account_df['std_sent'] = df_features.groupby('from_acct')['txn_amt_twd'].std()
account_df['max_sent'] = df_features.groupby('from_acct')['txn_amt_twd'].max()
account_df['min_sent'] = df_features.groupby('from_acct')['txn_amt_twd'].min()
received_stats = df_features.groupby('to_acct')['txn_amt_twd'].agg(['mean', 'std', 'max', 'min'])
account_df['avg_received'] = received_stats['mean']
account_df['std_received'] = received_stats['std']
account_df['max_received'] = received_stats['max']
account_df['min_received'] = received_stats['min']
account_df['in_degree'] = df_features.groupby('to_acct')['from_acct'].nunique()
account_df['out_degree'] = df_features.groupby('from_acct')['to_acct'].nunique()


# ==============================================================================
# SECTION B: SELF-TRANSACTION FEATURES
# ==============================================================================
print("  → Creating self-transaction features...")
# (No changes needed in this section)
self_txn_data = df_features[df_features['is_self_txn_enc'] == 1]
account_df['self_txn_count'] = self_txn_data['from_acct'].value_counts()
account_df['self_txn_ratio'] = account_df['self_txn_count'] / (account_df['sent_count'] + 1e-6)
account_df['total_self_txn_amount'] = self_txn_data.groupby('from_acct')['txn_amt_twd'].sum()
account_df['avg_self_txn_amount'] = self_txn_data.groupby('from_acct')['txn_amt_twd'].mean()

# ==============================================================================
# SECTION C: TEMPORAL FEATURES
# ==============================================================================
print("  → Creating temporal features...")
# (No changes needed in this section, it uses the fixed 'txn_timestamp')
df_features['is_night_txn'] = (df_features['txn_hour'] >= 1) & (df_features['txn_hour'] <= 5)
night_txns_sent = df_features.groupby('from_acct')['is_night_txn'].sum()
night_txns_received = df_features.groupby('to_acct')['is_night_txn'].sum()
account_df['night_txn_count'] = night_txns_sent.add(night_txns_received, fill_value=0)
account_df['night_txn_ratio'] = account_df['night_txn_count'] / (account_df['total_txn_count'] + 1e-6)
df_features_sorted_out = df_features.sort_values(['from_acct', 'txn_timestamp'])
df_features_sorted_out['time_diff_seconds_out'] = df_features_sorted_out.groupby('from_acct')['txn_timestamp'].diff()
time_velocity_out = df_features_sorted_out.groupby('from_acct')['time_diff_seconds_out'].agg(['mean', 'std', 'min'])
account_df['avg_time_between_sent'] = time_velocity_out['mean'].fillna(dataset_duration_seconds)
account_df['std_time_between_sent'] = time_velocity_out['std'].fillna(0)
account_df['min_time_between_sent'] = time_velocity_out['min'].fillna(dataset_duration_seconds)
df_features_sorted_in = df_features.sort_values(['to_acct', 'txn_timestamp'])
df_features_sorted_in['time_diff_seconds_in'] = df_features_sorted_in.groupby('to_acct')['txn_timestamp'].diff()
time_velocity_in = df_features_sorted_in.groupby('to_acct')['time_diff_seconds_in'].agg(['mean', 'std', 'min'])
account_df['avg_time_between_received'] = time_velocity_in['mean'].fillna(dataset_duration_seconds)
account_df['std_time_between_received'] = time_velocity_in['std'].fillna(0)
account_df['min_time_between_received'] = time_velocity_in['min'].fillna(dataset_duration_seconds)

# ==============================================================================
# SECTION D: PASS-THROUGH BEHAVIOR
# ==============================================================================
print("  → Creating pass-through features...")
# (No changes needed, this uses integer 'txn_date')
sent_dates = df_features.groupby(['from_acct', 'txn_date']).size().reset_index()[['from_acct', 'txn_date']]
received_dates = df_features.groupby(['to_acct', 'txn_date']).size().reset_index()[['to_acct', 'txn_date']]
sent_dates.columns = ['acct', 'txn_date']
received_dates.columns = ['acct', 'txn_date']
pass_through_dates = pd.merge(sent_dates, received_dates, on=['acct', 'txn_date'])
account_df['pass_through_day_count'] = pass_through_dates.groupby('acct').size()

# ==============================================================================
# SECTION E: BEHAVIORAL FEATURES
# ==============================================================================
print("  → Creating behavioral features...")
# (No changes needed)
df_features['is_round_amount'] = (df_features['txn_amt_twd'] % 1000 == 0) | (df_features['txn_amt_twd'] % 10000 == 0)
round_amt_txns = df_features[df_features['is_round_amount']]
account_df['round_amount_count'] = round_amt_txns['from_acct'].value_counts()
account_df['round_amount_ratio'] = account_df['round_amount_count'] / (account_df['sent_count'] + 1e-6)

# ==============================================================================
# SECTION F: RECENCY FEATURES (7 days)
# ==============================================================================
print("  → Creating recency features (last 7 days)...")
# (No changes needed, this uses the fixed 'df_recent_7d')
account_df['sent_count_last_7d'] = df_recent_7d['from_acct'].value_counts()
account_df['received_count_last_7d'] = df_recent_7d['to_acct'].value_counts()
account_df['total_sent_twd_last_7d'] = df_recent_7d.groupby('from_acct')['txn_amt_twd'].sum()
account_df['total_received_twd_last_7d'] = df_recent_7d.groupby('to_acct')['txn_amt_twd'].sum()
account_df['unique_recipients_last_7d'] = df_recent_7d.groupby('from_acct')['to_acct'].nunique()
account_df['unique_senders_last_7d'] = df_recent_7d.groupby('to_acct')['from_acct'].nunique()
account_df['recent_sent_count_ratio_7d'] = account_df['sent_count_last_7d'] / (account_df['sent_count'] + 1e-6)
account_df['recent_sent_value_ratio_7d'] = account_df['total_sent_twd_last_7d'] / (account_df['total_sent_twd'] + 1e-6)

# ==============================================================================
# SECTION G: RECENCY FEATURES (30 days)
# ==============================================================================
print("  → Creating recency features (last 30 days)...")
# (No changes needed, this uses the fixed 'df_recent_30d')
account_df['sent_count_last_30d'] = df_recent_30d['from_acct'].value_counts()
account_df['received_count_last_30d'] = df_recent_30d['to_acct'].value_counts()
account_df['total_sent_twd_last_30d'] = df_recent_30d.groupby('from_acct')['txn_amt_twd'].sum()
account_df['total_received_twd_last_30d'] = df_recent_30d.groupby('to_acct')['txn_amt_twd'].sum()
account_df['recent_sent_count_ratio_30d'] = account_df['sent_count_last_30d'] / (account_df['sent_count'] + 1e-6)
account_df['recent_sent_value_ratio_30d'] = account_df['total_sent_twd_last_30d'] / (account_df['total_sent_twd'] + 1e-6)

# ==============================================================================
# SECTION H: CURRENCY FEATURES
# ==============================================================================
print("  → Creating currency features...")
# (No changes needed)
for col in currency_cols:
    sent_col_name = f'sent_{col}_total'
    received_col_name = f'received_{col}_total'
    total_col_name = f'total_{col}_total'
    account_df[sent_col_name] = df_features[df_features[col] == 1].groupby('from_acct')[col].sum()
    account_df[received_col_name] = df_features[df_features[col] == 1].groupby('to_acct')[col].sum()
    account_df[total_col_name] = account_df[sent_col_name].fillna(0) + account_df[received_col_name].fillna(0)
currency_feature_cols = [f'total_{col}_total' for col in currency_cols]
account_df['currency_diversity'] = (account_df[currency_feature_cols] > 0).sum(axis=1)
total_currency_txns = account_df[currency_feature_cols].sum(axis=1)
max_currency_txns = account_df[currency_feature_cols].max(axis=1)
account_df['dominant_currency_ratio'] = max_currency_txns / (total_currency_txns + 1e-6)
for col in currency_cols:
    recent_col_name = f'total_{col}_last_7d'
    sent_recent = df_recent_7d[df_recent_7d[col] == 1].groupby('from_acct')[col].sum()
    received_recent = df_recent_7d[df_recent_7d[col] == 1].groupby('to_acct')[col].sum()
    account_df[recent_col_name] = sent_recent.add(received_recent, fill_value=0)
currency_recent_cols = [f'total_{col}_last_7d' for col in currency_cols]
account_df['currency_diversity_last_7d'] = (account_df[currency_recent_cols] > 0).sum(axis=1)

# ==============================================================================
# SECTION I: BASIC INTERACTION FEATURES
# ==============================================================================
print("  → Creating basic interaction features...")
# (No changes needed, just be aware 114 and 121 are now hardcoded)
# (Ideally, you'd replace 121 with T and 114 with T-7)
account_df['txn_per_day'] = account_df['total_txn_count'] / (account_df['pass_through_day_count'].fillna(1) + 1)
account_df['txn_per_partner'] = account_df['sent_count'] / (account_df['out_degree'].fillna(0) + 1e-6)
account_df['activity_spike_7d'] = (
    account_df['sent_count_last_7d'].fillna(0) / 
    ((account_df['sent_count'].fillna(0) - account_df['sent_count_last_7d'].fillna(0)) / (T - 7) + 1e-6)
)
account_df['currency_diversity_change'] = (
    account_df['currency_diversity_last_7d'] - 
    (account_df['currency_diversity'] * 7 / T)
)


# ==============================================================================
# SECTION J: ADDITIONAL HIGH-IMPACT FEATURES
# ==============================================================================
print("  → Creating additional high-impact features...")

# 1. TEMPORAL CONCENTRATION FEATURES
print("     - (J-1) Temporal Concentration...")
df_features_all_sorted = pd.concat([
    df_features[['from_acct', 'txn_timestamp']].rename(columns={'from_acct': 'acct'}),
    df_features[['to_acct', 'txn_timestamp']].rename(columns={'to_acct': 'acct'})
]).sort_values(['acct', 'txn_timestamp'])
df_features_all_sorted['time_diff_all'] = df_features_all_sorted.groupby('acct')['txn_timestamp'].diff()
time_stats_all = df_features_all_sorted.groupby('acct')['time_diff_all'].agg(['mean', 'std', 'median'])
account_df['avg_time_between_all_txns'] = time_stats_all['mean'].fillna(dataset_duration_seconds)
account_df['std_time_between_all_txns'] = time_stats_all['std'].fillna(0)
account_df['median_time_between_all_txns'] = time_stats_all['median'].fillna(dataset_duration_seconds)
df_features_all_sorted['time_diff_hours'] = df_features_all_sorted['time_diff_all'] / 3600
burst_txns = df_features_all_sorted[df_features_all_sorted['time_diff_hours'] < 1]
account_df['burst_txn_count'] = burst_txns.groupby('acct').size()
account_df['burst_txn_ratio'] = account_df['burst_txn_count'] / (account_df['total_txn_count'] + 1e-6)

# 2. RECEIVE/SEND ASYMMETRY FEATURES
print("     - (J-2) Asymmetry...")
account_df['receive_send_ratio'] = account_df['received_count'] / (account_df['sent_count'] + 1e-6)
account_df['receive_dominance'] = (account_df['received_count'] - account_df['sent_count']) / (account_df['total_txn_count'] + 1e-6)
account_df['received_value_ratio'] = account_df['total_received_twd'] / (account_df['total_sent_twd'] + 1e-6)
account_df['recent_receive_ratio_7d'] = (
    account_df['received_count_last_7d'] / (account_df['received_count'] + 1e-6)
)
account_df['recent_send_ratio_7d'] = (
    account_df['sent_count_last_7d'] / (account_df['sent_count'] + 1e-6)
)
account_df['asymmetry_change_7d'] = account_df['recent_receive_ratio_7d'] - account_df['recent_send_ratio_7d']

# 3. PASS-THROUGH EFFICIENCY FEATURES
print("     - (J-3) Pass-Through Efficiency (this may be slow)...")
account_df['pass_through_ratio'] = account_df['pass_through_day_count'] / (account_df['total_txn_count'] + 1e-6)
sent_with_time = df_features[['from_acct', 'txn_date', 'txn_timestamp']].rename(
    columns={'from_acct': 'acct', 'txn_timestamp': 'send_time'}
)
received_with_time = df_features[['to_acct', 'txn_date', 'txn_timestamp']].rename(
    columns={'to_acct': 'acct', 'txn_timestamp': 'receive_time'}
)
pass_through_times = pd.merge(
    received_with_time, 
    sent_with_time, 
    on=['acct', 'txn_date'], 
    how='inner'
)
pass_through_times['time_to_forward'] = (
    pass_through_times['send_time'] - pass_through_times['receive_time']
) / 3600 # Already in seconds, just convert to hours
fast_forwards = pass_through_times[
    (pass_through_times['time_to_forward'] > 0) & 
    (pass_through_times['time_to_forward'] < 24)
]
account_df['fast_forward_count'] = fast_forwards.groupby('acct').size()
account_df['fast_forward_ratio'] = account_df['fast_forward_count'] / (account_df['received_count'] + 1e-6)
account_df['avg_time_to_forward_hours'] = fast_forwards.groupby('acct')['time_to_forward'].mean()

# 4. NETWORK CONCENTRATION FEATURES
print("     - (J-4) Network Gini (this is very slow)...")
def calculate_gini(values):
    if len(values) == 0 or values.sum() == 0:
        return 0
    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
recipient_txn_counts = df_features.groupby(['from_acct', 'to_acct']).size().reset_index(name='count')
recipient_gini = recipient_txn_counts.groupby('from_acct')['count'].apply(calculate_gini)
account_df['recipient_concentration_gini'] = recipient_gini
sender_txn_counts = df_features.groupby(['to_acct', 'from_acct']).size().reset_index(name='count')
sender_gini = sender_txn_counts.groupby('to_acct')['count'].apply(calculate_gini)
account_df['sender_concentration_gini'] = sender_gini

# 5. ACTIVITY PATTERN FEATURES (FIXED)
print("     - (J-5) Activity Patterns (this is very slow)...")
# FIX: Use modulo 7 on the integer day
df_features['day_of_week'] = df_features['txn_date'] % 7 
df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int) # Assuming day 5 & 6 are weekend
weekend_sent = df_features.groupby('from_acct')['is_weekend'].sum()
weekend_received = df_features.groupby('to_acct')['is_weekend'].sum()
account_df['weekend_txn_count'] = weekend_sent.add(weekend_received, fill_value=0)
account_df['weekend_txn_ratio'] = account_df['weekend_txn_count'] / (account_df['total_txn_count'] + 1e-6)
def calculate_entropy(hour_dist):
    if len(hour_dist) == 0:
        return 0
    hour_counts = hour_dist.value_counts(normalize=True)
    return -np.sum(hour_counts * np.log2(hour_counts + 1e-10))
sent_hour_entropy = df_features.groupby('from_acct')['txn_hour'].apply(calculate_entropy)
received_hour_entropy = df_features.groupby('to_acct')['txn_hour'].apply(calculate_entropy)
account_df['hour_entropy_sent'] = sent_hour_entropy
account_df['hour_entropy_received'] = received_hour_entropy
account_df['hour_entropy_avg'] = (
    account_df['hour_entropy_sent'].fillna(0) + 
    account_df['hour_entropy_received'].fillna(0)
) / 2

# 6. RECENCY MOMENTUM FEATURES (FIXED)
print("     - (J-6) Recency Momentum...")
# FIX: Use (T - 30) or (T-7) as the non-recent period length
non_recent_30d_period = T - 30
if non_recent_30d_period <= 0: non_recent_30d_period = 1 # Avoid division by zero
account_df['activity_acceleration_30d_vs_7d'] = (
    (account_df['sent_count_last_7d'].fillna(0) / 7) / 
    ((account_df['sent_count_last_30d'].fillna(0) - account_df['sent_count_last_7d'].fillna(0)) / (30 - 7) + 1e-6)
)
account_df['network_expansion_7d'] = (
    account_df['unique_recipients_last_7d'].fillna(0) / 
    (account_df['out_degree'].fillna(0) + 1e-6)
)

# 7. ADVANCED INTERACTION FEATURES
print("     - (J-7) Advanced Interactions...")
account_df['pass_through_per_sender'] = (
    account_df['pass_through_day_count'] / 
    (account_df['unique_senders_last_7d'].fillna(0) + 1e-6)
)
account_df['received_per_sender'] = (
    account_df['received_count'] / 
    (account_df['in_degree'].fillna(0) + 1e-6)
)

# ==============================================================================
# 6. DEFINE TARGET AND CLEAN FINAL DATASET
# ==============================================================================
print("\n--- 6. Defining Target Variable and Final Cleaning ---")

account_df['is_alert'] = account_df.index.isin(alerted_accounts_set).astype(int)

account_df.fillna(0, inplace=True)
account_df.replace([np.inf, -np.inf], 0, inplace=True)

print(f"  ✓ Final dataset shape: {account_df.shape}")
print(f"  ✓ Number of features: {account_df.shape[1] - 1}")
print(f"  ✓ Alert accounts: {account_df['is_alert'].sum():,} ({account_df['is_alert'].mean()*100:.3f}%)")