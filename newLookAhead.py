import pandas as pd
import numpy as np

# ==============================================================================
# 1. LOAD AND PREPROCESS DATA
# ==============================================================================
print("--- 1. Loading and Preprocessing Data ---")

# Load the raw data
filename1 = "testData/firstTestData/acct_transaction.csv"
filename3 = "testData/firstTestData/acct_alert.csv"
df1 = pd.read_csv(filename1)
df3 = pd.read_csv(filename3)

# --- Preprocessing steps ---
self_map = {"Y": 1, "N": 0, "UNK": -1}
df1['is_self_txn_enc'] = df1['is_self_txn'].map(self_map)

exchange_rates = {
    'TWD': 1.0, 'USD': 32.42, 'JPY': 0.20, 'AUD': 21.55, 'CNY': 4.46, 'EUR': 34.71,
    'SEK': 2.91, 'GBP': 41.23, 'HKD': 4.15, 'THB': 0.94, 'CAD': 23.51, 'NZD': 20.15,
    'CHF': 35.60, 'SGD': 24.08, 'ZAR': 1.78, 'MXN': 1.85
}
df1['txn_amt_twd'] = df1['txn_amt'] * df1['currency_type'].map(exchange_rates)

# Convert time string to numerical seconds from midnight
time_dt = pd.to_datetime(df1['txn_time'], format='%H:%M:%S')
df1['txn_time_seconds'] = time_dt.dt.hour * 3600 + time_dt.dt.minute * 60 + time_dt.dt.second

df1['channel_type'] = df1['channel_type'].replace('UNK', -1).astype(int)

# Create currency dummies BEFORE dropping currency_type
currency_dummies = pd.get_dummies(df1['currency_type'], prefix='currency', dtype=int)
df1 = pd.concat([df1, currency_dummies], axis=1)

# Get list of currency DUMMY columns only (exclude 'currency_type' string column)
currency_cols = [col for col in currency_dummies.columns]
print(f"Currency dummy columns created: {currency_cols}")

print("Preprocessing of transaction data complete.")

# ==============================================================================
# 2. TIME-AWARE FEATURE ENGINEERING (Preventing Lookahead Bias)
# ==============================================================================
print("\n--- 2. Building Time-Aware Daily Feature Set with Currency Features ---")

# CRITICAL: Sort all transactions chronologically
df1.sort_values(by=['txn_date', 'txn_time_seconds'], inplace=True)

# --- Step 2a: Create daily aggregations for SENT transactions ---
agg_dict_sent = {
    'txn_amt_twd': ['sum', 'mean', 'count', 'std'],
    'to_acct': 'nunique',  # unique recipients
    'is_self_txn_enc': 'mean',  # proportion of self transactions
    'channel_type': 'mean'  # average channel type
}

# Add currency columns - sum gives us daily transaction count per currency
for col in currency_cols:
    agg_dict_sent[col] = 'sum'

daily_sent = df1.groupby(['from_acct', 'txn_date']).agg(agg_dict_sent).reset_index()

# Flatten multi-level column names
daily_sent.columns = ['acct', 'txn_date', 
                      'daily_sent_sum', 'daily_sent_mean', 'daily_sent_count', 'daily_sent_std',
                      'daily_unique_recipients', 'daily_self_txn_ratio', 'daily_channel_mean'] + \
                     [f'daily_sent_{col}' for col in currency_cols]

# --- Step 2b: Create daily aggregations for RECEIVED transactions ---
agg_dict_received = {
    'txn_amt_twd': ['sum', 'mean', 'count', 'std'],
    'from_acct': 'nunique'  # unique senders
}

# Add currency columns for received transactions
for col in currency_cols:
    agg_dict_received[col] = 'sum'

daily_received = df1.groupby(['to_acct', 'txn_date']).agg(agg_dict_received).reset_index()

# Flatten column names
daily_received.columns = ['acct', 'txn_date',
                          'daily_received_sum', 'daily_received_mean', 'daily_received_count', 
                          'daily_received_std', 'daily_unique_senders'] + \
                         [f'daily_received_{col}' for col in currency_cols]

# Merge daily sending and receiving activity
daily_activity = pd.merge(daily_sent, daily_received, on=['acct', 'txn_date'], how='outer').fillna(0)

# --- Step 2c: Create daily-level features ---
# Pass-through day flag (both sent AND received on same day)
daily_activity['is_pass_through_day'] = (
    (daily_activity['daily_sent_count'] > 0) & 
    (daily_activity['daily_received_count'] > 0)
).astype(int)

# Daily velocity metrics
daily_activity['daily_total_txn_count'] = (
    daily_activity['daily_sent_count'] + daily_activity['daily_received_count']
)

daily_activity['daily_net_flow'] = (
    daily_activity['daily_received_sum'] - daily_activity['daily_sent_sum']
)

# Currency diversity (how many different currencies used today)
sent_currency_cols = [f'daily_sent_{col}' for col in currency_cols]
received_currency_cols = [f'daily_received_{col}' for col in currency_cols]

daily_activity['daily_currency_diversity'] = (
    (daily_activity[sent_currency_cols + received_currency_cols] > 0).sum(axis=1)
)

# --- Step 2d: Calculate CUMULATIVE (expanding window) features ---
# This is the KEY to preventing lookahead bias!
daily_activity.sort_values(by=['acct', 'txn_date'], inplace=True)

# Cumulative financial totals
daily_activity['total_sent_so_far'] = daily_activity.groupby('acct')['daily_sent_sum'].cumsum()
daily_activity['total_received_so_far'] = daily_activity.groupby('acct')['daily_received_sum'].cumsum()

# Cumulative transaction counts
daily_activity['sent_count_so_far'] = daily_activity.groupby('acct')['daily_sent_count'].cumsum()
daily_activity['received_count_so_far'] = daily_activity.groupby('acct')['daily_received_count'].cumsum()

# Cumulative pass-through days
daily_activity['pass_through_days_so_far'] = daily_activity.groupby('acct')['is_pass_through_day'].cumsum()

# Cumulative network metrics
daily_activity['unique_recipients_so_far'] = daily_activity.groupby('acct')['daily_unique_recipients'].cumsum()
daily_activity['unique_senders_so_far'] = daily_activity.groupby('acct')['daily_unique_senders'].cumsum()

# CUMULATIVE CURRENCY FEATURES - This is powerful for fraud detection!
for col in currency_cols:
    # Total sent transactions per currency so far
    daily_activity[f'sent_{col}_so_far'] = (
        daily_activity.groupby('acct')[f'daily_sent_{col}'].cumsum()
    )
    # Total received transactions per currency so far
    daily_activity[f'received_{col}_so_far'] = (
        daily_activity.groupby('acct')[f'daily_received_{col}'].cumsum()
    )
    # Total transactions (sent + received) per currency so far
    daily_activity[f'total_{col}_so_far'] = (
        daily_activity[f'sent_{col}_so_far'] + daily_activity[f'received_{col}_so_far']
    )

# Number of unique currencies used so far
currency_so_far_cols = [f'total_{col}_so_far' for col in currency_cols]
daily_activity['currency_diversity_so_far'] = (
    (daily_activity[currency_so_far_cols] > 0).sum(axis=1)
)

# --- Step 2e: Calculate expanding ratios and averages ---
# In/Out ratio (with small epsilon to avoid division by zero)
daily_activity['in_out_ratio_so_far'] = (
    daily_activity['total_sent_so_far'] / (daily_activity['total_received_so_far'] + 1e-6)
)

# Average transaction amount so far
total_txn_amount_so_far = (
    daily_activity['total_sent_so_far'] + daily_activity['total_received_so_far']
)
total_txn_count_so_far = (
    daily_activity['sent_count_so_far'] + daily_activity['received_count_so_far']
)
daily_activity['avg_txn_amount_so_far'] = total_txn_amount_so_far / (total_txn_count_so_far + 1e-6)

# Network centrality metrics
daily_activity['network_degree_so_far'] = (
    daily_activity['unique_recipients_so_far'] + daily_activity['unique_senders_so_far']
)

# Pass-through ratio
daily_activity['days_active_so_far'] = daily_activity.groupby('acct').cumcount() + 1
daily_activity['pass_through_ratio_so_far'] = (
    daily_activity['pass_through_days_so_far'] / daily_activity['days_active_so_far']
)

# Currency concentration - is the account using mostly one currency or diversified?
# Calculate percentage of transactions in dominant currency
total_currency_txns = daily_activity[currency_so_far_cols].sum(axis=1)
max_currency_txns = daily_activity[currency_so_far_cols].max(axis=1)
daily_activity['dominant_currency_ratio'] = max_currency_txns / (total_currency_txns + 1e-6)

# --- Step 2f: Lag features (yesterday's values) ---
lag_cols = ['daily_sent_sum', 'daily_received_sum', 'daily_sent_count', 
            'daily_received_count', 'is_pass_through_day', 'daily_currency_diversity']

for col in lag_cols:
    daily_activity[f'{col}_lag1'] = daily_activity.groupby('acct')[col].shift(1).fillna(0)

# --- Step 2g: Rolling window features (past 3 days, past 7 days) ---
# 3-day rolling averages (excluding current day)
daily_activity['avg_sent_3d'] = (
    daily_activity.groupby('acct')['daily_sent_sum']
    .shift(1)
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

daily_activity['avg_received_3d'] = (
    daily_activity.groupby('acct')['daily_received_sum']
    .shift(1)
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# 7-day rolling averages (excluding current day)
daily_activity['avg_sent_7d'] = (
    daily_activity.groupby('acct')['daily_sent_sum']
    .shift(1)
    .rolling(window=7, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

daily_activity['avg_received_7d'] = (
    daily_activity.groupby('acct')['daily_received_sum']
    .shift(1)
    .rolling(window=7, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Currency diversity trend (is it increasing?)
daily_activity['currency_diversity_7d'] = (
    daily_activity.groupby('acct')['daily_currency_diversity']
    .shift(1)
    .rolling(window=7, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Fill NaN values
daily_activity.fillna(0, inplace=True)

print(" Time-aware feature engineering with currency features complete.")

# ==============================================================================
# 3. DEFINE TARGET AND CREATE FINAL DATASET
# ==============================================================================
print("\n--- 3. Defining Target Variable and Final Cleaning ---")

# Prepare alert data
df3.rename(columns={'event_date': 'txn_date'}, inplace=True)
df3['is_alert'] = 1

# Merge the target variable onto our daily feature set
final_df = pd.merge(daily_activity, df3[['acct', 'txn_date', 'is_alert']], 
                    on=['acct', 'txn_date'], how='left')

# Fill NaN in 'is_alert' with 0 (days where no alert occurred)
final_df['is_alert'].fillna(0, inplace=True)
final_df['is_alert'] = final_df['is_alert'].astype(int)

# Select final feature columns for modeling
feature_columns = [
    'acct', 'txn_date', 
    # Cumulative financial features
    'total_sent_so_far', 'total_received_so_far',
    'sent_count_so_far', 'received_count_so_far',
    'pass_through_days_so_far', 'pass_through_ratio_so_far',
    'in_out_ratio_so_far', 'avg_txn_amount_so_far',
    'unique_recipients_so_far', 'unique_senders_so_far', 'network_degree_so_far',
    # Currency features
    'currency_diversity_so_far', 'dominant_currency_ratio',
    # Lag features
    'daily_sent_sum_lag1', 'daily_received_sum_lag1',
    'daily_sent_count_lag1', 'daily_received_count_lag1',
    'is_pass_through_day_lag1', 'daily_currency_diversity_lag1',
    # Rolling window features
    'avg_sent_3d', 'avg_received_3d',
    'avg_sent_7d', 'avg_received_7d',
    'currency_diversity_7d',
    # Target
    'is_alert'
]

# Add cumulative currency count columns
for col in currency_cols:
    feature_columns.extend([f'sent_{col}_so_far', f'received_{col}_so_far', f'total_{col}_so_far'])

final_df = final_df[feature_columns]

print("Target variable created and final dataset cleaned.")
print(f"Dataset shape: {final_df.shape}")
print(f"Total features: {len(feature_columns) - 3}")  # Exclude acct, txn_date, is_alert
print(f"Total account-day observations: {len(final_df)}")
print(f"Number of alert days: {final_df['is_alert'].sum()}")
print(f"Alert rate: {final_df['is_alert'].mean():.4%}")

# Display feature importance preview
print("\n Currency Usage Summary:")
currency_usage = final_df[[col for col in final_df.columns if 'total_currency' in col]].sum()
print(currency_usage[currency_usage > 0].sort_values(ascending=False))

# ==============================================================================
# 4. SAVE THE FINAL DATASET FOR THE TEAM
# ==============================================================================
print("\n--- 4. Saving the final time-aware dataset ---")

output_filename = "processed_daily_features_with_currency.csv"
final_df.to_csv(output_filename, index=False)

print(f"Successfully saved to '{output_filename}'")
print("\n This dataset includes powerful currency features!")
print("\nKey currency features added:")
print("   sent_currency_XXX_so_far: Cumulative sent transactions per currency")
print("   received_currency_XXX_so_far: Cumulative received transactions per currency")
print("   total_currency_XXX_so_far: Total transactions per currency")
print("   currency_diversity_so_far: Number of unique currencies used")
print("   dominant_currency_ratio: Concentration in primary currency")
print("   daily_currency_diversity: Daily currency variety")
print("\n Why currency features matter for fraud detection:")
print("  • Sudden currency changes can indicate account takeover")
print("  • High currency diversity may signal money laundering")
print("  • Unusual currency patterns for account type (e.g., domestic account using exotic currencies)")
