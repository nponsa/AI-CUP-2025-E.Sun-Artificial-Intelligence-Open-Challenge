import pandas as pd

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

# Convert time string to numerical seconds from midnight (more efficient method)
time_dt = pd.to_datetime(df1['txn_time'], format='%H:%M:%S')
df1['txn_time_seconds'] = time_dt.dt.hour * 3600 + time_dt.dt.minute * 60 + time_dt.dt.second

df1['channel_type'] = df1['channel_type'].replace('UNK', -1).astype(int)

currency_dummies = pd.get_dummies(df1['currency_type'], prefix='currency', dtype=int)
df1 = pd.concat([df1, currency_dummies], axis=1)

# --- Clean up original columns after transformation ---
df1.drop(columns=['is_self_txn', 'txn_amt', 'currency_type', 'txn_time'], inplace=True)
print(" Preprocessing of transaction data complete.")


# ==============================================================================
# 2. FEATURE ENGINEERING (Account-Level DataFrame)
# ==============================================================================
print("\n--- 2. Building Account-Level Feature Set ---")


# Master DataFrame with one row per account
all_accts = pd.concat([df1['from_acct'], df1['to_acct']]).unique()
account_df = pd.DataFrame(index=all_accts)
account_df.index.name = 'acct'

# --- Feature 1: Pass-Through Day Count ---
sent_days = df1.groupby(['from_acct', 'txn_date']).size().reset_index()
received_days = df1.groupby(['to_acct', 'txn_date']).size().reset_index()
pass_through_days = pd.merge(sent_days, received_days, left_on=['from_acct', 'txn_date'], right_on=['to_acct', 'txn_date'])
pass_through_counts = pass_through_days.groupby('from_acct').size()
account_df['pass_through_day_count'] = account_df.index.map(pass_through_counts)

# --- Feature 2: Average Transaction Amount ---
avg_sent_amt = df1.groupby('from_acct')['txn_amt_twd'].mean()
avg_received_amt = df1.groupby('to_acct')['txn_amt_twd'].mean()
account_df['avg_sent'] = account_df.index.map(avg_sent_amt)
account_df['avg_received'] = account_df.index.map(avg_received_amt)
account_df['avg_txn_amount'] = (account_df['avg_sent'].fillna(0) + account_df['avg_received'].fillna(0)) / 2
account_df.drop(columns=['avg_sent', 'avg_received'], inplace=True)


# --- Feature 3: Financial Totals and In/Out Ratio ---
account_df['total_sent_twd'] = df1.groupby('from_acct')['txn_amt_twd'].sum()
account_df['total_received_twd'] = df1.groupby('to_acct')['txn_amt_twd'].sum()
account_df['in_out_ratio'] = account_df['total_sent_twd'] / (account_df['total_received_twd'] + 1e-6)

# --- Feature 4: Network Centrality ---

account_df['in_degree'] = df1.groupby('to_acct')['from_acct'].nunique()
account_df['out_degree'] = df1.groupby('from_acct')['to_acct'].nunique()

# --- Feature 5: Transaction Counts ---
account_df['sent_count'] = df1['from_acct'].value_counts()
account_df['received_count'] = df1['to_acct'].value_counts()
print(" Feature engineering complete.")


# ==============================================================================
# 3. DEFINE TARGET AND CLEAN FINAL DATASET
# ==============================================================================
print("\n--- 3. Defining Target Variable and Final Cleaning ---")
officially_flagged_set = set(df3['acct'])
account_df['is_alert'] = account_df.index.isin(officially_flagged_set).astype(int)

# Fill any remaining NaNs that might have resulted from feature creation
account_df.fillna(0, inplace=True)
print("Target variable created and final dataset cleaned.")


# ==============================================================================
# 4. SAVE THE FINAL DATASET FOR THE TEAM
# ==============================================================================
print("\n--- 4. Saving the final account-level data for the team ---")

output_filename = "processed_account_features.csv"
account_df.to_csv(output_filename)

print(f"Successfully saved the final dataset to '{output_filename}'.")
print("This file is now ready.")

# ==============================================================================
# 5. CHECKS
# ==============================================================================

