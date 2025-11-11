import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import gc

#1. Load Data
print("Loading data...")
DATA_PATH = '/kaggle/input/40-v3-1-zip/ªìÁÉ¸ê®Æ/' 

try:
    df_txn = pd.read_csv(f'{DATA_PATH}acct_transaction.csv')
    df_alert = pd.read_csv(f'{DATA_PATH}acct_alert.csv')
    df_predict = pd.read_csv(f'{DATA_PATH}acct_predict.csv')
except FileNotFoundError:
    print(f"Error: Could not find files at {DATA_PATH}")
    print("Trying local path '/kaggle/working/'...")
    df_alert = pd.read_csv('acct_alert.csv')
    df_predict = pd.read_csv('acct_predict.csv')
    df_txn = pd.read_csv('acct_transaction.csv')


print("Data loaded successfully.")
print(f"Transaction data shape: {df_txn.shape}")
print(f"Alert data shape: {df_alert.shape}")
print(f"Prediction data shape: {df_predict.shape}")


#2. Feature Engineering
print("\nStarting feature engineering...")

all_accts = set(df_alert['acct']).union(set(df_predict['acct']))
print(f"Total unique accounts to analyze: {len(all_accts)}")

#txn_time to txn_hours
df_txn['txn_hour'] = pd.to_datetime(df_txn['txn_time'], format='%H:%M:%S', errors='coerce').dt.hour
df_txn['txn_hour'] = df_txn['txn_hour'].fillna(0).astype(int)
df_txn['is_night_txn'] = ((df_txn['txn_hour'] >= 0) & (df_txn['txn_hour'] < 5)).astype(int)
df_txn['datetime'] = pd.to_datetime(
    df_txn['txn_date'].astype(str) + df_txn['txn_hour'].astype(str).str.zfill(2), 
    format='%j%H', 
    errors='coerce'
)

df_txn_filtered = df_txn[
    (df_txn['from_acct'].isin(all_accts)) | 
    (df_txn['to_acct'].isin(all_accts))
].copy()

del df_txn
gc.collect()
print(f"Filtered transaction shape: {df_txn_filtered.shape}")


print("Calculating debit (send) features...")
df_debit = df_txn_filtered[df_txn_filtered['from_acct'].isin(all_accts)]
debit_features = df_debit.groupby('from_acct').agg(
    send_amt_mean=('txn_amt', 'mean'),
    send_amt_std=('txn_amt', 'std'),
    send_amt_max=('txn_amt', 'max'),
    send_count=('txn_amt', 'count'),
    unique_recipients=('to_acct', 'nunique'),
    send_channel_diversity=('channel_type', 'nunique'),
    send_night_ratio=('is_night_txn', 'mean')
)
debit_features['send_amt_cv'] = debit_features['send_amt_std'] / debit_features['send_amt_mean']
debit_features.index.name = 'acct'


print("Calculating credit (receive) features...")
df_credit = df_txn_filtered[df_txn_filtered['to_acct'].isin(all_accts)]
credit_features = df_credit.groupby('to_acct').agg(
    recv_amt_mean=('txn_amt', 'mean'),
    recv_amt_std=('txn_amt', 'std'),
    recv_amt_max=('txn_amt', 'max'),
    recv_count=('txn_amt', 'count'),
    unique_senders=('from_acct', 'nunique'),
    recv_channel_diversity=('channel_type', 'nunique'),
    recv_night_ratio=('is_night_txn', 'mean')
)
credit_features['recv_amt_cv'] = credit_features['recv_amt_std'] / credit_features['recv_amt_mean']
credit_features.index.name = 'acct'

print("Calculating time-based features...")
df_debit_time = df_txn_filtered.loc[df_txn_filtered['from_acct'].isin(all_accts), ['from_acct', 'datetime']]
df_debit_time.rename(columns={'from_acct': 'acct'}, inplace=True)
df_credit_time = df_txn_filtered.loc[df_txn_filtered['to_acct'].isin(all_accts), ['to_acct', 'datetime']]
df_credit_time.rename(columns={'to_acct': 'acct'}, inplace=True)

df_all_txns_time = pd.concat([df_debit_time, df_credit_time]).sort_values(by=['acct', 'datetime'])
df_all_txns_time['datetime'] = df_all_txns_time['datetime'].ffill().bfill() # Corrected ffill
df_all_txns_time['time_diff_seconds'] = df_all_txns_time.groupby('acct')['datetime'].diff().dt.total_seconds()
time_features = df_all_txns_time.groupby('acct')['time_diff_seconds'].agg(
    time_diff_mean='mean',
    time_diff_std='std',
    time_diff_max='max',
    time_diff_min='min'
)
time_features.index.name = 'acct'

del df_txn_filtered, df_debit, df_credit, df_debit_time, df_credit_time, df_all_txns_time
gc.collect()


print("Combining all features...")
all_accts_df = pd.DataFrame(index=list(all_accts)) # Corrected list()
all_accts_df.index.name = 'acct'

features = all_accts_df.join(debit_features).join(credit_features).join(time_features)
features = features.fillna(0)
features.replace([np.inf, -np.inf], 0, inplace=True)

print("Feature engineering complete.")
print(f"Feature matrix shape: {features.shape}")


#3. Prepare Training and Test Data
print("\nPreparing training and test sets with Time-Series Split...")

df_alert['label'] = 1
train_pos = df_alert.join(features, on='acct', how='inner')

alert_accts = set(df_alert['acct'])
negative_sample_accts = [acct for acct in features.index if acct not in alert_accts]
train_neg = features.loc[negative_sample_accts].copy()
train_neg['label'] = 0
train_neg.reset_index(inplace=True) 

#TIME SPLIT
date_threshold = 90 
pos_train = train_pos[train_pos['event_date'] < date_threshold]
pos_val = train_pos[train_pos['event_date'] >= date_threshold]

neg_train, neg_val = train_test_split(train_neg, test_size=0.25, random_state=42)

df_train_set = pd.concat([pos_train, neg_train], ignore_index=True)
df_val_set = pd.concat([pos_val, neg_val], ignore_index=True)

df_train_set = df_train_set.sample(frac=1, random_state=42)
df_val_set = df_val_set.sample(frac=1, random_state=42)

feature_cols = [col for col in features.columns if col not in ['acct', 'label', 'event_date']]

X_train = df_train_set[feature_cols]
y_train = df_train_set['label']
X_val = df_val_set[feature_cols]
y_val = df_val_set['label']

X_submission = df_predict.join(features, on='acct', how='left').fillna(0)
X_test = X_submission[feature_cols]

print(f"Time-Split Train shape: {X_train.shape}")
print(f"Time-Split Valid shape: {X_val.shape}")
print(f"Class distribution in Validation set:\n{y_val.value_counts()}")


#4. Model Training and F1-Score Optimization

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"\nNew Scale Pos Weight: {scale_pos_weight:.2f}")

print("Training LightGBM model on time-split data...")
lgb_clf = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    n_jobs=-1,
    colsample_bytree=0.8,
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    scale_pos_weight=scale_pos_weight
)

lgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='f1',
    callbacks=[lgb.early_stopping(100, verbose=True)]
)

#Find the best F1 threshold on the validation set
print("\nOptimizing F1-score threshold on (time-split) validation data...")
y_pred_probs = lgb_clf.predict_proba(X_val)[:, 1]

#Check if model produced any non-zero probabilities
if len(np.unique(y_pred_probs)) > 1:
    thresholds = np.arange(0.01, 0.5, 0.01)
    f1_scores = [f1_score(y_val, (y_pred_probs >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)
else:
    print("Warning: Model predicted all 0s or 1s for validation set. Defaulting threshold to 0.5")
    best_threshold = 0.5
    best_f1 = f1_score(y_val, (y_pred_probs >= 0.5).astype(int))

print(f"Best threshold: {best_threshold:.4f}")
print(f"Best F1-score on validation set: {best_f1:.4f}")

#Display full classification report
y_pred_best_thresh = (y_pred_probs >= best_threshold).astype(int)
print("\nClassification Report on Validation Set (with optimized threshold):")
print(classification_report(y_val, y_pred_best_thresh))

if best_f1 < 0.5:
    print(f"NOTE: Achieved F1-score {best_f1:.4f} is below the target of 0.50000 (This is now an 'honest' score)")
else:
    print("SUCCESS: F1-score target achieved on validation set.")


#5. Generate Final Predictions
print("\nGenerating predictions for the submission file...")

#1. Get best iteration (from the model trained in Step 4)
best_n_estimators = lgb_clf.best_iteration_

if best_n_estimators is None or best_n_estimators <= 0:
    print("Warning: Best iteration not found. Using default 100.")
    best_n_estimators = 100 
else:
    print(f"Using optimal {best_n_estimators} estimators for final model.")

#2. Get params
final_params = lgb_clf.get_params()
final_params['n_estimators'] = best_n_estimators
final_params.pop('early_stopping_round', None) 

#3.
#Re-define X and y as the *full* dataset for retraining
#Combine the time-split train and validation sets back together
print("Re-combining full dataset for final training...")
X = pd.concat([X_train, X_val], ignore_index=True)
y = pd.concat([y_train, y_val], ignore_index=True)

print(f"Full dataset shape for retraining: {X.shape}")

#4. Retrain the model on the FULL dataset
print("Retraining model on full dataset with optimal estimators...")
lgb_clf_full = lgb.LGBMClassifier(**final_params)
lgb_clf_full.fit(X, y)

#5. Predict
final_probs = lgb_clf_full.predict_proba(X_test)[:, 1]
final_predictions = (final_probs >= best_threshold).astype(int) 

#6. Output Submission File
submission_df = pd.DataFrame({'acct': X_submission['acct'], 'label': final_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully.")
print(f"Predicted alert count: {submission_df['label'].sum()}")
print(submission_df.head())