# 1. SETUP AND DATA LOADING
# ==================================
import numpy as np
import pandas as pd
import warnings
import os
import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load all datasets
print("Loading data...")
train_df_orig = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df_orig = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
org_df = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')
org_df_ = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')
print("Data loaded successfully.")

# Combine training data
train = pd.concat([train_df_orig, org_df, org_df_], ignore_index=True, axis=0)

# Separate test ID and drop from test set
test_id = test_df_orig['id']
test = test_df_orig.drop(['id'], axis=1)

# 2. DATA PREPROCESSING
# ==================================
print("Starting data preprocessing...")

# Map categorical string values to numbers
train['Stage_fear'] = train['Stage_fear'].map({'Yes': 1, 'No': 0})
test['Stage_fear'] = test['Stage_fear'].map({'Yes': 1, 'No': 0})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Map target variable 'Personality' to numbers
train['Personality'] = train['Personality'].map({'Extrovert': 0, 'Introvert': 1})

# Define feature columns and target variable
feature_cols = test.columns
X = train[feature_cols]
y = train['Personality']

# Impute missing values and scale features
scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=5)

# Fit on training data and transform both train and test
X_scaled = scaler.fit_transform(X)
X_imputed = imputer.fit_transform(X_scaled)

test_scaled = scaler.transform(test)
test_imputed = imputer.transform(test_scaled)

# Convert back to DataFrame
X_processed = pd.DataFrame(X_imputed, columns=feature_cols)
test_processed = pd.DataFrame(test_imputed, columns=feature_cols)
print("Preprocessing complete.")


# 3. HYPERPARAMETER TUNING WITH OPTUNA 🚀
# ==========================================

# Split data for tuning validation
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, stratify=y, random_state=42
)

def objective(trial):
    """Optuna objective function with a reduced search space for faster tuning."""
    # -- Hyperparameters for XGBoost --
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 200, 800), # Reduced range
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 7),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
    }

    # -- Hyperparameters for CatBoost --
    cat_params = {
        'iterations': trial.suggest_int('cat_iterations', 200, 800), # Reduced range
        'depth': trial.suggest_int('cat_depth', 4, 8),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1, 10),
    }

    # -- Hyperparameters for LightGBM --
    lgbm_params = {
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 200, 800), # Reduced range
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('lgbm_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0),
    }
    
    # Initialize models
    xgb = XGBClassifier(**xgb_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    cat = CatBoostClassifier(**cat_params, random_seed=42, verbose=0)
    lgbm = LGBMClassifier(**lgbm_params, random_state=42)

    # Create the voting ensemble
    ensemble = VotingClassifier(estimators=[('xgb', xgb), ('cat', cat), ('lgbm', lgbm)], voting='soft')
    
    # Train and evaluate
    ensemble.fit(X_train, y_train)
    val_preds = ensemble.predict(X_val)
    accuracy = accuracy_score(y_val, val_preds)
    
    return accuracy

print("Starting Optuna hyperparameter search...")
study = optuna.create_study(direction='maximize')
# Reduced number of trials for a faster run
study.optimize(objective, n_trials=15) 

print(f"\nBest trial accuracy: {study.best_value:.4f}")
print("Best hyperparameters found:")
print(study.best_params)


# 4. FINAL MODEL TRAINING AND SUBMISSION
# ========================================
print("\nTraining final model on all data with best hyperparameters...")

# Extract the best parameters for each model
best_params = study.best_params
final_xgb_params = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}
final_cat_params = {k.replace('cat_', ''): v for k, v in best_params.items() if k.startswith('cat_')}
final_lgbm_params = {k.replace('lgbm_', ''): v for k, v in best_params.items() if k.startswith('lgbm_')}

# Initialize models with the best found hyperparameters
final_xgb = XGBClassifier(**final_xgb_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
final_cat = CatBoostClassifier(**final_cat_params, random_seed=42, verbose=0)
final_lgbm = LGBMClassifier(**final_lgbm_params, random_state=42)

# Create the final voting ensemble
final_ensemble = VotingClassifier(
    estimators=[('xgb', final_xgb), ('cat', final_cat), ('lgbm', final_lgbm)],
    voting='soft'
)

# Train the final model on the ENTIRE processed dataset
final_ensemble.fit(X_processed, y)

# Make predictions
test_preds = final_ensemble.predict(test_processed)

# Create and save the submission file
submission = pd.DataFrame({'id': test_id, 'Personality': test_preds})
submission['Personality'] = submission['Personality'].map({1: 'Introvert', 0: 'Extrovert'})
submission.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
print(submission.head())
Loading data...
Data loaded successfully.
Starting data preprocessing...
[I 2025-08-01 07:42:00,609] A new study created in memory with name: no-name-af3a8d68-e63a-4e06-9d09-bcabff4aa633
Preprocessing complete.
Starting Optuna hyperparameter search...
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.001507 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:04,383] Trial 0 finished with value: 0.9617677286742035 and parameters: {'xgb_n_estimators': 773, 'xgb_max_depth': 3, 'xgb_learning_rate': 0.12674350254148473, 'xgb_subsample': 0.6588473954621925, 'xgb_colsample_bytree': 0.9648386727045767, 'cat_iterations': 556, 'cat_depth': 4, 'cat_learning_rate': 0.05667464867414144, 'cat_l2_leaf_reg': 4.019072644382492, 'lgbm_n_estimators': 434, 'lgbm_num_leaves': 32, 'lgbm_learning_rate': 0.05143518357922243, 'lgbm_subsample': 0.6337716698741362, 'lgbm_colsample_bytree': 0.9361562880833824}. Best is trial 0 with value: 0.9617677286742035.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000205 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:09,550] Trial 1 finished with value: 0.9621788283658788 and parameters: {'xgb_n_estimators': 255, 'xgb_max_depth': 4, 'xgb_learning_rate': 0.02300988115479625, 'xgb_subsample': 0.6019618216464371, 'xgb_colsample_bytree': 0.6463509564195872, 'cat_iterations': 741, 'cat_depth': 5, 'cat_learning_rate': 0.010638622000567607, 'cat_l2_leaf_reg': 5.403728482514485, 'lgbm_n_estimators': 684, 'lgbm_num_leaves': 52, 'lgbm_learning_rate': 0.1464261264895688, 'lgbm_subsample': 0.7501657791011547, 'lgbm_colsample_bytree': 0.6096348951640888}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000850 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:13,969] Trial 2 finished with value: 0.9611510791366906 and parameters: {'xgb_n_estimators': 258, 'xgb_max_depth': 5, 'xgb_learning_rate': 0.030779880628711766, 'xgb_subsample': 0.6439611142498819, 'xgb_colsample_bytree': 0.725716315505115, 'cat_iterations': 280, 'cat_depth': 8, 'cat_learning_rate': 0.1556600075810285, 'cat_l2_leaf_reg': 6.297093430595454, 'lgbm_n_estimators': 712, 'lgbm_num_leaves': 76, 'lgbm_learning_rate': 0.126671136966152, 'lgbm_subsample': 0.6260039206703983, 'lgbm_colsample_bytree': 0.6786688662334721}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000816 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:18,423] Trial 3 finished with value: 0.9621788283658788 and parameters: {'xgb_n_estimators': 642, 'xgb_max_depth': 7, 'xgb_learning_rate': 0.02990082795716975, 'xgb_subsample': 0.6135653972398647, 'xgb_colsample_bytree': 0.865119691029539, 'cat_iterations': 470, 'cat_depth': 7, 'cat_learning_rate': 0.10782969202284111, 'cat_l2_leaf_reg': 6.478315336770665, 'lgbm_n_estimators': 246, 'lgbm_num_leaves': 44, 'lgbm_learning_rate': 0.07445897860221089, 'lgbm_subsample': 0.7009950796417143, 'lgbm_colsample_bytree': 0.7552507497776738}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000809 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:23,060] Trial 4 finished with value: 0.9615621788283659 and parameters: {'xgb_n_estimators': 355, 'xgb_max_depth': 6, 'xgb_learning_rate': 0.12784972972331432, 'xgb_subsample': 0.7879746801332735, 'xgb_colsample_bytree': 0.7836524850018233, 'cat_iterations': 776, 'cat_depth': 4, 'cat_learning_rate': 0.011841193747887028, 'cat_l2_leaf_reg': 6.2450912937638146, 'lgbm_n_estimators': 326, 'lgbm_num_leaves': 100, 'lgbm_learning_rate': 0.17845236308586687, 'lgbm_subsample': 0.8296785538843232, 'lgbm_colsample_bytree': 0.6669701944869629}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000811 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:26,991] Trial 5 finished with value: 0.9621788283658788 and parameters: {'xgb_n_estimators': 347, 'xgb_max_depth': 3, 'xgb_learning_rate': 0.035248053718290394, 'xgb_subsample': 0.6382785963203411, 'xgb_colsample_bytree': 0.7172214247436062, 'cat_iterations': 452, 'cat_depth': 4, 'cat_learning_rate': 0.04052070526512519, 'cat_l2_leaf_reg': 2.6484170361310353, 'lgbm_n_estimators': 790, 'lgbm_num_leaves': 49, 'lgbm_learning_rate': 0.01836885475685153, 'lgbm_subsample': 0.9342234317254048, 'lgbm_colsample_bytree': 0.6488481909194045}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000285 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:31,784] Trial 6 finished with value: 0.9621788283658788 and parameters: {'xgb_n_estimators': 401, 'xgb_max_depth': 7, 'xgb_learning_rate': 0.08561752344070041, 'xgb_subsample': 0.7101020274734906, 'xgb_colsample_bytree': 0.8568311552993415, 'cat_iterations': 771, 'cat_depth': 4, 'cat_learning_rate': 0.017141785153834504, 'cat_l2_leaf_reg': 1.152238871034434, 'lgbm_n_estimators': 322, 'lgbm_num_leaves': 95, 'lgbm_learning_rate': 0.02055479426317783, 'lgbm_subsample': 0.7404862605562751, 'lgbm_colsample_bytree': 0.847549912113257}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000277 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:35,090] Trial 7 finished with value: 0.9621788283658788 and parameters: {'xgb_n_estimators': 434, 'xgb_max_depth': 4, 'xgb_learning_rate': 0.08678324321430538, 'xgb_subsample': 0.9815789944750153, 'xgb_colsample_bytree': 0.7016678512689646, 'cat_iterations': 315, 'cat_depth': 5, 'cat_learning_rate': 0.09252001554664878, 'cat_l2_leaf_reg': 7.721526714974204, 'lgbm_n_estimators': 800, 'lgbm_num_leaves': 45, 'lgbm_learning_rate': 0.057176058722002286, 'lgbm_subsample': 0.6556156016822504, 'lgbm_colsample_bytree': 0.8972152404420559}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000808 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:38,932] Trial 8 finished with value: 0.9615621788283659 and parameters: {'xgb_n_estimators': 584, 'xgb_max_depth': 3, 'xgb_learning_rate': 0.0937882586546546, 'xgb_subsample': 0.6106142822006265, 'xgb_colsample_bytree': 0.8759081582250914, 'cat_iterations': 231, 'cat_depth': 7, 'cat_learning_rate': 0.12916490804809996, 'cat_l2_leaf_reg': 8.65249366961622, 'lgbm_n_estimators': 678, 'lgbm_num_leaves': 80, 'lgbm_learning_rate': 0.05163959865890142, 'lgbm_subsample': 0.9895719109056068, 'lgbm_colsample_bytree': 0.6528241575354685}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000297 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:42,778] Trial 9 finished with value: 0.9617677286742035 and parameters: {'xgb_n_estimators': 763, 'xgb_max_depth': 3, 'xgb_learning_rate': 0.05817058645667214, 'xgb_subsample': 0.8559915112814331, 'xgb_colsample_bytree': 0.6603477480900437, 'cat_iterations': 256, 'cat_depth': 8, 'cat_learning_rate': 0.05759446548180628, 'cat_l2_leaf_reg': 7.46438925111325, 'lgbm_n_estimators': 443, 'lgbm_num_leaves': 88, 'lgbm_learning_rate': 0.012764992375856779, 'lgbm_subsample': 0.6131500747557804, 'lgbm_colsample_bytree': 0.997296486886218}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001325 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:47,814] Trial 10 finished with value: 0.9617677286742035 and parameters: {'xgb_n_estimators': 200, 'xgb_max_depth': 5, 'xgb_learning_rate': 0.0127705773547297, 'xgb_subsample': 0.7606039661838502, 'xgb_colsample_bytree': 0.6231120929517647, 'cat_iterations': 646, 'cat_depth': 6, 'cat_learning_rate': 0.021327713804234257, 'cat_l2_leaf_reg': 4.466038759944273, 'lgbm_n_estimators': 587, 'lgbm_num_leaves': 66, 'lgbm_learning_rate': 0.10916257397852, 'lgbm_subsample': 0.8459182959332289, 'lgbm_colsample_bytree': 0.7610976122702912}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001437 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:51,377] Trial 11 finished with value: 0.9621788283658788 and parameters: {'xgb_n_estimators': 598, 'xgb_max_depth': 7, 'xgb_learning_rate': 0.01735124231465819, 'xgb_subsample': 0.8877623016276431, 'xgb_colsample_bytree': 0.9344036628466104, 'cat_iterations': 450, 'cat_depth': 6, 'cat_learning_rate': 0.030189010452318348, 'cat_l2_leaf_reg': 5.169178034071815, 'lgbm_n_estimators': 256, 'lgbm_num_leaves': 20, 'lgbm_learning_rate': 0.09563833614333267, 'lgbm_subsample': 0.7301939578629717, 'lgbm_colsample_bytree': 0.7780604931744547}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000282 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:57,166] Trial 12 finished with value: 0.9607399794450154 and parameters: {'xgb_n_estimators': 632, 'xgb_max_depth': 6, 'xgb_learning_rate': 0.021443379126763938, 'xgb_subsample': 0.7128080708304795, 'xgb_colsample_bytree': 0.8365487007216363, 'cat_iterations': 591, 'cat_depth': 7, 'cat_learning_rate': 0.0973923421862854, 'cat_l2_leaf_reg': 9.567918255642013, 'lgbm_n_estimators': 584, 'lgbm_num_leaves': 53, 'lgbm_learning_rate': 0.18843775943892177, 'lgbm_subsample': 0.7339742703164107, 'lgbm_colsample_bytree': 0.6024182970505576}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001311 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:42:59,812] Trial 13 finished with value: 0.9621788283658788 and parameters: {'xgb_n_estimators': 487, 'xgb_max_depth': 4, 'xgb_learning_rate': 0.025618757670396123, 'xgb_subsample': 0.6032058702571644, 'xgb_colsample_bytree': 0.7894289684161276, 'cat_iterations': 383, 'cat_depth': 5, 'cat_learning_rate': 0.012002313256205583, 'cat_l2_leaf_reg': 3.4694025441818113, 'lgbm_n_estimators': 221, 'lgbm_num_leaves': 37, 'lgbm_learning_rate': 0.07796910991536282, 'lgbm_subsample': 0.7000614486068898, 'lgbm_colsample_bytree': 0.7170305376178312}. Best is trial 1 with value: 0.9621788283658788.
[LightGBM] [Info] Number of positive: 6114, number of negative: 13345
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001331 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 258
[LightGBM] [Info] Number of data points in the train set: 19459, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314199 -> initscore=-0.780561
[LightGBM] [Info] Start training from score -0.780561
[I 2025-08-01 07:43:05,892] Trial 14 finished with value: 0.9617677286742035 and parameters: {'xgb_n_estimators': 685, 'xgb_max_depth': 4, 'xgb_learning_rate': 0.04540353588590965, 'xgb_subsample': 0.7069084167133699, 'xgb_colsample_bytree': 0.9105969595704857, 'cat_iterations': 691, 'cat_depth': 7, 'cat_learning_rate': 0.19195034466168076, 'cat_l2_leaf_reg': 6.2796615033583505, 'lgbm_n_estimators': 532, 'lgbm_num_leaves': 60, 'lgbm_learning_rate': 0.03117595426372078, 'lgbm_subsample': 0.7938669154076096, 'lgbm_colsample_bytree': 0.8405794770545059}. Best is trial 1 with value: 0.9621788283658788.
Best trial accuracy: 0.9622
Best hyperparameters found:
{'xgb_n_estimators': 255, 'xgb_max_depth': 4, 'xgb_learning_rate': 0.02300988115479625, 'xgb_subsample': 0.6019618216464371, 'xgb_colsample_bytree': 0.6463509564195872, 'cat_iterations': 741, 'cat_depth': 5, 'cat_learning_rate': 0.010638622000567607, 'cat_l2_leaf_reg': 5.403728482514485, 'lgbm_n_estimators': 684, 'lgbm_num_leaves': 52, 'lgbm_learning_rate': 0.1464261264895688, 'lgbm_subsample': 0.7501657791011547, 'lgbm_colsample_bytree': 0.6096348951640888}

Training final model on all data with best hyperparameters...
[LightGBM] [Info] Number of positive: 7643, number of negative: 16681
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000268 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 268
[LightGBM] [Info] Number of data points in the train set: 24324, number of used features: 7
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.314216 -> initscore=-0.780480
[LightGBM] [Info] Start training from score -0.780480

Submission file created successfully!
      id Personality
0  18524   Extrovert
1  18525   Introvert
2  18526   Extrovert
3  18527   Extrovert
4  18528   Introvert