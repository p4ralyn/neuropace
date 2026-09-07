import xgboost as xgb
import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.signal import welch
from scipy.stats import entropy

# Function to extract improved EEG features
def extract_features(eeg_data):
    features = []
    for signal in eeg_data:
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        psd_vals, freqs = welch(signal)  # Power Spectral Density
        psd_mean = np.mean(psd_vals)
        psd_std = np.std(psd_vals)
        entropy_val = entropy(psd_vals)  # Spectral entropy
        delta_power = np.sum(psd_vals[(freqs >= 0.5) & (freqs < 4)])  # Delta band power
        theta_power = np.sum(psd_vals[(freqs >= 4) & (freqs < 8)])  # Theta band power
        alpha_power = np.sum(psd_vals[(freqs >= 8) & (freqs < 12)])  # Alpha band power
        beta_power = np.sum(psd_vals[(freqs >= 12) & (freqs < 30)])  # Beta band power
        gamma_power = np.sum(psd_vals[(freqs >= 30)])  # Gamma band power
        features.append([mean_val, std_val, psd_mean, psd_std, entropy_val,
                         delta_power, theta_power, alpha_power, beta_power, gamma_power])
    return np.array(features)

# Load dataset (assuming preprocessed EEG data is available)
def load_eeg_data():
    X = np.random.rand(1000, 50)  # Example: 1000 samples, 50 raw features
    y = np.random.randint(0, 2, 1000)  # Binary classification (mood states)
    return extract_features(X), y

# Load and split the data
X, y = load_eeg_data()
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning using Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    model = xgb.XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        scores.append(accuracy_score(y_v, preds))
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Train final model with best hyperparameters
best_params = study.best_params
best_params['eval_metric'] = 'logloss'
best_params['use_label_encoder'] = False
xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Best Model Accuracy: {accuracy:.4f}")

# SHAP for model interpretability
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_val)
shap.summary_plot(shap_values, X_val)
