import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sdg import generate_synthetic_dataset
import multiprocessing
from dsc import EEGDataConfig  # Import EEGDataConfig


# Step 1: Generate Synthetic Data
config = EEGDataConfig()
dataset = generate_synthetic_dataset(config, n_recordings_per_condition=50)

# Extract EEG features (example: mean, std, and frequency-domain features)
def extract_features(eeg_data):
    features = []
    for recording in eeg_data.recordings:
        signal = np.array(recording.data)
        mean_val = np.mean(signal, axis=0)
        std_val = np.std(signal, axis=0)
        features.append(np.concatenate([mean_val, std_val]))
    return np.array(features)

X = extract_features(dataset)
y = np.array([recording.condition for recording in dataset.recordings])

# Encode labels
y_encoded = pd.factorize(y)[0]

# Step 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 3: Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train XGBoost Model with Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'n_jobs': [multiprocessing.cpu_count()]  # Wrap it in a list!
}


xgb = XGBClassifier(eval_metric='logloss', n_jobs=multiprocessing.cpu_count())
grid_search = GridSearchCV(xgb, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Step 5: Evaluate Model
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"Best Model Accuracy: {accuracy:.2f}")
