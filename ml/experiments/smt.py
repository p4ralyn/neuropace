import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset (assuming preprocessed EEG data is available)
def load_eeg_data():
    # Placeholder for actual EEG feature extraction
    # Replace with real data loading logic
    X = np.random.rand(1000, 50)  # Example: 1000 samples, 50 extracted features
    y = np.random.randint(0, 2, 1000)  # Binary classification (mood states)
    return X, y

# Load and split the data
X, y = load_eeg_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False
)

# Train the model
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="logloss",  # Add evaluation metric
    early_stopping_rounds=10,  # Keep early stopping
    verbose=True
)

# Evaluate the model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
