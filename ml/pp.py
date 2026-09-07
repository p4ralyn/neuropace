import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from dsc import EEGDataConfig, EEGDataset
from sdg import generate_synthetic_dataset
from ma import build_eeg_model

# Enable GPU acceleration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU acceleration enabled!")
    except RuntimeError as e:
        print(e)

        
class EEGDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, indices, config, target_type='ailment', augment=False, preprocess=True):
        self.dataset = dataset
        self.indices = indices
        self.config = config
        self.target_type = target_type
        self.augment = augment
        self.preprocess = preprocess

        # Determine number of classes based on target type
        self.num_classes = {
            'ailment': self.config.num_classes,
            'stress': self.config.num_stress_levels,
            'mood': self.config.num_mood_states
        }
        
        # Prepare EEG windows and labels
        self.windows = []
        self.labels = {key: [] for key in self.num_classes.keys()}

        for idx in self.indices:
            recording = self.dataset.get_recording(idx)
            if not recording.windows:
                recording.create_windows(self.config.window_size, self.config.window_overlap)

            for window in recording.windows:
                if self.augment:
                    window = self.apply_augmentation(window)
                self.windows.append(window)
                self.labels['ailment'].append(recording.condition_id)
                self.labels['stress'].append(recording.stress_level_id)
                self.labels['mood'].append(recording.mood_state_id)

        # Convert to NumPy arrays
        self.windows = np.array(self.windows)
        for key in self.labels.keys():
            self.labels[key] = tf.keras.utils.to_categorical(self.labels[key], num_classes=self.num_classes[key])

        # Convert to TensorFlow dataset format
        self.dataset_tf = tf.data.Dataset.from_tensor_slices(
            (self.windows, (self.labels['ailment'], self.labels['stress'], self.labels['mood']))
        ).batch(self.config.batch_size)

    def __len__(self):
        return len(self.dataset_tf)

    def __getitem__(self, idx):
        return next(iter(self.dataset_tf))
    
    def apply_augmentation(self, signal):
        noise = np.random.normal(0, 0.05, signal.shape)  # Add Gaussian noise
        scaling_factor = np.random.uniform(0.9, 1.1)  # Scale amplitude
        return (signal + noise) * scaling_factor

# Step 1: Configure EEG Data
config = EEGDataConfig(
    sampling_rate=250,
    channels=19,
    recording_duration=60,
    window_size=5,
    window_overlap=0.5,
    batch_size=32
)

# Step 2: Generate a Larger Synthetic EEG Dataset
dataset = generate_synthetic_dataset(config, n_recordings_per_condition=200)  # Increased from 50 to 200
print(f"Generated {len(dataset.recordings)} synthetic EEG recordings.")

# Step 3: Check Class Distributions & Avoid Data Imbalance
print("Class distributions:")
for condition in config.ailment_classes:
    count = sum(1 for r in dataset.recordings if r.condition == condition)
    print(f"{condition}: {count} recordings")

# Step 4: Split Dataset into Train, Validation, and Test Sets
train_indices, val_indices, test_indices = dataset.split_dataset()
assert not set(train_indices) & set(val_indices) & set(test_indices), "Data leakage detected!"
print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test.")

# Step 5: Create Data Generators with Augmentation
train_generator = EEGDataGenerator(dataset, train_indices, config, target_type='all', augment=True)
val_generator = EEGDataGenerator(dataset, val_indices, config, target_type='all')
test_generator = EEGDataGenerator(dataset, test_indices, config, target_type='all')

# Step 6: Build Improved Model with Dropout Regularization
model = build_eeg_model(config, model_type='hybrid', output_type='all')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
    metrics={
        'ailment_output': 'accuracy',
        'stress_output': 'accuracy',
        'mood_output': 'accuracy'
    }
)
model.summary()

# Step 7: Train Model with Better Regularization & Auto-Saving
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='my_eeg_model.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        checkpoint_callback
    ]
)

# Step 8: Evaluate Model
results = model.evaluate(test_generator)
print(f"Test Accuracy - Ailment: {results[1]:.2f}, Stress: {results[2]:.2f}, Mood: {results[3]:.2f}")

# Step 9: Ensure Model is Saved at the End
model.save("final_eeg_model.h5")
print("Model training complete and saved as 'final_eeg_model.h5'.")

# Step 10: Plot Sample EEG Time-Series Data
def plot_eeg_timeseries(dataset, num_samples=5):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        sample = dataset.recordings[i].windows[0]  # Get first window of each sample
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(sample[:, 0], label='Channel 1')  # Plot first channel
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(f'EEG Sample {i+1}')
    plt.tight_layout()
    plt.show()

plot_eeg_timeseries(dataset)
