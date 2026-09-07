
import tensorflow as tf
import matplotlib.pyplot as plt
from dsc import EEGDataConfig, EEGDataGenerator
from sdg import generate_synthetic_dataset
from ma import build_eeg_model

# Step 1: Configure EEG Data
config = EEGDataConfig(
    sampling_rate=250,
    channels=19,
    recording_duration=60,
    window_size=5,
    window_overlap=0.5,
    batch_size=32
)

# Step 2: Generate Synthetic EEG Dataset with More Data Diversity
dataset = generate_synthetic_dataset(config, n_recordings_per_condition=50)
print(f"Generated {len(dataset.recordings)} synthetic EEG recordings.")

# Step 3: Check Class Distributions
print("Class distributions:")
for condition in config.ailment_classes:
    count = sum(1 for r in dataset.recordings if r.condition == condition)
    print(f"{condition}: {count} recordings")

# Step 4: Split Dataset into Train, Validation, and Test Sets
train_indices, val_indices, test_indices = dataset.split_dataset()
print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test.")

# Step 5: Create Data Generators
train_generator = EEGDataGenerator(dataset, train_indices, config, target_type='all', augment=True)
val_generator = EEGDataGenerator(dataset, val_indices, config, target_type='all')
test_generator = EEGDataGenerator(dataset, test_indices, config, target_type='all')

# Step 6: Build Simplified Model with Lower Learning Rate
model = build_eeg_model(config, model_type='hybrid', output_type='all')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
    metrics={
        'ailment_output': 'accuracy',
        'stress_output': 'accuracy',
        'mood_output': 'accuracy'
    }
)
model.summary()

# Step 7: Train Model with Improved Stopping Criteria & Auto-Save
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_eeg_model.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=75,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
        checkpoint_callback
    ]
)

# Step 8: Evaluate Model
results = model.evaluate(test_generator)
print(f"Test Accuracy - Ailment: {results[1]:.2f}, Stress: {results[2]:.2f}, Mood: {results[3]:.2f}")

# Step 9: Plot Training Performance Without Blocking Execution
def plot_training_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.pause(0.1)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['ailment_output_accuracy'], label='Ailment Accuracy')
    plt.plot(history.history['val_ailment_output_accuracy'], label='Val Ailment Accuracy')
    plt.plot(history.history['stress_output_accuracy'], label='Stress Accuracy')
    plt.plot(history.history['val_stress_output_accuracy'], label='Val Stress Accuracy')
    plt.plot(history.history['mood_output_accuracy'], label='Mood Accuracy')
    plt.plot(history.history['val_mood_output_accuracy'], label='Val Mood Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.pause(0.1)
    plt.show(block=False)

plot_training_curves(history)

# Step 10: Ensure Model is Saved at the End
model.save("final_eeg_model.h5")
print("Model training complete and saved as 'final_eeg_model.h5'.")