import numpy as np
import pandas as pd
import os
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf

class EEGDataConfig:
    """Configuration parameters for EEG data and processing"""
    
    def __init__(self, 
                 sampling_rate=250,         # Hz
                 channels=19,               # Number of EEG channels
                 recording_duration=60,     # Seconds
                 window_size=5,             # Seconds
                 window_overlap=0.5,        # Overlap ratio between windows
                 batch_size=32,             # Batch size for training
                 ailment_classes=None,      # List of ailment classes
                 stress_levels=None,        # List of stress level classes
                 mood_states=None,          # List of mood state classes
                 frequency_bands=None,      # Frequency bands of interest
                 electrode_positions=None,  # Channel positions
                 ):
        
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.recording_duration = recording_duration
        self.samples_per_recording = int(recording_duration * sampling_rate)
        self.window_size = window_size
        self.samples_per_window = int(window_size * sampling_rate)
        self.window_overlap = window_overlap
        self.window_step = int(self.samples_per_window * (1 - window_overlap))
        self.batch_size = batch_size
        
        # Define default mood states if not provided
        if mood_states is None:
            self.mood_states = [
                'Pain & Discomfort',
                'Anxiety & Fear',
                'Depression & Mental Fatigue',
                'Insomnia & Sleep Disturbances',
                'Cognitive Dysfunction',
                'Emotional Exhaustion',
                'Post-Surgery PTSD',
                'Relaxation & Recovery',
                'Deep Sleep',
                'Focused Attention'
            ]
        else:
            self.mood_states = mood_states
        
        # Define default ailment classes if not provided
        if ailment_classes is None:
            self.ailment_classes = [
                'normal',           # Typical post-surgical recovery
                'seizure',          # Post-operative seizure
                'delayed_recovery', # Slow to regain normal brain function
                'ischemia',         # Reduced blood flow
                'hemorrhage',       # Bleeding in the brain
                'infection'         # Post-surgical infection signs
            ]
        else:
            self.ailment_classes = ailment_classes
            
        # Define default stress levels if not provided
        if stress_levels is None:
            self.stress_levels = [
                'minimal',          # Very low stress
                'mild',             # Low stress
                'moderate',         # Medium stress
                'high',             # High stress
                'severe'            # Very high stress
            ]
        else:
            self.stress_levels = stress_levels
        
        # Number of classes for each category
        self.num_classes = len(self.ailment_classes)
        self.num_stress_levels = len(self.stress_levels)
        self.num_mood_states = len(self.mood_states)
        
        # Define default frequency bands if not provided
        if frequency_bands is None:
            self.frequency_bands = {
                'delta': (0.5, 4),   # Deep sleep, unconscious states
                'theta': (4, 8),     # Drowsiness, meditation, some pathologies
                'alpha': (8, 13),    # Relaxed wakefulness, closed eyes
                'beta': (13, 30),    # Active thinking, focus, alert state
                'gamma': (30, 80)    # Cognitive processing, active problem solving
            }
        else:
            self.frequency_bands = frequency_bands
            
        # Define default electrode positions based on international 10-20 system
        if electrode_positions is None and channels == 19:
            self.electrode_positions = {
                0: 'Fp1', 1: 'Fp2', 2: 'F7', 3: 'F3', 4: 'Fz', 5: 'F4', 6: 'F8',
                7: 'T3', 8: 'C3', 9: 'Cz', 10: 'C4', 11: 'T4', 12: 'T5', 13: 'P3',
                14: 'Pz', 15: 'P4', 16: 'T6', 17: 'O1', 18: 'O2'
            }
        else:
            self.electrode_positions = electrode_positions
    
    def to_dict(self):
        """Convert configuration to dictionary for saving"""
        return {
            'sampling_rate': self.sampling_rate,
            'channels': self.channels,
            'recording_duration': self.recording_duration,
            'window_size': self.window_size,
            'window_overlap': self.window_overlap,
            'batch_size': self.batch_size,
            'ailment_classes': self.ailment_classes,
            'stress_levels': self.stress_levels,
            'mood_states': self.mood_states,
            'frequency_bands': self.frequency_bands,
            'electrode_positions': self.electrode_positions
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def save(self, filepath):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, filepath):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class EEGRecording:
    """Class to represent an EEG recording with metadata"""
    
    def __init__(self,
                 data,                # Raw EEG data array (samples, channels)
                 sampling_rate,       # Sampling rate in Hz
                 patient_id=None,     # Patient identifier
                 timestamp=None,      # Recording timestamp
                 condition=None,      # Patient condition/ailment
                 condition_id=None,   # Numeric label for condition
                 stress_level=None,   # Stress level (string label)
                 stress_level_id=None,# Numeric label for stress level
                 mood_state=None,     # Mood state (string label)
                 mood_state_id=None,  # Numeric label for mood state
                 metadata=None,       # Additional metadata dictionary
                 windows=None):       # Preprocessed data windows if any
        
        self.data = data
        self.sampling_rate = sampling_rate
        self.patient_id = patient_id
        self.timestamp = timestamp or datetime.now()
        self.condition = condition
        self.condition_id = condition_id
        self.stress_level = stress_level
        self.stress_level_id = stress_level_id
        self.mood_state = mood_state
        self.mood_state_id = mood_state_id
        self.metadata = metadata or {}
        self.windows = windows or []
    
    def create_windows(self, window_size, window_overlap=0.5):
        """
        Split recording into overlapping windows
        
        Parameters:
        - window_size: Window size in seconds
        - window_overlap: Overlap between windows (0-1)
        
        Returns:
        - List of windows (numpy arrays)
        """
        samples = self.data.shape[0]
        window_samples = int(window_size * self.sampling_rate)
        step = int(window_samples * (1 - window_overlap))
        
        windows = []
        for i in range(0, samples - window_samples + 1, step):
            windows.append(self.data[i:i+window_samples, :])
        
        self.windows = windows
        return windows
    
    def save(self, filepath):
        """Save recording to file using pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        """Load recording from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class EEGDataset:
    """Class to manage a collection of EEG recordings"""
    
    def __init__(self, config):
        self.config = config
        self.recordings = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
    
    def add_recording(self, recording):
        """Add a recording to the dataset"""
        self.recordings.append(recording)
        return len(self.recordings) - 1  # Return index of added recording
    
    def add_recordings(self, recordings):
        """Add multiple recordings to the dataset"""
        indices = []
        for recording in recordings:
            idx = self.add_recording(recording)
            indices.append(idx)
        return indices
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=True):
        """
        Split dataset into train, validation and test sets
        
        Parameters:
        - train_ratio: Ratio of data for training
        - val_ratio: Ratio of data for validation
        - test_ratio: Ratio of data for testing
        - stratify: Whether to maintain class distribution in splits
        
        Returns:
        - Tuple of (train_indices, val_indices, test_indices)
        """
        # Check ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError("Split ratios must sum to 1.0")
        
        indices = list(range(len(self.recordings)))
        
        if stratify:
            # Get conditions for stratification
            conditions = [r.condition_id for r in self.recordings]
            
            # First split: separate test set
            train_val_indices, test_indices = train_test_split(
                indices, 
                test_size=test_ratio, 
                stratify=conditions, 
                random_state=42
            )
            
            # Recalculate conditions for train/val split
            train_val_conditions = [conditions[i] for i in train_val_indices]
            
            # Second split: separate train and validation sets
            # Adjust val_ratio to account for the removed test set
            adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
            
            train_indices, val_indices = train_test_split(
                train_val_indices, 
                test_size=adjusted_val_ratio, 
                stratify=train_val_conditions, 
                random_state=42
            )
        else:
            # Simple non-stratified splits
            train_end = int(len(indices) * train_ratio)
            val_end = int(len(indices) * (train_ratio + val_ratio))
            
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
        
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        
        return train_indices, val_indices, test_indices
    
    def get_recording(self, idx):
        """Get recording by index"""
        return self.recordings[idx]
    
    def save(self, filepath):
        """Save dataset to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        """Load dataset from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def convert_to_tabular(self, output_dir="tabular_eeg_data"):
        """
        Convert all EEG recordings to tabular format and save as CSV files
        
        Parameters:
        - output_dir: Directory to save the tabular data
        
        Returns:
        - List of DataFrame paths
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        file_paths = []
        
        for idx, recording in enumerate(self.recordings):
            # Create a list to store rows of the tabular data
            tabular_data = []
            
            # Time step in milliseconds
            time_step = 1000 / recording.sampling_rate
            
            # For each time point
            for t in range(recording.data.shape[0]):
                # For each channel
                for ch in range(recording.data.shape[1]):
                    # Create a row for this time point and channel
                    row = {
                        'patient_id': recording.patient_id or f"P{idx}",
                        'recording_id': idx,
                        'time_ms': t * time_step,
                        'channel': ch,
                        'channel_name': self.config.electrode_positions.get(ch, f"CH{ch}") if self.config.electrode_positions else f"CH{ch}",
                        'amplitude': recording.data[t, ch],
                        'condition': recording.condition,
                        'condition_id': recording.condition_id,
                        'stress_level': recording.stress_level,
                        'stress_level_id': recording.stress_level_id,
                        'mood_state': recording.mood_state,
                        'mood_state_id': recording.mood_state_id
                    }
                    
                    # Add any additional metadata
                    for key, value in recording.metadata.items():
                        if not isinstance(value, dict) and not isinstance(value, list):
                            row[key] = value
                    
                    tabular_data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(tabular_data)
            
            # Save to CSV
            filepath = os.path.join(output_dir, f"recording_{idx}.csv")
            df.to_csv(filepath, index=False)
            file_paths.append(filepath)
            
            # Also save a downsampled version for easier analysis
            if recording.data.shape[0] > 1000:
                downsample_ratio = recording.data.shape[0] // 1000
                downsampled_df = df[df.index % downsample_ratio == 0]
                downsampled_filepath = os.path.join(output_dir, f"recording_{idx}_downsampled.csv")
                downsampled_df.to_csv(downsampled_filepath, index=False)
                file_paths.append(downsampled_filepath)
        
        # Create a summary file
        summary_filepath = os.path.join(output_dir, "tabular_data_summary.csv")
        summary_data = []
        
        for idx, recording in enumerate(self.recordings):
            summary_data.append({
                'recording_id': idx,
                'patient_id': recording.patient_id or f"P{idx}",
                'duration_sec': recording.data.shape[0] / recording.sampling_rate,
                'channels': recording.data.shape[1],
                'condition': recording.condition,
                'stress_level': recording.stress_level,
                'mood_state': recording.mood_state,
                'timestamp': recording.timestamp
            })
        
        pd.DataFrame(summary_data).to_csv(summary_filepath, index=False)
        file_paths.append(summary_filepath)
        
        return file_paths


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
    
    def shuffle_indices(self):
        """Shuffle indices for each epoch"""
        self.window_indices = np.arange(len(self.windows))
        np.random.shuffle(self.window_indices)
    
    """def __len__(self):
        Return number of batches per epoch
        return int(np.ceil(len(self.windows) / self.config.batch_size))
    
    def __getitem__(self, idx):
        Get batch at index idx
        # Get indices for this batch
        batch_indices = self.window_indices[
            idx * self.config.batch_size:(idx + 1) * self.config.batch_size
        ]
        
        # Get windows and labels
        batch_windows = self.windows[batch_indices]
        
        if self.target_type == 'ailment':
            batch_labels = self.ailment_labels_one_hot[batch_indices]
        elif self.target_type == 'stress':
            batch_labels = self.stress_labels_one_hot[batch_indices]
        elif self.target_type == 'mood':
            batch_labels = self.mood_labels_one_hot[batch_indices]
        else:  # 'all' or any other string - return all targets
            batch_labels = [
                self.ailment_labels_one_hot[batch_indices],
                self.stress_labels_one_hot[batch_indices],
                self.mood_labels_one_hot[batch_indices]
            ]
        
        # Apply preprocessing if needed
        if self.preprocess:
            batch_windows = self.preprocess_batch(batch_windows)
        
        # Apply augmentation if needed
        if self.augment:
            batch_windows = self.augment_batch(batch_windows)
        
        return batch_windows, batch_labels
    """

    def on_epoch_end(self):
        """Called at the end of each epoch"""
        self.shuffle_indices()
        
    def preprocess_batch(self, batch):
        """
        Preprocess a batch of EEG windows
        
        Parameters:
        - batch: Batch of EEG windows (batch_size, samples, channels)
        
        Returns:
        - Preprocessed batch
        """
        # Implement simple preprocessing
        # 1. Normalize each channel to zero mean and unit variance
        for i in range(batch.shape[0]):
            for c in range(batch.shape[2]):
                channel_data = batch[i, :, c]
                if np.std(channel_data) > 0:
                    batch[i, :, c] = (channel_data - np.mean(channel_data)) / np.std(channel_data)
        
        return batch
    
    def augment_batch(self, batch):
        """
        Apply data augmentation to a batch of EEG windows
        
        Parameters:
        - batch: Batch of EEG windows (batch_size, samples, channels)
        
        Returns:
        - Augmented batch
        """
        augmented_batch = batch.copy()
        
        for i in range(augmented_batch.shape[0]):
            # Randomly choose an augmentation technique
            aug_type = np.random.choice([
                'noise', 'scaling', 'channel_dropout', 'time_shift', 'none'
            ])
            
            if aug_type == 'noise':
                # Add random noise
                noise_level = np.random.uniform(0.01, 0.1)
                noise = np.random.normal(0, noise_level, augmented_batch[i].shape)
                augmented_batch[i] += noise
                
            elif aug_type == 'scaling':
                # Random amplitude scaling
                scale_factor = np.random.uniform(0.8, 1.2)
                augmented_batch[i] *= scale_factor
                
            elif aug_type == 'channel_dropout':
                # Randomly zero out some channels
                n_channels = augmented_batch.shape[2]
                n_dropout = np.random.randint(1, max(2, n_channels // 5))
                channels_to_drop = np.random.choice(n_channels, n_dropout, replace=False)
                augmented_batch[i, :, channels_to_drop] = 0
                
            elif aug_type == 'time_shift':
                # Shift time axis slightly
                shift = np.random.randint(-20, 21)
                if shift > 0:
                    augmented_batch[i, shift:, :] = augmented_batch[i, :-shift, :]
                    augmented_batch[i, :shift, :] = 0
                elif shift < 0:
                    shift = abs(shift)
                    augmented_batch[i, :-shift, :] = augmented_batch[i, shift:, :]
                    augmented_batch[i, -shift:, :] = 0
        
        return augmented_batch