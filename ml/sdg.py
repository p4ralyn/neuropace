import numpy as np
from datetime import datetime


from dsc import*

def generate_synthetic_eeg(config, condition, stress_level=None, mood_state=None, patient_id=None):
    """
    Generate synthetic EEG data for a specific condition, stress level, and mood state
    
    Parameters:
    - config: EEGDataConfig object with parameters
    - condition: Condition name (must be in config.ailment_classes)
    - stress_level: Stress level name (must be in config.stress_levels), random if None
    - mood_state: Mood state name (must be in config.mood_states), random if None
    - patient_id: Optional patient identifier
    
    Returns:
    - EEGRecording object with synthetic data
    """
    # Validate condition
    if condition not in config.ailment_classes:
        raise ValueError(f"Unknown condition: {condition}. Must be one of {config.ailment_classes}")
    
    # Get condition ID
    condition_id = config.ailment_classes.index(condition)
    
    # If stress_level not specified, choose randomly
    if stress_level is None:
        stress_level = np.random.choice(config.stress_levels)
    elif stress_level not in config.stress_levels:
        raise ValueError(f"Unknown stress level: {stress_level}. Must be one of {config.stress_levels}")
    
    # Get stress level ID
    stress_level_id = config.stress_levels.index(stress_level)
    
    # If mood_state not specified, choose randomly
    if mood_state is None:
        mood_state = np.random.choice(config.mood_states)
    elif mood_state not in config.mood_states:
        raise ValueError(f"Unknown mood state: {mood_state}. Must be one of {config.mood_states}")
    
    # Get mood state ID
    mood_state_id = config.mood_states.index(mood_state)
    
    # Initialize data array
    samples = config.samples_per_recording
    channels = config.channels
    fs = config.sampling_rate
    duration = config.recording_duration
    t = np.arange(0, duration, 1/fs)
    
    # Create empty array for EEG data
    eeg_data = np.zeros((samples, channels))
    
    # Define frequency components for each condition
    # These are the default parameters that will be adjusted for specific conditions
    base_params = {
        'delta': {'min_freq': 0.5, 'max_freq': 4, 'min_amp': 0.5, 'max_amp': 1.5},
        'theta': {'min_freq': 4, 'max_freq': 8, 'min_amp': 0.3, 'max_amp': 1.0},
        'alpha': {'min_freq': 8, 'max_freq': 13, 'min_amp': 0.5, 'max_amp': 1.5},
        'beta': {'min_freq': 13, 'max_freq': 30, 'min_amp': 0.2, 'max_amp': 0.8},
        'gamma': {'min_freq': 30, 'max_freq': 80, 'min_amp': 0.1, 'max_amp': 0.4}
    }
    
    # Condition-specific adjustments
    condition_params = base_params.copy()
    
    # Metadata to store with the recording
    metadata = {
        'is_synthetic': True,
        'generation_time': datetime.now().isoformat(),
        'condition_params': {},
        'stress_params': {},
        'mood_params': {}
    }
    
    # Adjust frequency parameters based on condition
    if condition == 'normal':
        # Normal post-surgery: moderate alpha and beta, balanced overall
        condition_params['alpha']['min_amp'] = 0.8
        condition_params['alpha']['max_amp'] = 1.8
        condition_params['beta']['min_amp'] = 0.4
        condition_params['beta']['max_amp'] = 1.0
        
    elif condition == 'seizure':
        # Seizure: increased high-frequency activity with rhythmic patterns
        condition_params['beta']['min_amp'] = 1.0
        condition_params['beta']['max_amp'] = 2.5
        condition_params['gamma']['min_amp'] = 0.8
        condition_params['gamma']['max_amp'] = 2.0
        
        # Add seizure-specific metadata
        metadata['seizure_type'] = np.random.choice(['focal', 'generalized'])
        metadata['seizure_severity'] = np.random.choice(['mild', 'moderate', 'severe'])
        
    elif condition == 'delayed_recovery':
        # Delayed recovery: increased delta, decreased beta/gamma
        condition_params['delta']['min_amp'] = 1.2
        condition_params['delta']['max_amp'] = 2.5
        condition_params['beta']['min_amp'] = 0.1
        condition_params['beta']['max_amp'] = 0.4
        condition_params['gamma']['min_amp'] = 0.05
        condition_params['gamma']['max_amp'] = 0.2
        
    elif condition == 'ischemia':
        # Ischemia: slowing with increased delta and theta
        condition_params['delta']['min_amp'] = 1.5
        condition_params['delta']['max_amp'] = 3.0
        condition_params['theta']['min_amp'] = 0.8
        condition_params['theta']['max_amp'] = 1.8
        condition_params['alpha']['min_amp'] = 0.2
        condition_params['alpha']['max_amp'] = 0.6
        
        # Add ischemia-specific metadata
        metadata['ischemia_location'] = np.random.choice(['frontal', 'temporal', 'parietal', 'occipital'])
        metadata['ischemia_severity'] = np.random.choice(['mild', 'moderate', 'severe'])
        
    elif condition == 'hemorrhage':
        # Hemorrhage: asymmetric patterns, suppression on affected side
        condition_params['delta']['min_amp'] = 1.2
        condition_params['delta']['max_amp'] = 2.8
        
        # Add hemorrhage-specific metadata
        metadata['hemorrhage_type'] = np.random.choice(['subdural', 'epidural', 'subarachnoid', 'intracerebral'])
        metadata['hemorrhage_location'] = np.random.choice(['left', 'right'])
        metadata['hemorrhage_severity'] = np.random.choice(['mild', 'moderate', 'severe'])
        
    elif condition == 'infection':
        # Infection: diffuse slowing with some focal abnormalities
        condition_params['theta']['min_amp'] = 0.8
        condition_params['theta']['max_amp'] = 1.6
        condition_params['delta']['min_amp'] = 0.8
        condition_params['delta']['max_amp'] = 2.0
        condition_params['alpha']['min_amp'] = 0.3
        condition_params['alpha']['max_amp'] = 0.9
        
        # Add infection-specific metadata
        metadata['infection_type'] = np.random.choice(['bacterial', 'viral', 'fungal'])
        metadata['infection_severity'] = np.random.choice(['mild', 'moderate', 'severe'])
    
    # Store condition parameters in metadata
    metadata['condition_params'] = condition_params
    
    # Adjust parameters based on stress level
    stress_params = {}
    if stress_level == 'minimal':
        # Minimal stress: increased alpha, decreased beta/gamma
        stress_params['alpha_boost'] = np.random.uniform(1.3, 1.5)
        stress_params['beta_reduction'] = np.random.uniform(0.6, 0.8)
        stress_params['gamma_reduction'] = np.random.uniform(0.5, 0.7)
    elif stress_level == 'mild':
        # Mild stress: slightly decreased alpha, slightly increased beta
        stress_params['alpha_reduction'] = np.random.uniform(0.8, 0.9)
        stress_params['beta_boost'] = np.random.uniform(1.1, 1.3)
    elif stress_level == 'moderate':
        # Moderate stress: moderately decreased alpha, increased beta
        stress_params['alpha_reduction'] = np.random.uniform(0.6, 0.8)
        stress_params['beta_boost'] = np.random.uniform(1.3, 1.6)
    elif stress_level == 'high':
        # High stress: significantly decreased alpha, highly increased beta/gamma
        stress_params['alpha_reduction'] = np.random.uniform(0.4, 0.6)
        stress_params['beta_boost'] = np.random.uniform(1.5, 1.8)
        stress_params['gamma_boost'] = np.random.uniform(1.3, 1.6)
    elif stress_level == 'severe':
        # Severe stress: greatly decreased alpha, very high beta/gamma
        stress_params['alpha_reduction'] = np.random.uniform(0.2, 0.4)
        stress_params['beta_boost'] = np.random.uniform(1.8, 2.2)
        stress_params['gamma_boost'] = np.random.uniform(1.6, 2.0)
    
    # Store stress parameters in metadata
    metadata['stress_params'] = stress_params
    
    # Adjust parameters based on mood state
    mood_params = {}
    if mood_state in ['Pain & Discomfort', 'Anxiety & Fear']:
        # Increased beta and gamma activity
        mood_params['beta_boost'] = np.random.uniform(1.2, 1.5)
        mood_params['gamma_boost'] = np.random.uniform(1.1, 1.4)
    elif mood_state in ['Depression & Mental Fatigue', 'Emotional Exhaustion']:
        # Increased theta, decreased alpha and beta
        mood_params['theta_boost'] = np.random.uniform(1.3, 1.6)
        mood_params['alpha_reduction'] = np.random.uniform(0.6, 0.8)
        mood_params['beta_reduction'] = np.random.uniform(0.7, 0.9)
    elif mood_state in ['Insomnia & Sleep Disturbances']:
        # Decreased delta, increased beta
        mood_params['delta_reduction'] = np.random.uniform(0.5, 0.7)
        mood_params['beta_boost'] = np.random.uniform(1.2, 1.5)
    elif mood_state in ['Cognitive Dysfunction']:
        # Increased theta, decreased alpha
        mood_params['theta_boost'] = np.random.uniform(1.2, 1.5)
        mood_params['alpha_reduction'] = np.random.uniform(0.5, 0.7)
    elif mood_state in ['Post-Surgery PTSD']:
        # Increased beta and gamma, decreased alpha
        mood_params['beta_boost'] = np.random.uniform(1.3, 1.6)
        mood_params['gamma_boost'] = np.random.uniform(1.2, 1.5)
        mood_params['alpha_reduction'] = np.random.uniform(0.5, 0.7)
    elif mood_state in ['Relaxation & Recovery']:
        # Increased alpha, decreased beta and gamma
        mood_params['alpha_boost'] = np.random.uniform(1.4, 1.7)
        mood_params['beta_reduction'] = np.random.uniform(0.6, 0.8)
        mood_params['gamma_reduction'] = np.random.uniform(0.5, 0.7)
    elif mood_state in ['Deep Sleep']:
        # High delta, low everything else
        mood_params['delta_boost'] = np.random.uniform(1.8, 2.2)
        mood_params['theta_reduction'] = np.random.uniform(0.5, 0.7)
        mood_params['alpha_reduction'] = np.random.uniform(0.3, 0.5)
        mood_params['beta_reduction'] = np.random.uniform(0.2, 0.4)
        mood_params['gamma_reduction'] = np.random.uniform(0.1, 0.3)
    elif mood_state in ['Focused Attention']:
        # Increased beta, moderate alpha
        mood_params['beta_boost'] = np.random.uniform(1.3, 1.6)
        mood_params['alpha_boost'] = np.random.uniform(1.1, 1.3)
    
    # Store mood parameters in metadata
    metadata['mood_params'] = mood_params
    
    # Generate base signals for each channel
    for c in range(channels):
        # Initialize channel data
        channel_data = np.zeros(samples)
        
        # Generate components for each frequency band
        for band, params in condition_params.items():
            # Determine number of component frequencies in this band
            n_components = np.random.randint(3, 6)
            
            # Generate each component
            for _ in range(n_components):
                # Select random frequency within band
                freq = np.random.uniform(params['min_freq'], params['max_freq'])
                
                # Select random amplitude within band
                base_amp = np.random.uniform(params['min_amp'], params['max_amp'])
                
                # Apply stress and mood modifiers to amplitude
                amp = base_amp
                
                # Apply stress modifiers
                if band == 'alpha' and 'alpha_reduction' in stress_params:
                    amp *= stress_params['alpha_reduction']
                elif band == 'alpha' and 'alpha_boost' in stress_params:
                    amp *= stress_params['alpha_boost']
                elif band == 'beta' and 'beta_reduction' in stress_params:
                    amp *= stress_params['beta_reduction']
                elif band == 'beta' and 'beta_boost' in stress_params:
                    amp *= stress_params['beta_boost']
                elif band == 'gamma' and 'gamma_reduction' in stress_params:
                    amp *= stress_params['gamma_reduction']
                elif band == 'gamma' and 'gamma_boost' in stress_params:
                    amp *= stress_params['gamma_boost']
                
                # Apply mood modifiers
                if band == 'delta' and 'delta_reduction' in mood_params:
                    amp *= mood_params['delta_reduction']
                elif band == 'delta' and 'delta_boost' in mood_params:
                    amp *= mood_params['delta_boost']
                elif band == 'theta' and 'theta_reduction' in mood_params:
                    amp *= mood_params['theta_reduction']
                elif band == 'theta' and 'theta_boost' in mood_params:
                    amp *= mood_params['theta_boost']
                elif band == 'alpha' and 'alpha_reduction' in mood_params:
                    amp *= mood_params['alpha_reduction']
                elif band == 'alpha' and 'alpha_boost' in mood_params:
                    amp *= mood_params['alpha_boost']
                elif band == 'beta' and 'beta_reduction' in mood_params:
                    amp *= mood_params['beta_reduction']
                elif band == 'beta' and 'beta_boost' in mood_params:
                    amp *= mood_params['beta_boost']
                elif band == 'gamma' and 'gamma_reduction' in mood_params:
                    amp *= mood_params['gamma_reduction']
                elif band == 'gamma' and 'gamma_boost' in mood_params:
                    amp *= mood_params['gamma_boost']
                
                # Add special condition-specific patterns
                if condition == 'seizure' and (band == 'beta' or band == 'gamma'):
                    # Add rhythmic seizure patterns
                    if np.random.random() < 0.7:  # 70% chance of adding seizure pattern
                        # Create envelope for seizure burst
                        seizure_start = np.random.randint(0, samples - samples//4)
                        seizure_duration = np.random.randint(samples//10, samples//4)
                        seizure_envelope = np.zeros(samples)
                        seizure_envelope[seizure_start:seizure_start+seizure_duration] = np.hanning(seizure_duration)
                        
                        # Add high-frequency oscillation with envelope
                        seizure_freq = np.random.uniform(15, 25)  # Typical seizure frequency
                        seizure_component = amp * 3 * seizure_envelope * np.sin(2 * np.pi * seizure_freq * t)
                        channel_data += seizure_component
                
                # Generate random phase
                phase = np.random.uniform(0, 2 * np.pi)
                
                # Generate the signal component
                component = amp * np.sin(2 * np.pi * freq * t + phase)
                
                # Add to channel data
                channel_data += component
        
        # Apply channel-specific modulations for conditions with spatial characteristics
        if condition == 'hemorrhage':
            # Simulate asymmetric suppression for hemorrhage
            if metadata['hemorrhage_location'] == 'left' and c < channels // 2:
                # Suppress left side channels
                channel_data *= np.random.uniform(0.3, 0.6)
            elif metadata['hemorrhage_location'] == 'right' and c >= channels // 2:
                # Suppress right side channels
                channel_data *= np.random.uniform(0.3, 0.6)
        
        # Add random noise
        noise_level = np.random.uniform(0.05, 0.15)
        noise = np.random.normal(0, noise_level, samples)
        channel_data += noise
        
        # Store in EEG data array
        eeg_data[:, c] = channel_data
    
    # Create EEG recording object
    recording = EEGRecording(
        data=eeg_data,
        sampling_rate=config.sampling_rate,
        patient_id=patient_id or f"SYN{np.random.randint(1000, 9999)}",
        timestamp=datetime.now(),
        condition=condition,
        condition_id=condition_id,
        stress_level=stress_level,
        stress_level_id=stress_level_id,
        mood_state=mood_state,
        mood_state_id=mood_state_id,
        metadata=metadata
    )
    
    return recording


def generate_synthetic_dataset(config, n_recordings_per_condition=20):
    """
    Generate a synthetic dataset with multiple recordings for each condition
    
    Parameters:
    - config: EEGDataConfig object with parameters
    - n_recordings_per_condition: Number of recordings to generate per condition
    
    Returns:
    - EEGDataset object with synthetic recordings
    """
    dataset = EEGDataset(config)
    
    for condition in config.ailment_classes:
        for _ in range(n_recordings_per_condition):
            # Randomly select stress level and mood state
            stress_level = np.random.choice(config.stress_levels)
            mood_state = np.random.choice(config.mood_states)
            
            # Generate synthetic recording
            recording = generate_synthetic_eeg(
                config=config,
                condition=condition,
                stress_level=stress_level,
                mood_state=mood_state
            )
            
            # Add to dataset
            dataset.add_recording(recording)
    
    # Split dataset into train, validation, and test sets
    dataset.split_dataset(stratify=True)
    
    return dataset