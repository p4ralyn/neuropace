from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    Dense, Flatten, GlobalAveragePooling1D, Concatenate, LSTM,
    Bidirectional
)

from tensorflow.keras.optimizers import Adam

def build_eeg_model(config, model_type='cnn', output_type='ailment'):
    """
    Build a neural network model for EEG classification
    
    Parameters:
    - config: EEGDataConfig object with parameters
    - model_type: Type of model architecture ('cnn', 'lstm', 'hybrid')
    - output_type: Type of output ('ailment', 'stress', 'mood', or 'all')
    
    Returns:
    - Compiled Keras model
    """
    # Define input shape
    input_shape = (config.samples_per_window, config.channels)
    
    # Define number of output classes based on output type
    if output_type == 'ailment':
        num_classes = config.num_classes
    elif output_type == 'stress':
        num_classes = config.num_stress_levels
    elif output_type == 'mood':
        num_classes = config.num_mood_states
    else:  # 'all'
        num_classes = [config.num_classes, config.num_stress_levels, config.num_mood_states]
    
    # Create input layer
    inputs = Input(shape=input_shape)
    
    # Build different model architectures
    if model_type == 'cnn':
        # CNN model architecture
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        
    elif model_type == 'lstm':
        # LSTM model architecture
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(64))(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        
    elif model_type == 'hybrid':
        # Hybrid CNN-LSTM model
        # CNN branch
        cnn = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        cnn = BatchNormalization()(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Conv1D(128, kernel_size=3, activation='relu', padding='same')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        
        # LSTM branch
        lstm = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        lstm = Dropout(0.3)(lstm)
        lstm = Bidirectional(LSTM(64))(lstm)
        
        # Combine branches
        cnn_flat = GlobalAveragePooling1D()(cnn)
        x = Concatenate()([cnn_flat, lstm])
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create output layer(s)
    if output_type == 'all':
        # Multiple outputs for multi-task learning
        outputs = [
            Dense(num_classes[0], activation='softmax', name='ailment_output')(x),
            Dense(num_classes[1], activation='softmax', name='stress_output')(x),
            Dense(num_classes[2], activation='softmax', name='mood_output')(x)
        ]
        
        # Compile model with multiple losses and metrics
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
            metrics=['accuracy']
        )
    else:
        # Single output
        outputs = Dense(num_classes, activation='softmax')(x)
        
        # Compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model