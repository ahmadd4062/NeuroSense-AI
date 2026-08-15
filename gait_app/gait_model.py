import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Custom Dense layer that ignores quantization_config
class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        # Remove quantization_config if present
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        # Remove quantization_config from config
        config.pop('quantization_config', None)
        return super().from_config(config)

# Register custom objects
custom_objects = {
    'Dense': CustomDense,
    'LSTM': LSTM,
    'Dropout': Dropout,
    'Sequential': tf.keras.models.Sequential,
}

class GaitPredictor:
    def __init__(self):
        """Load the LSTM model with custom objects"""
        model_path = os.path.join(BASE_DIR, 'models/lstm_gait_model.h5')
        
        if not os.path.exists(model_path):
            model_path = os.path.join(BASE_DIR, 'models/lstm_gait_model.keras')
        
        if not os.path.exists(model_path):
            raise Exception(f"No model found in gait_app/models/")
        
        # Load with custom objects
        with tf.keras.utils.custom_object_scope(custom_objects):
            self.model = load_model(model_path, compile=False)
        
        print(f"✅ LSTM model loaded from: {model_path}")
    
    def predict(self, pose_sequence):
        prob = self.model.predict(pose_sequence, verbose=0)[0][0]
        pred = 1 if prob > 0.5 else 0
        
        return {
            'prediction': 'PD' if pred == 1 else 'Control',
            'pd_probability': float(prob),
            'control_probability': 1 - float(prob)
        }

predictor = GaitPredictor()