import joblib
import os
import numpy as np

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class GaitPredictor:
    def __init__(self):
        """Load the trained model, scaler, and feature names"""
        model_path = os.path.join(BASE_DIR, 'models/gait_final_model.pkl')
        scaler_path = os.path.join(BASE_DIR, 'models/gait_final_scaler.pkl')
        features_path = os.path.join(BASE_DIR, 'models/gait_final_features.pkl')
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)
        
        print("✅ Gait model loaded successfully!")
        print(f"   Features: {self.features}")
    
    def predict(self, feature_array):
        """
        Predict PD or Control from 10 gait features.
        feature_array: numpy array of 10 features (in correct order)
        Returns: dict with prediction and probabilities
        """
        # Scale the features
        scaled = self.scaler.transform([feature_array])
        
        # Get prediction
        pred = self.model.predict(scaled)[0]
        
        # Get probabilities
        prob = self.model.predict_proba(scaled)[0]
        
        return {
            'prediction': 'PD' if pred == 1 else 'Control',
            'pd_probability': float(prob[1]),
            'control_probability': float(prob[0])
        }

# Create a global instance for use in views
predictor = GaitPredictor()