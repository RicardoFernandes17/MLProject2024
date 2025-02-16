import pickle
import pandas as pd
import numpy as np

class BehaviorPredictor:
    """Handles predictions using the trained jaguar behavior model."""
    
    def __init__(self, model_path='models/best_model.pkl'):
        """Load the trained model."""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
    def predict_behavior(self, movement_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict behavior states from movement data.
        
        Args:
            movement_data: DataFrame containing required features:
                - speed_mean
                - speed_max 
                - speed_std
                - distance_sum
                - distance_mean
                - direction_mean
                - direction_std
                - area_covered
                - movement_intensity
                - path_efficiency
                - direction_variability
                
        Returns:
            DataFrame with original data plus predicted behaviors
        """
        required_features = [
            'speed_mean', 'speed_max', 'speed_std',
            'distance_sum', 'distance_mean',
            'direction_mean', 'direction_std',
            'area_covered', 'movement_intensity',
            'path_efficiency', 'direction_variability'
        ]
        
        # Verify all required features are present
        missing_features = [f for f in required_features if f not in movement_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Make predictions
        predictions = self.model.predict(movement_data[required_features])
        probabilities = self.model.predict_proba(movement_data[required_features])
        
        # Add predictions to data
        result = movement_data.copy()
        result['predicted_state'] = predictions
        
        # Add probability for each state
        for i, state in enumerate(self.model.classes_):
            result[f'probability_{state}'] = probabilities[:, i]
            
        return result