import pickle
import pandas as pd
import numpy as np

class WeatherMovementPredictor:
    """Handles predictions using the trained weather movement model."""
    
    def __init__(self, model_path='models/weather_movement_model.pkl'):
        """Load the trained model."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
                
        except Exception as e:
            raise Exception(f"Error loading model from {model_path}: {str(e)}")
            
    def _generate_synthetic_movement_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic movement features based on weather characteristics.
        
        Args:
            weather_data: DataFrame containing weather features
        
        Returns:
            DataFrame with synthetic movement features added
        """
        data = weather_data.copy()
        
        # Synthetic speed calculation based on wind speed and temperature
        data['speed_mean'] = (data['wind_speed'] / 10) + np.abs(data['temperature'] - 25) / 10
        data['speed_max'] = data['speed_mean'] * 1.5
        data['speed_std'] = data['speed_mean'] * 0.3
        
        # Synthetic distance calculation based on weather conditions
        data['distance_sum'] = data['speed_mean'] * 2  # assume 2-hour window
        data['distance_mean'] = data['distance_sum'] / 4  # divide by number of points
        
        return data
    
    def predict_movement_state(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict movement states based on weather data.
        
        Args:
            weather_data: DataFrame containing weather features
        
        Returns:
            DataFrame with original data plus predicted movement states and probabilities
        """
        # Generate synthetic movement features
        data = self._generate_synthetic_movement_features(weather_data)
        
        # Select features
        X = data[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Add predictions to result
        result = data.copy()
        result['predicted_movement_state'] = predictions
        
        # Add probability for each state
        for i, state in enumerate(self.model.classes_):
            result[f'probability_{state}'] = probabilities[:, i]
        
        return result