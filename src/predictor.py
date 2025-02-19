import pickle
import pandas as pd
import numpy as np

import pickle
import pandas as pd
from sklearn.impute import SimpleImputer

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class PositionPredictor:
    """Handles predictions using the trained jaguar location model."""
    
    def __init__(self, model_path='models/location_predictor.pkl'):
        """Load the trained model."""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            # Initialize the imputer and scaler
            self.imputer = SimpleImputer(strategy='mean')
            self.scaler = StandardScaler()
                
        except Exception as e:
            raise Exception(f"Error loading model from {model_path}: {str(e)}")
            
    def predict_position(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict locations from input data.
        
        Args:
            data: DataFrame containing required features:
                - sex
                - age
                - weight
                - month
                
        Returns:
            DataFrame with original data plus predicted locations
        """
        required_features = ['sex', 'age', 'weight', 'month']
        
        # Verify all required features are present
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Create a copy of the data
        result = data.copy()
        
        # Convert sex to numeric if needed
        if result['sex'].dtype == 'O':  # Object dtype indicates categorical
            result['sex'] = result['sex'].map({'M': 0, 'F': 1})
        
        # Get feature data and handle NaN values
        X = result[required_features]
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        # Use the loaded model to make predictions
        lat_preds = self.model.lat_model.predict(X_scaled)
        lon_preds = self.model.lon_model.predict(X_scaled)
        
        # Add predictions to result
        result['predicted_latitude'] = lat_preds
        result['predicted_longitude'] = lon_preds
        
        # Calculate confidence scores
        for name, preds, model in [
            ('latitude', lat_preds, self.model.lat_model),
            ('longitude', lon_preds, self.model.lon_model)
        ]:
            importances = model.feature_importances_
            confidence_scores = []
            for _, row in X_scaled.iterrows():
                confidence = np.sum(importances * (1 - np.abs(row)))
                confidence_scores.append(min(1.0, max(0.0, confidence)))
            
            result[f'confidence_{name}'] = confidence_scores
            
        return result
    

class BehaviorPredictor:
    """Handles predictions using the trained jaguar behavior model."""
    
    def __init__(self, model_path='models/best_model.pkl'):
        """Load the trained model."""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            # Initialize the imputer here
            self.imputer = SimpleImputer(strategy='mean')
                
        except Exception as e:
            raise Exception(f"Error loading model from {model_path}: {str(e)}")
        
        
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
            
        # Create a copy of the data
        result = movement_data.copy()
        
        # Get feature data and handle NaN values
        X = result[required_features]
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Make predictions
        predictions = self.model.predict(X_imputed)
        probabilities = self.model.predict_proba(X_imputed)
        
        # Add predictions to result
        result['predicted_state'] = predictions
        
        # Add probability for each state
        for i, state in enumerate(self.model.classes_):
            result[f'probability_{state}'] = probabilities[:, i]
            
        return result

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