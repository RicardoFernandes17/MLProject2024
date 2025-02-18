import pickle
import pandas as pd
import numpy as np

class WeatherMovementPredictor:
    def __init__(self, model_path='models/weather_movement_model.pkl'):
        """
        Load the trained weather movement model.
        
        Args:
            model_path (str): Path to the saved model file
        """
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
            weather_data (pd.DataFrame): Input weather data
        
        Returns:
            pd.DataFrame: Weather data with synthetic movement features added
        """
        # Use weather features to generate synthetic movement features
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
            weather_data (pd.DataFrame): DataFrame containing weather features
        
        Returns:
            pd.DataFrame: Original data with predicted movement states and probabilities
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

def predict_new_data(data_path):
    """
    Process new weather data and predict movement states.
    
    Args:
        data_path (str): Path to CSV file with weather data
    
    Returns:
        DataFrame with predictions
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Initialize predictor
    predictor = WeatherMovementPredictor()
    
    # Make predictions
    predictions = predictor.predict_movement_state(data)
    
    return predictions

def main():
    # Example usage
    predictions = predict_new_data("data/predict/new_weather_data.csv")
    
    # Save predictions
    output_path = "data/results/weather_movement_predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"\nFull predictions saved to: {output_path}")
    
    # Print some insights
    print("\nPrediction Summary:")
    print(predictions['predicted_movement_state'].value_counts())
    
    print("\nMovement State Probabilities:")
    for state in ['resting', 'foraging', 'traveling', 'exploring']:
        prob_col = f'probability_{state}'
        print(f"{state.capitalize()} - Mean Probability: {predictions[prob_col].mean():.2f}")

if __name__ == "__main__":
    main()