import sys
import os
import pickle
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import DataLoader
from src.weather.weather_service import WeatherService
from src.predictor.weather_movement_model import WeatherMovementModel
from src.data.feature_engineering import FeatureEngineer
import pandas as pd
import numpy as np

def main():
    # Load movement data
    data_loader = DataLoader(
        'data/raw/jaguar_movement_data.csv',
        'data/raw/jaguar_additional_information.csv'
    )
    movement_data = data_loader.load_data()

    # Group data into villages
    movement_data_with_villages, village_centers = data_loader.get_villages(movement_data, distance_km=10.0)

    # Add date column
    movement_data_with_villages['date'] = movement_data_with_villages['timestamp'].dt.date

    # Load weather service
    weather_service = WeatherService()
    
    # Get unique locations by date
    unique_locations = movement_data_with_villages[['date', 'village_latitude', 'village_longitude']].groupby(
        ['date', 'village_latitude', 'village_longitude']
    ).first().reset_index()
    
    # Add weather data
    locations_with_weather = weather_service.add_weather_data(unique_locations)

    # Merge movement data with weather data
    merged_data = pd.merge(
        movement_data_with_villages, 
        locations_with_weather, 
        on=['date', 'village_latitude', 'village_longitude'],
        how='inner'
    )

    # Manually calculate movement features
    merged_data = FeatureEngineer.add_time_features(merged_data)
    
    merged_data = FeatureEngineer.calculate_movement_features(merged_data)

    # Create movement windows
    window_data = FeatureEngineer.create_movement_windows(merged_data)

    # Add village location information to window data
    window_data['date'] = window_data['timestamp'].dt.date
    window_data = pd.merge(
        window_data, 
        merged_data[['timestamp', 'village_latitude', 'village_longitude']].drop_duplicates(), 
        on='timestamp', 
        how='left'
    )

    # Train weather movement model
    weather_model = WeatherMovementModel()
    
    # Prepare and train model
    X_train, X_test, y_train, y_test = weather_model.prepare_data(
        window_data, 
        locations_with_weather
    )
    
    # Train and evaluate model
    model, scaler = weather_model.train_model(X_train, X_test, y_train, y_test)
    
    # Analyze feature importance
    weather_model.analyze_weather_importance(model, weather_model.feature_names)

    # Save the model
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save the model
    with open(models_dir / 'weather_movement_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_names': weather_model.feature_names
        }, f)
    
if __name__ == "__main__":
    main()