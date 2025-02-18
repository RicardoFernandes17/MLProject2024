import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import DataLoader
from weather.weather_service import WeatherService
import pandas as pd
from pathlib import Path

def main():
    # Initialize services
    data_loader = DataLoader(
        'data/raw/jaguar_movement_data.csv',
        'data/raw/jaguar_additional_information.csv',
        cache_dir='data/processed/cache'
    )
    weather_service = WeatherService(cache_dir='data/weather_cache')
    
    # Load data
    print("Loading data...")
    data = data_loader.load_data()
    
    # Group into villages
    print("\nGrouping into villages...")    
    data_with_villages, village_centers = data_loader.get_villages(data, distance_km=10.0)
    
    # Get unique village locations by date
    print("\nGetting unique locations by date...")
    data_with_villages['date'] = data_with_villages['timestamp'].dt.date
    unique_locations = data_with_villages[['date', 'village_latitude', 'village_longitude']].groupby(
        ['date', 'village_latitude', 'village_longitude']
    ).first().reset_index()
    unique_locations = unique_locations.sort_values('date')
    
    # Add weather data
    print("\nAdding weather data...")
    locations_with_weather = weather_service.add_weather_data(unique_locations)
    
    # Save results
    print("\nSaving results...")
    output_dir = Path('data/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    locations_with_weather.to_csv(output_dir / 'village_weather.csv', index=False)
    print(f"Weather data saved to {output_dir / 'village_weather.csv'}")
    
     # Train weather movement model
    weather_model = WeatherMovementModel()
    X_train, X_test, y_train, y_test = weather_model.prepare_data(movement_data, locations_with_weather)
    
    # Train and evaluate model
    model, scaler = weather_model.train_model(X_train, X_test, y_train, y_test)
    
    # Analyze feature importance
    weather_model.analyze_weather_importance(model, weather_model.weather_features)

if __name__ == "__main__":
    main()