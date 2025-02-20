import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models import MLPipeline, JaguarLocationPredictor, WeatherMovementModel
from src.weather.weather_service import WeatherService
from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def train_lat_long_model(data):
    """Train the latitude/longitude prediction model."""
    print("\nTraining location predictor...")
    location_model = JaguarLocationPredictor()
    results = location_model.train(data)
    
    # Save model
    print("\nSaving model...")
    with open('models/location_predictor.pkl', 'wb') as f:
        pickle.dump(location_model, f)
    
    # Save feature importance plot
    print("Saving feature importance plot...")
    importance_plot = location_model.plot_feature_importance()
    importance_plot.savefig('models/location_feature_importance.png')
    plt.close()
    
    return location_model

def train_ml_pipeline(data):
    """Train the ML pipeline model."""
    print("\nML Pipeline training...")
    ml_pipeline = MLPipeline()
    results = ml_pipeline.train(data)
    
    # Save best model
    print("\nSaving best model...")
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(ml_pipeline.best_model, f)
    
    # Save feature importance plot
    print("Saving feature importance plot...")
    importance_plot = ml_pipeline.plot_feature_importance()
    if importance_plot:
        importance_plot.savefig('models/ml_feature_importance.png')
        plt.close()
    
    return ml_pipeline

def train_weather_predictor(data_loader, data):
    """Train the weather movement model."""
    print("\nWeather predictor training...")
    
    # Group data into villages
    movement_data_with_villages, _ = data_loader.get_villages(data, distance_km=10.0)
    
    # Add time features
    print("Adding time features...")
    movement_data_with_villages = FeatureEngineer.add_time_features(movement_data_with_villages)
    
    # Add movement features
    print("Calculating movement features...")
    movement_data_with_villages = FeatureEngineer.calculate_movement_features(movement_data_with_villages)
    
    # Create movement windows
    print("Creating movement windows...")
    window_data = FeatureEngineer.create_movement_windows(movement_data_with_villages)
    
    # Merge village information back into window data
    print("Adding village information to windows...")
    window_data = pd.merge(
        window_data,
        movement_data_with_villages[['timestamp', 'village_latitude', 'village_longitude']].drop_duplicates(),
        on='timestamp',
        how='left'
    )
    
    # Classify movement states
    print("Classifying movement states...")
    window_data = FeatureEngineer.classify_movement_state(window_data)
    
    # Clean data
    window_data = window_data.dropna()
    window_data = window_data[window_data['movement_state'] != 'unknown']

    # Extract date for both datasets
    window_data['date'] = pd.to_datetime(window_data['timestamp']).dt.date

    # Get unique locations by date
    print("Getting unique locations...")
    unique_locations = window_data[[
        'date', 'village_latitude', 'village_longitude'
    ]].drop_duplicates()
    
    # Add weather data
    print("Adding weather data...")
    weather_service = WeatherService()
    locations_with_weather = weather_service.add_weather_data(unique_locations)

    # Train weather model
    print("Training model...")
    weather_model = WeatherMovementModel()
    weather_model.train(window_data, locations_with_weather)
    
    # Save model
    print("\nSaving weather model...")
    model_data = weather_model.get_model_data()
    with open('models/weather_movement_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # Save feature importance plot
    print("Saving feature importance plot...")
    importance_plot = weather_model.plot_feature_importance()
    importance_plot.savefig('models/weather_feature_importance.png')
    plt.close()
    
    return weather_model

def main():
    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader(
        'data/raw/jaguar_movement_data.csv',
        'data/raw/jaguar_additional_information.csv',
        cache_dir='data/processed/cache'
    )
    
    # Load and preprocess data
    print("Loading data...")
    data = data_loader.load_data()
    
    # Train weather predictor
    weather_model = train_weather_predictor(data_loader, data)
    
    # Add time features
    print("\nAdding time features...")
    data = FeatureEngineer.add_time_features(data)
    
    # Train location predictor
    location_model = train_lat_long_model(data)
    
    # Add movement features
    print("\nCalculating movement features...")
    data = FeatureEngineer.calculate_movement_features(data)
    
    # Create movement windows
    print("\nCreating movement windows...")
    window_data = FeatureEngineer.create_movement_windows(data)
    
    # Classify movement states
    print("\nClassifying movement states...")
    window_data = FeatureEngineer.classify_movement_state(window_data)
    
    # Remove rows with unknown states or NaN values
    window_data = window_data.dropna()
    window_data = window_data[window_data['movement_state'] != 'unknown']
    
    # Train ML pipeline
    ml_pipeline = train_ml_pipeline(window_data)

    print("\nTraining complete! Models and visualizations saved in 'models' directory.")

if __name__ == "__main__":
    main()