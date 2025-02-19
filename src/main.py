import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models import MLPipeline, JaguarLocationPredictor, WeatherMovementModel
from src.weather.weather_service import WeatherService
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def train_lat_long_model(data):
    data_copy = data.copy()
    model = JaguarLocationPredictor()
    model.train(data_copy)
    
    # Saves the trained model
    print("\nSaving model...")
    with open('models/location_predictor.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to models/location_predictor.pkl")
    
def train_ml_pipeline(data):
    
    data_copy = data.copy()
    # Define feature columns for model
    feature_cols = [
        'speed_mean', 'speed_max', 'speed_std',
        'distance_sum', 'distance_mean',
        'direction_mean', 'direction_std',
        'area_covered', 'movement_intensity',
        'path_efficiency', 'direction_variability'
    ]
    
    # Initialize the ML pipeline
    print("Initializing ML pipeline...")
    ml_pipeline = MLPipeline()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data_copy[feature_cols],
        data_copy['movement_state'],
        test_size=0.2,
        random_state=42,
        stratify=data_copy['movement_state']
    )
    
    # Train and evaluate models
    print("Training and evaluating models...")
    results = ml_pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Print results
    print("\nModel Results:")
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"Best parameters: {result['best_params']}")
        print(f"Best CV score: {result['best_score']:.4f}")
        print(f"Test score: {result['test_score']:.4f}")
        print(f"CV scores mean ± std: {result['cv_scores'].mean():.4f} ± {result['cv_scores'].std():.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    ml_pipeline.feature_names = feature_cols
    plt = ml_pipeline.plot_results(results)
    plt.savefig('models/model_comparison.png')
    
    # Save best model
    print("\nSaving best model...")
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(ml_pipeline.best_model, f)
        
def train_weather_predictor(data_loader, data):
    data_copy = data.copy()
    # Group data into villages
    movement_data_with_villages, village_centers = data_loader.get_villages(data_copy, distance_km=10.0)

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

def main():
    # Initializing the DataLoader class with the movement data and jaguar info dataset path's.
    data_loader = DataLoader(
        'data/raw/jaguar_movement_data.csv',
        'data/raw/jaguar_additional_information.csv',
        cache_dir='data/processed/cache'  # No trailing comma
    )
    
    # Loads and preprocess data
    print("Loading data...")
    data = data_loader.load_data()
    
    print("Weathe predictor training")
    train_weather_predictor(data_loader, data)
    
    # Add features (Time)
    print("Adding time features...")
    data = FeatureEngineer.add_time_features(data)
    
    # Training our latitute longitude model
    print("Latitude Longitude predictor training")
    train_lat_long_model(data)
    
    # Add features (Movement)
    print("Calculating movement features...")
    data = FeatureEngineer.calculate_movement_features(data)
    
    # Create movement windows
    print("Creating movement windows...")
    window_data = FeatureEngineer.create_movement_windows(data)
    
    # Classify movement states
    print("Classifying movement states...")
    window_data = FeatureEngineer.classify_movement_state(window_data)
    
    # Remove rows with unknown states or NaN values
    window_data = window_data.dropna()
    window_data = window_data[window_data['movement_state'] != 'unknown']
    
    # Training our latitute longitude model
    print("ML Pipeline training")
    train_ml_pipeline(window_data)

if __name__ == "__main__":
    main()
