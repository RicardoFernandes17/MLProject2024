import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class WeatherMovementModel:
    def __init__(self):
        # Prepare features
        self.feature_names = [
            # Weather features
            'temperature', 
            'humidity', 
            'wind_speed', 
            'wind_direction', 
            'precipitation', 
            'cloud_cover',
            
            # Movement features
            'speed_mean', 
            'speed_max', 
            'speed_std', 
            'distance_sum', 
            'distance_mean'
        ]
    
    def prepare_data(self, window_data, weather_data):
        print("Preparing data for model...")
        
        # Print columns and first few rows of window_data and weather_data
        print("\nWindow Data Columns:", window_data.columns.tolist())
        print("Window Data First Row:", window_data.iloc[0].to_dict())
        
        print("\nWeather Data Columns:", weather_data.columns.tolist())
        print("Weather Data First Row:", weather_data.iloc[0].to_dict())

        # Ensure date columns are datetime
        if 'timestamp' in window_data.columns:
            window_data['date'] = pd.to_datetime(window_data['timestamp']).dt.date
        
        # Ensure weather data has date as datetime
        weather_data['date'] = pd.to_datetime(weather_data['date']).dt.date

        # Ensure movement state column exists
        if 'movement_state' not in window_data.columns:
            from src.data.feature_engineering import FeatureEngineer
            window_data = FeatureEngineer.classify_movement_state(window_data)
        
        # Remove rows with unknown movement state
        window_data = window_data[window_data['movement_state'] != 'unknown']

        print("\nMerging window data with weather data...")
        try:
            # Merge window data with weather data
            merged_data = pd.merge(
                window_data, 
                weather_data, 
                on=['date', 'village_latitude', 'village_longitude'],
                how='inner'
            )
        except Exception as e:
            print("\nMerge Error:", str(e))
            
            # Debug merge keys
            print("\nWindow Data Unique Dates:", window_data['date'].nunique())
            print("Window Data Unique Latitude:", window_data['village_latitude'].nunique())
            print("Window Data Unique Longitude:", window_data['village_longitude'].nunique())
            
            print("\nWeather Data Unique Dates:", weather_data['date'].nunique())
            print("Weather Data Unique Latitude:", weather_data['village_latitude'].nunique())
            print("Weather Data Unique Longitude:", weather_data['village_longitude'].nunique())
            
            raise

        print("\nMerged Data Columns:", merged_data.columns.tolist())
        print("Merged Data Shape:", merged_data.shape)

        # Select available features
        available_features = [f for f in self.feature_names if f in merged_data.columns]
        print("\nAvailable Features:", available_features)

        # If not enough features, modify the feature selection
        if len(available_features) < len(self.feature_names):
            print("\nWarning: Not all expected features are available.")
            print("Using available features:", available_features)

        X = merged_data[available_features]
        y = merged_data['movement_state']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    def train_model(self, X_train, X_test, y_train, y_test):
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_accuracy = model.score(X_train_scaled, y_train)
        test_accuracy = model.score(X_test_scaled, y_test)
        
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        
        return model, scaler
    
    def analyze_weather_importance(self, model, feature_names):
        # Create feature importance plot
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12,6))
        plt.title("Weather and Movement Feature Importance for Jaguar Movement")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                   [feature_names[i] for i in indices], 
                   rotation=45, 
                   ha='right')
        plt.tight_layout()
        plt.show()

        # Print feature importances
        feature_importance = pd.Series(
            importances, 
            index=feature_names
        ).sort_values(ascending=False)
        print("\nFeature Importances:")
        print(feature_importance)