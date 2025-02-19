import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class JaguarLocationPredictor:
    """
    Predicts jaguar locations (latitude and longitude) based on individual characteristics
    and temporal features.
    """
    def __init__(self, config_path='data/configs/location_config.json'):
        """
        Initialize predictor with configuration from JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        # Default configuration
        default_config = {
            'random_state': 42,
            'n_estimators': 200,
            'max_depth': 20,
            'cv': 5
        }
        
        # Load configuration if file exists, otherwise use defaults
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
        except FileNotFoundError:
            config = default_config
            
        # Initialize models with configuration
        self.lat_model = RandomForestRegressor(
            n_estimators=config.get('n_estimators', 200),
            max_depth=config.get('max_depth', 20),
            random_state=config.get('random_state', 42)
        )
        
        self.lon_model = RandomForestRegressor(
            n_estimators=config.get('n_estimators', 200),
            max_depth=config.get('max_depth', 20),
            random_state=config.get('random_state', 42)
        )
        
        self.scaler = StandardScaler()
        self.cv = config.get('cv', 5)
        self.feature_names = None
        
    def prepare_features(self, data):
        """
        Prepare features for the model.
        
        Args:
            data: DataFrame with required columns (sex, age, weight, month)
            
        Returns:
            Scaled features array
        """
        # Convert sex to numeric
        data = data.copy()
        data['sex'] = data['sex'].map({'M': 0, 'F': 1})
        
        # Select and scale features
        features = ['sex', 'age', 'weight', 'month']
        self.feature_names = features
        X = self.scaler.fit_transform(data[features])
        
        return X
        
    def train(self, data):
        """
        Train the location prediction models.
        
        Args:
            data: DataFrame containing training data with required columns
        """
        # Prepare features
        X = self.prepare_features(data)
        y_lat = data['latitude'].values
        y_lon = data['longitude'].values
        
        # Split data
        X_train, X_test, y_lat_train, y_lat_test, y_lon_train, y_lon_test = train_test_split(
            X, y_lat, y_lon, test_size=0.2, random_state=42
        )
        
        # Train latitude model
        print("Training latitude model...")
        self.lat_model.fit(X_train, y_lat_train)
        lat_score = self.lat_model.score(X_test, y_lat_test)
        print(f"Latitude model R² score: {lat_score:.4f}")
        
        # Train longitude model
        print("Training longitude model...")
        self.lon_model.fit(X_train, y_lon_train)
        lon_score = self.lon_model.score(X_test, y_lon_test)
        print(f"Longitude model R² score: {lon_score:.4f}")
        
        # Perform cross-validation
        lat_cv_scores = cross_val_score(self.lat_model, X, y_lat, cv=self.cv)
        lon_cv_scores = cross_val_score(self.lon_model, X, y_lon, cv=self.cv)
        
        return {
            'latitude': {
                'test_score': lat_score,
                'cv_scores': lat_cv_scores
            },
            'longitude': {
                'test_score': lon_score,
                'cv_scores': lon_cv_scores
            }
        }
    
    def predict(self, sex, age, weight, month):
        """
        Predict jaguar location based on individual characteristics and month.
        
        Args:
            sex: 'M' or 'F'
            age: Age in years
            weight: Weight in kg
            month: Month (1-12)
            
        Returns:
            Tuple of (predicted_latitude, predicted_longitude)
        """
        # Prepare input data
        input_data = pd.DataFrame({
            'sex': [sex],
            'age': [age],
            'weight': [weight],
            'month': [month]
        })
        
        # Scale features
        X = self.prepare_features(input_data)
        
        # Make predictions
        predicted_lat = self.lat_model.predict(X)[0]
        predicted_lon = self.lon_model.predict(X)[0]
        
        return predicted_lat, predicted_lon
    
    def get_feature_importance(self):
        """
        Get feature importance for both latitude and longitude models.
        
        Returns:
            Dictionary with feature importance for both models
        """
        if self.feature_names is None:
            raise ValueError("Model must be trained before getting feature importance")
            
        return {
            'latitude': pd.Series(
                self.lat_model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False),
            'longitude': pd.Series(
                self.lon_model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
        }
    
    def plot_results(self, results):
        """
        Plot model performance and feature importance.
        
        Args:
            results: Dictionary containing model results
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot test scores
        test_scores = [results[k]['test_score'] for k in ['latitude', 'longitude']]
        axes[0, 0].bar(['Latitude', 'Longitude'], test_scores)
        axes[0, 0].set_title('Model Test Scores')
        axes[0, 0].set_ylabel('R² Score')
        
        # Plot CV scores
        cv_means = [results[k]['cv_scores'].mean() for k in ['latitude', 'longitude']]
        cv_stds = [results[k]['cv_scores'].std() for k in ['latitude', 'longitude']]
        axes[0, 1].errorbar([0, 1], cv_means, yerr=cv_stds, fmt='o')
        axes[0, 1].set_xticks([0, 1])
        axes[0, 1].set_xticklabels(['Latitude', 'Longitude'])
        axes[0, 1].set_title('Cross-validation Scores')
        
        # Plot feature importance
        importance_dict = self.get_feature_importance()
        importance_dict['latitude'].plot(kind='barh', ax=axes[1, 0])
        axes[1, 0].set_title('Latitude Feature Importance')
        importance_dict['longitude'].plot(kind='barh', ax=axes[1, 1])
        axes[1, 1].set_title('Longitude Feature Importance')
        
        plt.tight_layout()
        return plt