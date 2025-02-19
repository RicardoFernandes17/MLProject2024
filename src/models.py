
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class JaguarLocationPredictor:
    """
    Predicts jaguar locations (latitude and longitude) based on individual characteristics
    and temporal features.
    """
    def __init__(self, config_path='data/configs/location_config.json'):
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
        """Prepare features for the model."""
        data = data.copy()
        data['sex'] = data['sex'].map({'M': 0, 'F': 1})
        
        features = ['sex', 'age', 'weight', 'month']
        self.feature_names = features
        X = self.scaler.fit_transform(data[features])
        
        return X
        
    def train(self, data):
        """Train the location prediction models."""
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
        """Predict jaguar location based on individual characteristics."""
        input_data = pd.DataFrame({
            'sex': [sex],
            'age': [age],
            'weight': [weight],
            'month': [month]
        })
        
        X = self.prepare_features(input_data)
        predicted_lat = self.lat_model.predict(X)[0]
        predicted_lon = self.lon_model.predict(X)[0]
        
        return predicted_lat, predicted_lon
    
    def get_feature_importance(self):
        """Get feature importance for both models."""
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
        
    def plot_feature_importance(self):
        """Plot feature importance for both models."""
        importance_dict = self.get_feature_importance()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        importance_dict['latitude'].plot(kind='barh', ax=ax1)
        ax1.set_title('Latitude Feature Importance')
        
        importance_dict['longitude'].plot(kind='barh', ax=ax2)
        ax2.set_title('Longitude Feature Importance')
        
        plt.tight_layout()
        return plt
    
class MLPipeline:
    """
    ML pipeline with configurable parameters via JSON config.
    Supports multiple models, grid search, and cross-validation.
    """
    def __init__(self, config_path='data/configs/ml_pipeline_config.json'):
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        
        self.pipelines = {
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=config.get('random_state', 42)))
            ]),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True, random_state=config.get('random_state', 42)))
            ]),
            'logistic': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=config.get('max_iter', 1000), 
                                                random_state=config.get('random_state', 42)))
            ])
        }
        
        self.param_grids = config.get('param_grids', {
            'random_forest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            },
            'svm': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['rbf', 'linear'],
                'classifier__gamma': ['scale', 'auto']
            },
            'logistic': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['lbfgs', 'liblinear'],
                'classifier__class_weight': [None, 'balanced']
            }
        })
        
        self.cv = config.get('cv', 5)
        self.scoring = config.get('scoring', 'accuracy')
        self.best_model = None
        self.grid_searches = {}
        self.cv_results = {}
        self.feature_names = None

    def train(self, data):
        """Train all models using the provided data."""
        feature_cols = [
            'speed_mean', 'speed_max', 'speed_std',
            'distance_sum', 'distance_mean',
            'direction_mean', 'direction_std',
            'area_covered', 'movement_intensity',
            'path_efficiency', 'direction_variability'
        ]
        
        X = data[feature_cols]
        y = data['movement_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y
        )
        
        results = {}
        for name, pipeline in self.pipelines.items():
            print(f"\nTraining {name}...")
            
            grid_search = GridSearchCV(
                pipeline,
                self.param_grids[name],
                cv=self.cv,
                n_jobs=-1,
                scoring=self.scoring,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.grid_searches[name] = grid_search
            
            results[name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_score': grid_search.score(X_test, y_test),
                'cv_scores': cross_val_score(
                    grid_search.best_estimator_,
                    X_train, y_train,
                    cv=self.cv
                )
            }
        
        best_model_name = max(results.items(), key=lambda x: x[1]['test_score'])[0]
        self.best_model = self.grid_searches[best_model_name].best_estimator_
        self.feature_names = feature_cols
        
        print("\nModel Results:")
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print(f"Best parameters: {result['best_params']}")
            print(f"Best CV score: {result['best_score']:.4f}")
            print(f"Test score: {result['test_score']:.4f}")
            print(f"CV scores mean ± std: {result['cv_scores'].mean():.4f} ± {result['cv_scores'].std():.4f}")
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance for the best model."""
        if self.best_model is None:
            raise ValueError("Model must be trained before getting feature importance")
            
        if hasattr(self.best_model.named_steps['classifier'], 'feature_importances_'):
            importances = self.best_model.named_steps['classifier'].feature_importances_
            return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
        return None
        
    def plot_feature_importance(self):
        """Plot feature importance if available."""
        importance = self.get_feature_importance()
        if importance is not None:
            plt.figure(figsize=(12, 6))
            importance.plot(kind='barh')
            plt.title('Feature Importance')
            plt.tight_layout()
            return plt
        return None
class WeatherMovementModel:
    """Weather-based movement prediction model."""
    
    def __init__(self):
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
        self.model = None
        self.scaler = None
    
    def train(self, window_data, weather_data):
        """Train the weather movement model using the provided data."""
        print("Preparing data for model...")
        
        # Ensure date columns are datetime
        if 'timestamp' in window_data.columns:
            window_data['date'] = pd.to_datetime(window_data['timestamp']).dt.date
        weather_data['date'] = pd.to_datetime(weather_data['date']).dt.date

        # Ensure movement state column exists
        if 'movement_state' not in window_data.columns:
            from src.data.feature_engineering import FeatureEngineer
            window_data = FeatureEngineer.classify_movement_state(window_data)
        
        window_data = window_data[window_data['movement_state'] != 'unknown']

        # Merge data
        merged_data = pd.merge(
            window_data, 
            weather_data, 
            on=['date', 'village_latitude', 'village_longitude'],
            how='inner'
        )

        # Select features and target
        available_features = [f for f in self.feature_names if f in merged_data.columns]
        X = merged_data[available_features]
        y = merged_data['movement_state']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        
        return self.model, self.scaler
    
    def get_model_data(self):
        """Get the model data for saving."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained first")
            
        return {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
    
    def get_feature_importance(self):
        """Get feature importance for the trained model."""
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
            
        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
        
    def plot_feature_importance(self):
        """Plot feature importance for the trained model."""
        importance = self.get_feature_importance()
        
        plt.figure(figsize=(12, 6))
        plt.title("Weather and Movement Feature Importance")
        importance.plot(kind='barh')
        plt.tight_layout()
        return plt

    def predict(self, data):
        """
        Predict movement states for new data.
        
        Args:
            data: DataFrame containing required features
            
        Returns:
            DataFrame with predictions and probabilities
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained first")

        # Select available features
        available_features = [f for f in self.feature_names if f in data.columns]
        X = data[available_features]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Add predictions to result
        result = data.copy()
        result['predicted_state'] = predictions
        
        # Add probability for each state
        for i, state in enumerate(self.model.classes_):
            result[f'probability_{state}'] = probabilities[:, i]
            
        return result