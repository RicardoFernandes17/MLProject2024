
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
    


class MLPipeline:
    """
    Comprehensive ML pipeline with configurable parameters via JSON config.
    Supports multiple models, grid search, and cross-validation.
    """
    def __init__(self, config_path='data/configs/ml_pipeline_config.json'):
        """
        Initialize MLPipeline with configuration from a JSON file.
        
        :param config_path: Path to the JSON configuration file
        """
        # Load configuration
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        
        # Define the pipelines with different models
        self.pipelines = {
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=config.get('random_state', 42)))
            ]),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    probability=True, 
                    random_state=config.get('random_state', 42)
                ))
            ]),
            'logistic': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    max_iter=config.get('max_iter', 1000), 
                    random_state=config.get('random_state', 42)
                ))
            ])
        }
        
        # Define parameter grids for each model from config
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
        
        # Other configuration parameters
        self.cv = config.get('cv', 5)
        self.scoring = config.get('scoring', 'accuracy')
        
        self.best_model = None
        self.grid_searches = {}
        self.cv_results = {}
        self.feature_names = None
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train all models, perform grid search, and evaluate results.
        """
        results = {}
        
        for name, pipeline in self.pipelines.items():
            print(f"\nTraining {name}...")
            
            # Perform grid search
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
            
            # Store results
            results[name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_score': grid_search.score(X_test, y_test),
                'cv_scores': cross_val_score(
                    grid_search.best_estimator_,
                    X_train,
                    y_train,
                    cv=self.cv
                )
            }
            
        # Find best model
        best_model_name = max(results.items(), key=lambda x: x[1]['test_score'])[0]
        self.best_model = self.grid_searches[best_model_name].best_estimator_
        
        return results
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance if available for the best model.
        """
        self.feature_names = feature_names
        if hasattr(self.best_model.named_steps['classifier'], 'feature_importances_'):
            importances = self.best_model.named_steps['classifier'].feature_importances_
            return pd.Series(importances, index=feature_names).sort_values(ascending=False)
        return None
    
    def plot_results(self, results):
        """
        Plot comparison of model performances.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Model Comparison
        plt.subplot(2, 2, 1)
        model_scores = {name: result['test_score'] for name, result in results.items()}
        plt.bar(model_scores.keys(), model_scores.values())
        plt.title('Model Test Scores Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Accuracy')
        
        # Plot 2: Cross-validation Scores
        plt.subplot(2, 2, 2)
        cv_means = [result['cv_scores'].mean() for result in results.values()]
        cv_stds = [result['cv_scores'].std() for result in results.values()]
        plt.errorbar(list(results.keys()), cv_means, yerr=cv_stds, fmt='o')
        plt.title('Cross-validation Scores with Standard Deviation')
        plt.xticks(rotation=45)
        plt.ylabel('CV Score')
        
        # Plot 3: Feature Importance (if available)
        plt.subplot(2, 2, 3)
        if self.feature_names:
            feature_importance = self.get_feature_importance(self.feature_names)
            if feature_importance is not None:
                feature_importance.plot(kind='barh')
                plt.title('Feature Importance')
        
        plt.tight_layout()
        return plt

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