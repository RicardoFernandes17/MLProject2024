import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os

class MLPipeline:
    """
    Comprehensive ML pipeline with configurable parameters via JSON config.
    Supports multiple models, grid search, and cross-validation.
    """
    def __init__(self, config_path='ml_pipeline_config.json'):
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

# Example configuration file creation function
def create_default_config(output_path='ml_pipeline_config.json'):
    """
    Create a default configuration JSON file for the ML Pipeline.
    
    :param output_path: Path to save the configuration file
    """
    default_config = {
        "random_state": 42,
        "max_iter": 1000,
        "cv": 5,
        "scoring": "accuracy",
        "param_grids": {
            "random_forest": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2]
            },
            "svm": {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto"]
            },
            "logistic": {
                "classifier__C": [0.1, 1, 10],
                "classifier__solver": ["lbfgs", "liblinear"],
                "classifier__class_weight": [None, "balanced"]
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(default_config, f, indent=4)
    
    print(f"Default configuration saved to {output_path}")

# Usage example
if __name__ == "__main__":
    # Create a default config file if it doesn't exist
    if not os.path.exists('ml_pipeline_config.json'):
        create_default_config()
    
    # Initialize pipeline with config
    pipeline = MLPipeline('ml_pipeline_config.json')
    
    # Note: The rest of your data preprocessing and model training 
    # would go here (X_train, X_test, y_train, y_test)