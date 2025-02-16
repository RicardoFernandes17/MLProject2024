from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

class MLPipeline:
    """
    Comprehensive ML pipeline for jaguar behavior analysis.
    Includes multiple models, grid search, and cross-validation.
    """
    def __init__(self):
        # Define the pipelines with different models
        self.pipelines = {
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True, random_state=42))
            ]),
            
            'logistic': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42))
            ])
        }
        
        # Define parameter grids for each model
        self.param_grids = {
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
        }
        
        self.best_model = None
        self.grid_searches = {}
        self.cv_results = {}
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, cv=5):
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
                cv=cv,
                n_jobs=-1,
                scoring='accuracy',
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
                    cv=cv
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
        if hasattr(self.best_model.named_steps['classifier'], 'feature_importances_'):
            importances = self.best_model.named_steps['classifier'].feature_importances_
            return pd.Series(importances, index=feature_names).sort_values(ascending=False)
        return None
    
    def plot_results(self, results):
        """
        Plot comparison of model performances.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
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
        plt.errorbar(results.keys(), cv_means, yerr=cv_stds, fmt='o')
        plt.title('Cross-validation Scores with Standard Deviation')
        plt.xticks(rotation=45)
        plt.ylabel('CV Score')
        
        # Plot 3: Feature Importance (if available)
        plt.subplot(2, 2, 3)
        feature_importance = self.get_feature_importance(self.feature_names)
        if feature_importance is not None:
            feature_importance.plot(kind='barh')
            plt.title('Feature Importance')
        
        plt.tight_layout()
        return plt