import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from src.pipeline.ml_pipeline import MLPipeline
from sklearn.model_selection import train_test_split

class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test data and pipeline instance."""
        # Create synthetic data for testing
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=4,  # Matching our movement states
            n_informative=5,
            random_state=42
        )
        
        # Convert to pandas DataFrame with meaningful feature names
        self.X = pd.DataFrame(
            X,
            columns=[f'feature_{i}' for i in range(X.shape[1])]
        )
        self.y = pd.Series(y)
        
        # Initialize pipeline
        self.pipeline = MLPipeline()

    def test_pipeline_initialization(self):
        """Test if pipeline is properly initialized with all models."""
        expected_models = {'random_forest', 'svm', 'logistic'}
        self.assertEqual(
            set(self.pipeline.pipelines.keys()), 
            expected_models
        )

    def test_model_training(self):
        """Test if models can be trained successfully."""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=0.2, 
            random_state=42
        )
        
        # Train and evaluate
        results = self.pipeline.train_and_evaluate(
            X_train, 
            X_test, 
            y_train, 
            y_test,
            cv=2  # Using 2 folds for faster testing
        )
        
        # Check results structure
        for model_name in self.pipeline.pipelines.keys():
            self.assertIn(model_name, results)
            self.assertIn('best_params', results[model_name])
            self.assertIn('best_score', results[model_name])
            self.assertIn('test_score', results[model_name])
            self.assertIn('cv_scores', results[model_name])

    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Train pipeline first
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=0.2, 
            random_state=42
        )
        
        self.pipeline.train_and_evaluate(
            X_train, 
            X_test, 
            y_train, 
            y_test,
            cv=2
        )
        
        # Get feature importance
        feature_importance = self.pipeline.get_feature_importance(self.X.columns)
        
        # Check if feature importance exists
        self.assertIsNotNone(feature_importance)
        self.assertEqual(len(feature_importance), len(self.X.columns))

    def test_model_prediction(self):
        """Test model prediction capabilities."""
        # Train pipeline
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=0.2, 
            random_state=42
        )
        
        self.pipeline.train_and_evaluate(
            X_train, 
            X_test, 
            y_train, 
            y_test,
            cv=2
        )
        
        # Test predictions
        predictions = self.pipeline.best_model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
        self.assertTrue(all(isinstance(pred, (np.int64, np.int32)) for pred in predictions))

    def test_parameter_grid_structure(self):
        """Test if parameter grids are properly structured."""
        for model_name, param_grid in self.pipeline.param_grids.items():
            # Check if param grid exists for each model
            self.assertIn(model_name, self.pipeline.pipelines)
            
            # Check parameter grid structure
            for param_name, param_values in param_grid.items():
                # Parameter names should include classifier prefix
                self.assertTrue(param_name.startswith('classifier__'))
                # Parameter values should be a list or similar
                self.assertTrue(hasattr(param_values, '__iter__'))

    def test_cross_validation_scores(self):
        """Test if cross-validation scores are properly calculated."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=0.2, 
            random_state=42
        )
        
        results = self.pipeline.train_and_evaluate(
            X_train, 
            X_test, 
            y_train, 
            y_test,
            cv=3
        )
        
        for model_results in results.values():
            cv_scores = model_results['cv_scores']
            self.assertEqual(len(cv_scores), 3)  # 3-fold CV
            self.assertTrue(all(0 <= score <= 1 for score in cv_scores))

if __name__ == '__main__':
    unittest.main()