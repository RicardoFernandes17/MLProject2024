import unittest
import numpy as np
from src.models.behavior_classifier import JaguarBehaviorClassifier

class TestBehaviorClassifier(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.classifier = JaguarBehaviorClassifier()
        
        # Create sample features
        np.random.seed(42)
        self.X = np.random.rand(100, 11)  # 11 features
        self.y = np.random.choice(['resting', 'foraging', 'traveling', 'exploring'], 100)

    def test_fit_predict(self):
        """Test if model can fit and predict"""
        self.classifier.fit(self.X, self.y)
        predictions = self.classifier.predict(self.X)
        
        # Check predictions shape and values
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(set(predictions).issubset(set(self.y)))
        
    def test_predict_proba(self):
        """Test probability predictions"""
        self.classifier.fit(self.X, self.y)
        probabilities = self.classifier.predict_proba(self.X)
        
        # Check probability shape and values
        self.assertEqual(probabilities.shape[0], len(self.y))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertTrue((probabilities >= 0).all() and (probabilities <= 1).all())

    def test_invalid_input(self):
        """Test model behavior with invalid input"""
        # Test with wrong number of features
        with self.assertRaises(ValueError):
            self.classifier.predict(np.random.rand(10, 5))  # Wrong number of features

if __name__ == '__main__':
    unittest.main()