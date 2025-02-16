import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
import numpy as np
from src.data.feature_engineering import FeatureEngineer

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        """Create sample data for testing"""
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='H'),
            'longitude': np.linspace(-50, -49, 10),
            'latitude': np.linspace(-20, -19, 10),
            'individual_id': ['JAG_001'] * 10
        })

    def test_add_time_features(self):
        """Test if time features are added correctly"""
        result = FeatureEngineer.add_time_features(self.sample_data)
        
        # Check if time features exist
        time_features = ['hour', 'day', 'month', 'year', 'dayofweek', 'time_of_day']
        for feature in time_features:
            self.assertIn(feature, result.columns)
            
        # Check time_of_day categories
        self.assertEqual(
            set(result['time_of_day'].unique()),
            {'Night', 'Morning', 'Afternoon', 'Evening'}
        )

    def test_calculate_movement_features(self):
        """Test if movement features are calculated correctly"""
        result = FeatureEngineer.calculate_movement_features(self.sample_data)
        
        # Check if movement features exist
        movement_features = ['distance', 'speed', 'direction']
        for feature in movement_features:
            self.assertIn(feature, result.columns)
            
        # Check if first distance is 0
        self.assertEqual(result['distance'].iloc[0], 0)
        
        # Check if speeds are non-negative
        self.assertTrue((result['speed'].dropna() >= 0).all())

    def test_create_movement_windows(self):
        """Test movement window creation"""
        # First add required features
        data = FeatureEngineer.add_time_features(self.sample_data)
        data = FeatureEngineer.calculate_movement_features(data)
        
        # Create windows
        window_data = FeatureEngineer.create_movement_windows(data)
        
        # Check required columns
        required_features = [
            'speed_mean', 'speed_max', 'area_covered',
            'movement_intensity', 'path_efficiency'
        ]
        for feature in required_features:
            self.assertIn(feature, window_data.columns)

if __name__ == '__main__':
    unittest.main()