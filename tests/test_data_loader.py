import unittest
import pandas as pd
from pathlib import Path
from src.data.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_path = Path('data/raw')
        self.data_loader = DataLoader(
            self.data_path / 'jaguar_movement_data.csv',
            self.data_path / 'jaguar_additional_information.csv'
        )

    def test_load_data(self):
        """Test if data is loaded correctly"""
        data = self.data_loader.load_data()
        
        # Check if DataFrame is returned
        self.assertIsInstance(data, pd.DataFrame)
        
        # Check if required columns exist
        required_columns = ['longitude', 'latitude', 'individual_id', 'timestamp', 'sex', 'age', 'weight']
        for col in required_columns:
            self.assertIn(col, data.columns)
        
        # Check if timestamps are datetime
        self.assertEqual(data['timestamp'].dtype, 'datetime64[ns]')
        
        # Check if there are no null values in critical columns
        critical_columns = ['longitude', 'latitude', 'individual_id', 'timestamp']
        for col in critical_columns:
            self.assertFalse(data[col].isnull().any())

    def test_data_types(self):
        """Test if data types are correct"""
        data = self.data_loader.load_data()
        
        # Check numeric columns
        self.assertTrue(pd.api.types.is_float_dtype(data['longitude']))
        self.assertTrue(pd.api.types.is_float_dtype(data['latitude']))
        
        # Check categorical columns
        self.assertTrue(pd.api.types.is_string_dtype(data['sex']))
        
        # Check ID column
        self.assertTrue(pd.api.types.is_integer_dtype(data['individual_id']))

if __name__ == '__main__':
    unittest.main()