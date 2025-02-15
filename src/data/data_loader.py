import pandas as pd
import numpy as np

class DataLoader:
    """Data loader for jaguar movement analysis."""
    
    def __init__(self, movement_data_path: str, info_data_path: str):
        """Initialize data loader with file paths.
        
        Args:
            movement_data_path: Path to movement CSV file
            info_data_path: Path to jaguar info CSV file
        """
        self.movement_data_path = movement_data_path
        self.info_data_path = info_data_path
    
    def load_data(self):
        """Load and merge datasets."""
        movement_data = pd.read_csv(self.movement_data_path)
        info_data = pd.read_csv(self.info_data_path)
        
        # Rename columns for clarity
        movement_data.rename(columns={
            'location.long': 'longitude',
            'location.lat': 'latitude',
            'individual.local.identifier (ID)': 'individual_id'
        }, inplace=True)
        
        info_data.rename(columns={
            'ID': 'individual_id',
            'Sex': 'sex',
            'Estimated Age': 'age',
            'Weight': 'weight'
        }, inplace=True)
        
        # Drop unnecessary columns
        movement_data.drop([
            'Event_ID',
            'individual.taxon.canonical.name',
            'tag.local.identifier',
            'study.name',
            'country'
        ], axis=1, inplace=True)
        
        info_data.drop([
            'Collar Type',
            'Collar Brand',
            'Planned Schedule',
            'Project Leader',
            'Contact'
        ], axis=1, inplace=True)
        
        # Convert timestamp
        movement_data['timestamp'] = pd.to_datetime(movement_data['timestamp'])
        
        # Merge datasets
        merged_data = movement_data.merge(info_data, on='individual_id', how='left')
        
        return merged_data
