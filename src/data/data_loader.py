import pandas as pd
import numpy as np

class DataLoader:
    """Data loader for jaguar movement analysis."""
    
    def __init__(self, movement_data_path: str, info_data_path: str):
        """ Initialize data loader with file paths.
        
        Args:
            movement_data_path: Path to jaguar movement data CSV file
            info_data_path: Path to jaguar info CSV file
        """
        self.movement_data_path = movement_data_path
        self.info_data_path = info_data_path
    
    def load_data(self):
        """
        Loads and merges datasets.
        """
        
        # Loads the jaguar data from CSV file
        movement_data = pd.read_csv(self.movement_data_path)
        
        # Loads the jaguar info data from CSV file
        info_data = pd.read_csv(self.info_data_path)
        
        # Rename columns in jaguar data for clarity
        # We standardize the individual ID column name
        movement_data.rename(columns={
            'location.long': 'longitude',
            'location.lat': 'latitude',
            'individual.local.identifier (ID)': 'individual_id'
        }, inplace=True)
                
        # Rename columns in jaguar info data for consistency
        # We also standardize the individual ID column name
        info_data.rename(columns={
            'ID': 'individual_id',
            'Sex': 'sex',
            'Estimated Age': 'age',
            'Weight': 'weight'
        }, inplace=True)
        
        # Drop unnecessary columns from jaguar data
        movement_data.drop([
            'Event_ID',
            'individual.taxon.canonical.name',
            'tag.local.identifier',
            'study.name',
            'country'
        ], axis=1, inplace=True)
        
        # Drop unnecessary columns from jaguar info data
        info_data.drop([
            'Collar Type',
            'Collar Brand',
            'Planned Schedule',
            'Project Leader',
            'Contact'
        ], axis=1, inplace=True)
        
        # Convert timestamp column to datetime format for easier time-based analysis
        movement_data['timestamp'] = pd.to_datetime(movement_data['timestamp'])
        
        # Merge movement data with jaguar info data using individual_id as key
        merged_data = movement_data.merge(info_data, on='individual_id', how='left')
        
        return merged_data
