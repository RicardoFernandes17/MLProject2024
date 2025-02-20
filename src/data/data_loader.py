import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN

class DataLoader:
    """
    A comprehensive data loading and preprocessing utility for jaguar movement analysis.
    
    Provides methods to:
    - Load and merge movement and individual jaguar datasets
    - Implement intelligent caching mechanism
    - Perform geographical clustering of locations
    
    Attributes:
        movement_data_path (Path): Path to movement data CSV
        info_data_path (Path): Path to individual information CSV
        cache_dir (Path): Directory for storing processed data
    """
    
    def __init__(self, movement_data_path: str, info_data_path: str, cache_dir: str = 'data/processed/cache'):
        self.movement_data_path = Path(movement_data_path)
        self.info_data_path = Path(info_data_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / 'merged_data_cache.csv'
    
    def _should_use_cache(self):
        """
        Determine if existing cached data is valid and up-to-date.
        
        Compares modification times of source files and cache file.
        
        Returns:
            bool: True if cache is valid, False otherwise
        """
        if not self.cache_path.exists():
            return False
            
        cache_mtime = self.cache_path.stat().st_mtime
        movement_mtime = self.movement_data_path.stat().st_mtime
        info_mtime = self.info_data_path.stat().st_mtime
        
        return cache_mtime > movement_mtime and cache_mtime > info_mtime
    
    def load_data(self):
        """
        Load and preprocess jaguar movement and individual datasets.
        
        Implements caching to improve performance:
        - Checks for existing cached data
        - Processes and merges data if cache is invalid
        - Standardizes column names
        
        Returns:
            pd.DataFrame: Merged and preprocessed dataset
        """
        if self._should_use_cache():
            print("Using cached data...")
            data = pd.read_csv(self.cache_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            return data
        
        print("Processing raw data...")
        
        movement_data = pd.read_csv(self.movement_data_path)
        info_data = pd.read_csv(self.info_data_path)
        
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
        
        movement_data['timestamp'] = pd.to_datetime(movement_data['timestamp'])
        
        merged_data = movement_data.merge(info_data, on='individual_id', how='left')
        
        merged_data.to_csv(self.cache_path, index=False)
        
        return merged_data
    
    def get_villages(self, data: pd.DataFrame, distance_km: float = 5.0):
        """
        Cluster geographical locations into villages using DBSCAN.
        
        Performs spatial clustering based on proximity:
        - Identifies unique location clusters
        - Calculates village centers
        - Assigns village identifiers to original dataset
        
        Args:
            data (pd.DataFrame): Input dataset with latitude and longitude
            distance_km (float): Maximum distance for clustering locations
        
        Returns:
            tuple: Enriched dataset and village centers information
        """
        unique_locations = data[['latitude', 'longitude']].drop_duplicates()
        
        eps_degrees = distance_km / 111.0 
        
        clustering = DBSCAN(
            eps=eps_degrees,
            min_samples=1,
            metric='euclidean'
        ).fit(unique_locations)
        
        unique_locations['village_id'] = clustering.labels_
        
        village_centers = unique_locations.groupby('village_id').agg({
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        village_sizes = unique_locations.groupby('village_id').size()
        village_centers['location_count'] = village_centers['village_id'].map(village_sizes)
        
        village_centers['village_name'] = 'Village_' + (village_centers['village_id'] + 1).astype(str)
        
        village_centers = village_centers.rename(columns={
            'latitude': 'village_latitude',
            'longitude': 'village_longitude'
        })
        
        location_to_village = unique_locations.set_index(['latitude', 'longitude'])['village_id']
        
        data_with_villages = data.copy()
        data_with_villages['village_id'] = data_with_villages.apply(
            lambda row: location_to_village.get((row['latitude'], row['longitude'])), 
            axis=1
        )
        data_with_villages['village_name'] = 'Village_' + (data_with_villages['village_id'] + 1).astype(str)
        
        data_with_villages = data_with_villages.merge(
            village_centers[['village_id', 'village_latitude', 'village_longitude']], 
            on='village_id', 
            how='left'
        )
        
        return data_with_villages, village_centers