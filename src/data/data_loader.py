import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN

class DataLoader:
    """Data loader for jaguar movement analysis with CSV caching."""
    
    def __init__(self, movement_data_path: str, info_data_path: str, cache_dir: str = 'data/processed/cache'):
        """Initialize data loader with file paths."""
        self.movement_data_path = Path(movement_data_path)
        self.info_data_path = Path(info_data_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / 'merged_data_cache.csv'
    
    def _should_use_cache(self) -> bool:
        """Check if cache is valid and up to date."""
        if not self.cache_path.exists():
            return False
            
        # Get modification times
        cache_mtime = self.cache_path.stat().st_mtime
        movement_mtime = self.movement_data_path.stat().st_mtime
        info_mtime = self.info_data_path.stat().st_mtime
        
        # Cache is valid if it's newer than both input files
        return cache_mtime > movement_mtime and cache_mtime > info_mtime
    
    def load_data(self) -> pd.DataFrame:
        """Load and merge data, using cache if available."""
        # Check if we can use cached data
        if self._should_use_cache():
            print("Using cached data...")
            data = pd.read_csv(self.cache_path)
            # Convert timestamp back to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            return data
        
        print("Processing raw data...")
        
        # Load data
        movement_data = pd.read_csv(self.movement_data_path)
        info_data = pd.read_csv(self.info_data_path)
        
        # Basic preprocessing
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
        
        # Convert timestamp
        movement_data['timestamp'] = pd.to_datetime(movement_data['timestamp'])
        
        # Merge data
        merged_data = movement_data.merge(info_data, on='individual_id', how='left')
        
        # Save to cache
        merged_data.to_csv(self.cache_path, index=False)
        
        return merged_data
    
    def get_villages(self, data: pd.DataFrame, distance_km: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Group locations into villages based on proximity and add village IDs to data.
        
        Args:
            data: DataFrame with latitude and longitude
            distance_km: Maximum distance (in km) between points in same village
            
        Returns:
            Tuple containing:
            - Original DataFrame with village information added
            - DataFrame with village centers information
        """
        # Get unique locations for clustering
        unique_locations = data[['latitude', 'longitude']].drop_duplicates()
        
        # Convert km to degrees (rough approximation)
        eps_degrees = distance_km / 111.0  # 1 degree ≈ 111 km
        
        # Perform clustering
        clustering = DBSCAN(
            eps=eps_degrees,
            min_samples=1,
            metric='euclidean'
        ).fit(unique_locations)
        
        # Add cluster labels to unique locations
        unique_locations['village_id'] = clustering.labels_
        
        # Calculate village centers
        village_centers = unique_locations.groupby('village_id').agg({
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        # Add number of points in each village
        village_sizes = unique_locations.groupby('village_id').size()
        village_centers['location_count'] = village_centers['village_id'].map(village_sizes)
        
        # Add village names
        village_centers['village_name'] = 'Village_' + (village_centers['village_id'] + 1).astype(str)
        
        # Rename center coordinates to be clear
        village_centers = village_centers.rename(columns={
            'latitude': 'village_latitude',
            'longitude': 'village_longitude'
        })
        
        # Create mapping dictionary for merging
        location_to_village = unique_locations.set_index(['latitude', 'longitude'])['village_id']
        
        # Add village information to original data
        data_with_villages = data.copy()
        data_with_villages['village_id'] = data_with_villages.apply(
            lambda row: location_to_village.get((row['latitude'], row['longitude'])), 
            axis=1
        )
        data_with_villages['village_name'] = 'Village_' + (data_with_villages['village_id'] + 1).astype(str)
        
        # Add village center coordinates
        data_with_villages = data_with_villages.merge(
            village_centers[['village_id', 'village_latitude', 'village_longitude']], 
            on='village_id', 
            how='left'
        )
        
        return data_with_villages, village_centers