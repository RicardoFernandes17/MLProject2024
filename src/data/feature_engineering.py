import pandas as pd 
import numpy as np
from geopy.distance import geodesic

class FeatureEngineer:
    """Feature engineering for jaguar movement data."""
    
    @staticmethod 
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataset."""
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        
        # Create time period categories
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening']
        )
        
        return df

    @staticmethod
    def calculate_movement_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate movement-related features."""
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Time differences
        df['time_diff'] = df.groupby('individual_id')['timestamp'].diff()
        df['time_diff_hours'] = df['time_diff'].dt.total_seconds() / 3600

        # Calculate distances
        distances = []
        for _, group in df.groupby('individual_id'):
            group_distances = [0]  # First point has no distance
            
            # Calculate distances between consecutive points
            for i in range(1, len(group)):
                prev_point = group.iloc[i-1]
                curr_point = group.iloc[i]
                
                distance = geodesic(
                    (prev_point['latitude'], prev_point['longitude']),
                    (curr_point['latitude'], curr_point['longitude'])
                ).kilometers
                
                group_distances.append(distance)
            
            distances.extend(group_distances)
        
        df['distance'] = distances
        
        # Calculate speed
        df['speed'] = df['distance'] / df['time_diff_hours'].replace({0: np.nan})
        
        # Calculate direction
        directions = []
        for _, group in df.groupby('individual_id'):
            group_directions = [np.nan]  # First point has no direction
            
            for i in range(1, len(group)):
                prev_point = group.iloc[i-1]
                curr_point = group.iloc[i]
                
                direction = np.degrees(
                    np.arctan2(
                        curr_point['longitude'] - prev_point['longitude'],
                        curr_point['latitude'] - prev_point['latitude']
                    )
                )
                group_directions.append(direction)
            
            directions.extend(group_directions)
        
        df['direction'] = directions
        
        return df

    @staticmethod
    def create_movement_windows(df: pd.DataFrame, window_size: str = '1h') -> pd.DataFrame:
        """Create movement windows for behavioral analysis."""
        def calculate_window_stats(group):
            if len(group) < 2:  # Skip windows with insufficient data
                return pd.Series({
                    'speed_mean': np.nan,
                    'speed_max': np.nan,
                    'speed_std': np.nan,
                    'distance_sum': np.nan,
                    'distance_mean': np.nan,
                    'direction_mean': np.nan,
                    'direction_std': np.nan,
                    'area_covered': np.nan,
                    'movement_intensity': np.nan,
                    'path_efficiency': np.nan,
                    'direction_variability': np.nan
                })
                
            return pd.Series({
                'speed_mean': group['speed'].mean(),
                'speed_max': group['speed'].max(),
                'speed_std': group['speed'].std(),
                'distance_sum': group['distance'].sum(),
                'distance_mean': group['distance'].mean(),
                'direction_mean': group['direction'].mean(),
                'direction_std': group['direction'].std(),
                'area_covered': np.sqrt(
                    (group['longitude'].max() - group['longitude'].min())**2 +
                    (group['latitude'].max() - group['latitude'].min())**2
                ),
                'movement_intensity': group['speed'].mean() * group['distance'].sum(),
                'path_efficiency': group['distance'].sum() / (
                    np.sqrt(
                        (group['longitude'].iloc[-1] - group['longitude'].iloc[0])**2 +
                        (group['latitude'].iloc[-1] - group['latitude'].iloc[0])**2
                    ) or 1
                ),
                'direction_variability': np.std(np.diff(group['direction'].dropna())) if len(group['direction'].dropna()) > 1 else np.nan
            })

        windows = []
        for jaguar_id in df['individual_id'].unique():
            jaguar_data = df[df['individual_id'] == jaguar_id].copy()
            jaguar_data = jaguar_data.set_index('timestamp')
            
            # Resample and calculate stats
            window_stats = jaguar_data.resample(window_size).apply(calculate_window_stats)
            window_stats['individual_id'] = jaguar_id
            windows.append(window_stats)
            
        return pd.concat(windows).reset_index()
    
    @staticmethod
    def classify_movement_state(df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify movement states based on speed and direction.
        States:
        - 'resting': low speed, low direction change
        - 'foraging': low-medium speed, high direction change
        - 'traveling': high speed, low direction change
        - 'exploring': medium speed, medium direction change
        """
        df = df.copy()
        
        # Define thresholds (these can be adjusted based on domain knowledge)
        speed_thresholds = {
            'low': 2,    # km/h
            'medium': 5,  # km/h
            'high': 8     # km/h
        }
        
        direction_var_thresholds = {
            'low': 30,    # degrees
            'medium': 60,  # degrees
            'high': 90     # degrees
        }
        
        def determine_state(row):
            speed = row['speed_mean']
            dir_var = row['direction_variability']
            
            if pd.isna(speed) or pd.isna(dir_var):
                return 'unknown'
                
            if speed <= speed_thresholds['low']:
                if dir_var <= direction_var_thresholds['low']:
                    return 'resting'
                else:
                    return 'foraging'
            elif speed <= speed_thresholds['medium']:
                if dir_var <= direction_var_thresholds['medium']:
                    return 'exploring'
                else:
                    return 'foraging'
            else:
                if dir_var <= direction_var_thresholds['low']:
                    return 'traveling'
                else:
                    return 'exploring'
        
        df['movement_state'] = df.apply(determine_state, axis=1)
        return df