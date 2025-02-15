import pandas as pd 
import numpy as np
from geopy.distance import geodesic

class FeatureEngineer:
    """
    Feature engineering for jaguar movement data.
    This class is designed to process jaguar movement data using GPS coordinates and timestamps. 
    It extracts meaningful features to analyze jaguar behavior, movement patterns, and state classification.
    """
    
    @staticmethod 
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        This method extracts useful time-based features from the timestamp column:

        -Hour, Day, Month, Year: Basic time components.
        - Day of the Week: Helps analyze movement patterns by weekday vs. weekend.
        - Time of Day: Categorizes into:
                - Night (0h - 6h)
                - Morning (6h - 12h)
                - Afternoon (12h - 18h)
                - Evening (18h - 24h)
        These features are useful for studying variations in jaguar activity at different times.
        """
        
        # Extract basic time components
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        
        # Categorize time of day into four periods
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening']
        )
        
        return df

    @staticmethod
    def calculate_movement_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        This method calculates various movement-related metrics for each jaguar:

        - Time Difference: Time elapsed since the last recorded point.
        - Time Difference in Hours: Converts it into hours for better analysis.
        - Distance (km): The geographic distance between consecutive GPS points using the Haversine formula.
        - Speed (km/h): Distance traveled divided by time difference.
        - Direction (degrees): Calculates the bearing between consecutive points.
        
        This provides insights into how jaguars move over time, including speed, distance traveled, and movement direction.
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Compute time difference between consecutive records per jaguar
        df['time_diff'] = df.groupby('individual_id')['timestamp'].diff()
        df['time_diff_hours'] = df['time_diff'].dt.total_seconds() / 3600

        # Initialize a list to store calculated distances
        distances = []
        
        # Compute distances between consecutive points for each jaguar
        for _, group in df.groupby('individual_id'):
            group_distances = [0]  # First point has no distance
            
            # Calculate distances between consecutive points
            for i in range(1, len(group)):
                prev_point = group.iloc[i-1]
                curr_point = group.iloc[i]
                
                # Calculate geodesic distance (in km) between consecutive GPS points
                distance = geodesic(
                    (prev_point['latitude'], prev_point['longitude']),
                    (curr_point['latitude'], curr_point['longitude'])
                ).kilometers
                
                group_distances.append(distance)
            
            distances.extend(group_distances)
        
        df['distance'] = distances
        
        # Compute speed (distance divided by time difference, avoiding division by zero)
        df['speed'] = df['distance'] / df['time_diff_hours'].replace({0: np.nan})
        
        # Compute movement direction (bearing between points)
        directions = []
        for _, group in df.groupby('individual_id'):
            group_directions = [np.nan]  # First point has no direction
            
            for i in range(1, len(group)):
                prev_point = group.iloc[i-1]
                curr_point = group.iloc[i]
                
                # Compute bearing using arctan2
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
        """
        This method aggregates movement data over a specific time window (default: 1 hour). For each window, it calculates:

        - Speed statistics: Mean, max, and standard deviation.
        - Distance statistics: Total and average distance traveled.
        - Direction statistics: Mean and standard deviation of movement direction.
        - Area Covered: Bounding box area of movement.
        - Movement Intensity: Speed x Total distance traveled (indicates activity level).
        - Path Efficiency: Ratio of total distance to straight-line distance.
        - Direction Variability: Measures how much the movement direction fluctuates.
        
        This helps in understanding movement patterns over different timeframes.
        """
        def calculate_window_stats(group):
            # Check if the window has enough data points; if not, return NaN values
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
                # Compute speed statistics
                'speed_mean': group['speed'].mean(),  # Average speed within the window
                'speed_max': group['speed'].max(),    # Maximum recorded speed
                'speed_std': group['speed'].std(),    # Speed variability
                
                # Compute distance statistics
                'distance_sum': group['distance'].sum(),  # Total distance traveled
                'distance_mean': group['distance'].mean(),  # Average distance between points
                
                # Compute direction statistics
                'direction_mean': group['direction'].mean(),  # Average movement direction
                'direction_std': group['direction'].std(),  # Direction variability
                
                # Calculate the area covered within the window using a bounding box approach
                'area_covered': np.sqrt(
                    (group['longitude'].max() - group['longitude'].min())**2 +
                    (group['latitude'].max() - group['latitude'].min())**2
                ),
                
                # Compute movement intensity: the product of mean speed and total distance
                'movement_intensity': group['speed'].mean() * group['distance'].sum(),
                
                # Compute path efficiency: ratio of total distance to straight-line distance
                'path_efficiency': group['distance'].sum() / (
                    np.sqrt(
                        (group['longitude'].iloc[-1] - group['longitude'].iloc[0])**2 +
                        (group['latitude'].iloc[-1] - group['latitude'].iloc[0])**2
                    ) or 1  # Avoid division by zero
                ),
                
                # Compute direction variability: standard deviation of direction changes
                'direction_variability': np.std(np.diff(group['direction'].dropna())) 
                if len(group['direction'].dropna()) > 1 else np.nan
            })

        windows = []
        for jaguar_id in df['individual_id'].unique():
            jaguar_data = df[df['individual_id'] == jaguar_id].copy()
            jaguar_data = jaguar_data.set_index('timestamp')
            
            # Resample data into specified time windows and compute statistics
            window_stats = jaguar_data.resample(window_size).apply(calculate_window_stats)
            window_stats['individual_id'] = jaguar_id
            windows.append(window_stats)
            
        return pd.concat(windows).reset_index()
    
    @staticmethod
    def classify_movement_state(df: pd.DataFrame) -> pd.DataFrame:
        """
        This method classifies jaguar movement into behavioral states based on:

        - Speed (low, medium, high)
        - Direction Variability (low, medium, high)
        - Movement states are:
            - Resting: Low speed, low direction change.
            - Foraging: Low-medium speed, high direction change.
            - Traveling: High speed, low direction change.
            - Exploring: Medium speed, medium direction change.
            
        These classifications help in studying jaguar behavior and identifying different movement strategies.
        """
        df = df.copy()
        
        # Define speed thresholds (km/h)
        speed_thresholds = { 'low': 2, 'medium': 5, 'high': 8 }
        
        # Define direction variability thresholds (degrees)
        direction_var_thresholds = { 'low': 30, 'medium': 60, 'high': 90 }
        
        def determine_state(row):
            speed = row['speed_mean']
            dir_var = row['direction_variability']
            
            if pd.isna(speed) or pd.isna(dir_var):
                return 'unknown' # Handle missing values
                
            if speed <= speed_thresholds['low']:
                return 'resting' if dir_var <= direction_var_thresholds['low'] else 'foraging'
            elif speed <= speed_thresholds['medium']:
                return 'exploring' if dir_var <= direction_var_thresholds['medium'] else 'foraging'
            else:
                return 'traveling' if dir_var <= direction_var_thresholds['low'] else 'exploring'
        
        df['movement_state'] = df.apply(determine_state, axis=1)
        return df
    
    