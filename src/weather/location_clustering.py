import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Tuple, List
from datetime import datetime, timedelta

class LocationClusterer:
    """Clusters GPS coordinates into geographical regions and time windows."""
    
    def __init__(self, 
                 eps_km: float = 10.0, 
                 min_samples: int = 1,
                 time_window_hours: int = 3):
        """
        Initialize the clusterer.
        
        Args:
            eps_km: Maximum distance (in kilometers) between points in a cluster
            min_samples: Minimum number of points to form a cluster
            time_window_hours: Size of time window for grouping
        """
        self.eps_degrees = eps_km / 111.0  # 1 degree ≈ 111 km
        self.min_samples = min_samples
        self.time_window_hours = time_window_hours
        
    def _round_time(self, dt: datetime) -> datetime:
        """Round time to nearest time window."""
        hours_to_round = self.time_window_hours
        return dt.replace(
            hour=(dt.hour // hours_to_round) * hours_to_round,
            minute=0,
            second=0,
            microsecond=0
        )
        
    def cluster_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cluster locations and timestamps.
        
        Args:
            df: DataFrame with 'latitude', 'longitude', and 'timestamp' columns
            
        Returns:
            DataFrame with cluster information
        """
        # Create copy of dataframe
        df = df.copy()
        
        # Round timestamps to time windows
        df['time_window'] = df['timestamp'].apply(self._round_time)
        
        # Get unique lat-lon pairs
        unique_locations = df[['latitude', 'longitude']].drop_duplicates()
        
        # Perform spatial clustering
        clustering = DBSCAN(
            eps=self.eps_degrees,
            min_samples=self.min_samples,
            metric='euclidean'
        ).fit(unique_locations)
        
        # Create mapping of lat-lon to cluster
        location_to_cluster = pd.DataFrame({
            'latitude': unique_locations['latitude'],
            'longitude': unique_locations['longitude'],
            'spatial_cluster': clustering.labels_
        })
        
        # Merge cluster labels back to original DataFrame
        df_with_clusters = df.merge(
            location_to_cluster,
            on=['latitude', 'longitude'],
            how='left'
        )
        
        # Calculate cluster centers
        cluster_centers = df_with_clusters.groupby('spatial_cluster').agg({
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        cluster_centers.columns = ['spatial_cluster', 'cluster_lat', 'cluster_lon']
        
        # Add cluster centers to DataFrame
        df_with_clusters = df_with_clusters.merge(
            cluster_centers,
            on='spatial_cluster',
            how='left'
        )
        
        # Create unique spatiotemporal cluster ID
        df_with_clusters['cluster_id'] = (
            df_with_clusters['spatial_cluster'].astype(str) + '_' + 
            df_with_clusters['time_window'].dt.strftime('%Y%m%d_%H')
        )
        
        return df_with_clusters
    
    @staticmethod
    def get_unique_timepoints(df: pd.DataFrame) -> List[Tuple]:
        """
        Get unique cluster-time combinations.
        
        Args:
            df: DataFrame with cluster information
            
        Returns:
            List of tuples (cluster_id, latitude, longitude, time_window)
        """
        unique_points = df.groupby(
            ['cluster_id', 'cluster_lat', 'cluster_lon', 'time_window']
        ).size().reset_index().drop(0, axis=1)
        
        return list(unique_points.itertuples(index=False, name=None))
        
    def get_cluster_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for each cluster.
        
        Args:
            df: DataFrame with cluster information
            
        Returns:
            DataFrame with cluster statistics
        """
        summary = df.groupby('cluster_id').agg({
            'latitude': ['count', 'min', 'max'],
            'longitude': ['min', 'max'],
            'timestamp': ['min', 'max'],
            'cluster_lat': 'first',
            'cluster_lon': 'first',
            'time_window': 'first'
        }).reset_index()
        
        # Calculate cluster span in km
        summary['cluster_span_km'] = summary.apply(
            lambda x: np.sqrt(
                (x[('latitude', 'max')] - x[('latitude', 'min')])**2 +
                (x[('longitude', 'max')] - x[('longitude', 'min')])**2
            ) * 111.0,
            axis=1
        )
        
        # Calculate time span in hours
        summary['time_span_hours'] = summary.apply(
            lambda x: (x[('timestamp', 'max')] - x[('timestamp', 'min')]).total_seconds() / 3600,
            axis=1
        )
        
        return summary