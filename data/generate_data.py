import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_realistic_movement_pattern(n_points, pattern_type='resting'):
    """
    Create realistic movement patterns for different jaguar behaviors.
    
    Args:
        n_points: Number of points to generate
        pattern_type: One of 'resting', 'foraging', 'traveling', 'exploring'
    """
    if pattern_type == 'resting':
        # Small movements around a central point
        center_lat, center_lon = -15.5, -55.5
        latitude = center_lat + np.random.normal(0, 0.001, n_points)
        longitude = center_lon + np.random.normal(0, 0.001, n_points)
        
    elif pattern_type == 'foraging':
        # Random walk with small steps
        steps_lat = np.random.normal(0, 0.005, n_points)
        steps_lon = np.random.normal(0, 0.005, n_points)
        latitude = -15.5 + np.cumsum(steps_lat)
        longitude = -55.5 + np.cumsum(steps_lon)
        
    elif pattern_type == 'traveling':
        # Directional movement
        t = np.linspace(0, 1, n_points)
        latitude = -15.5 + t * 0.1  # Moving north
        longitude = -55.5 + t * 0.1  # Moving east
        
    else:  # exploring
        # Random walk with medium steps
        steps_lat = np.random.normal(0, 0.01, n_points)
        steps_lon = np.random.normal(0, 0.01, n_points)
        latitude = -15.5 + np.cumsum(steps_lat)
        longitude = -55.5 + np.cumsum(steps_lon)
    
    return latitude, longitude

def create_test_dataset():
    """Create test dataset with different movement patterns."""
    
    # Ensure data/predict directory exists
    os.makedirs('data/predict', exist_ok=True)
    
    # Time settings
    start_time = datetime(2025, 1, 1)
    interval_minutes = 30
    
    # Create different movement patterns
    patterns = {
        'resting': 20,     # 10 hours of resting
        'foraging': 16,    # 8 hours of foraging
        'traveling': 12,   # 6 hours of traveling
        'exploring': 16    # 8 hours of exploring
    }
    
    data_list = []
    current_time = start_time
    
    for pattern, n_points in patterns.items():
        latitude, longitude = create_realistic_movement_pattern(n_points, pattern)
        
        for i in range(n_points):
            data_list.append({
                'timestamp': current_time,
                'longitude': longitude[i],
                'latitude': latitude[i],
                'individual_id': 1  # Using single jaguar for simplicity
            })
            current_time += timedelta(minutes=interval_minutes)
    
    # Create DataFrame and sort by timestamp
    df = pd.DataFrame(data_list)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('data/predict/new_jaguar_movement.csv', index=False)
    print("Created test dataset with following characteristics:")
    print(f"Total points: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("\nSample of generated data:")
    print(df.head())
    
    return df

def generate_random_data(n_samples=20):
    """Generate random test data for jaguar location prediction."""
    
    # Ensure data/predict directory exists
    os.makedirs('data/predict', exist_ok=True)
    
    data = {
        'sex': np.random.choice(['M', 'F'], n_samples),
        'age': np.random.randint(0, 12, n_samples),
        'weight': np.random.uniform(40, 130, n_samples),
        'month': np.random.randint(1, 13, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = 'data/predict/new_jaguar_data.csv'
    df.to_csv(output_path, index=False)
    
    print("Generated random test data:")
    print(f"Total samples: {n_samples}")
    print("\nSample of generated data:")
    print(df.head())
    print(f"\nData saved to: {output_path}")
    
    return df

def generate_weather_data(num_samples=50):
    """Generate random weather data for testing."""
    
    # Ensure data/predict directory exists
    os.makedirs('data/predict', exist_ok=True)
    
    # Create a DataFrame
    data = {
        'date': [datetime.now().date() - timedelta(days=x) for x in range(num_samples)],
        'village_latitude': np.round(np.random.uniform(-25, -15, num_samples), 6),
        'village_longitude': np.round(np.random.uniform(-55, -45, num_samples), 6),
        'temperature': np.round(np.random.uniform(15, 35, num_samples), 1),
        'humidity': np.round(np.random.uniform(40, 90, num_samples)).astype(int),
        'wind_speed': np.round(np.random.uniform(0, 20, num_samples), 1),
        'wind_direction': np.round(np.random.uniform(0, 360, num_samples)).astype(int),
        'precipitation': np.round(np.random.uniform(0, 50, num_samples), 1),
        'cloud_cover': np.round(np.random.uniform(0, 100, num_samples)).astype(int)
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure realistic ranges
    df.loc[df['wind_direction'] > 360, 'wind_direction'] = 360
    df.loc[df['wind_direction'] < 0, 'wind_direction'] = 0
    df.loc[df['cloud_cover'] > 100, 'cloud_cover'] = 100
    df.loc[df['cloud_cover'] < 0, 'cloud_cover'] = 0
    
    # Save to CSV
    output_path = 'data/predict/new_weather_data.csv'
    df.to_csv(output_path, index=False)
    
    print("Generated weather test data:")
    print(f"Total samples: {num_samples}")
    print("\nSample of generated data:")
    print(df.head())
    print(f"\nData saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    create_test_dataset()
    generate_random_data()
    generate_weather_data()