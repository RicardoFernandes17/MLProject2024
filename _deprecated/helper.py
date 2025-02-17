import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def get_dataset_with_copy(file_path):
    dataset_original = pd.read_csv(file_path)
    dataset = dataset_original.copy()
    return dataset_original, dataset

def remove_outliers(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    df[column] = df[column].mask(abs(df[column] - mean) > n_std * std)
    return df

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points using the Haversine formula
    Returns distance in kilometers
    """
    radius = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = radius * c

    return distance

def calculate_group_distances(group):
    distances = []
    for i in range(len(group)):
        if i < len(group) - 1:
            dist = calculate_distance(group['latitude'].iloc[i], group['longitude'].iloc[i],group['latitude'].iloc[i+1], group['longitude'].iloc[i+1])
            distances.append(dist)
        else:
            distances.append(np.nan)
    return pd.Series(distances, index=group.index)

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing (direction) between two points
    Returns angle in degrees (0-360)
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = np.degrees(atan2(y, x))
    
    # Normalize to 0-360
    return (bearing + 360) % 360

def calculate_group_directions(group):
    directions = []
    for i in range(len(group)):
        if i < len(group) - 1:
            bearing = calculate_bearing(group['latitude'].iloc[i], group['longitude'].iloc[i], group['latitude'].iloc[i+1], group['longitude'].iloc[i+1])
            directions.append(bearing)
        else:
            directions.append(np.nan)
    return pd.Series(directions, index=group.index)


def create_time_window_features(group, window_size='6H'):
    """
    Calculate features for each time window
    window_size: pandas offset string (e.g., '6H' for 6 hours)
    """
    # Check if timestamp is already an index
    if 'timestamp' not in group.index.names:
        group = group.set_index('timestamp')
        
    # Resample data into time windows
    resampled = group.resample(window_size).agg({
        'speed': ['mean', 'max', 'std'],
        'distance': ['sum', 'mean'],
        'direction': ['mean', 'std'],
        'latitude': ['min', 'max'],
        'longitude': ['min', 'max']
    })
    
    # Flatten column names
    resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
    
    return resampled

def calculate_movement_features(df):
    """
    Calculate additional movement pattern features
    """
    # Calculate area covered in each window
    df['area_covered'] = (df['latitude_max'] - df['latitude_min']) * \
                        (df['longitude_max'] - df['longitude_min'])
    
    # Calculate directional changes
    df['direction_variability'] = df['direction_std']
    
    # Calculate movement intensity
    df['movement_intensity'] = df['speed_mean'] * df['distance_sum']
    
    # Calculate straightness of movement
    df['path_efficiency'] = df['distance_mean'] / df['distance_sum']
    
    return df

def classify_movement_state(row):
    """
    Classify the movement state based on speed and direction variability
    """
    if row['speed_mean'] < 0.1:
        return 'resting'
    elif row['speed_mean'] > 2.0 and row['direction_std'] > 45:
        return 'hunting'
    elif row['speed_mean'] > 1.0 and row['direction_std'] < 30:
        return 'traveling'
    else:
        return 'exploring'
    
def get_columns_with_nulls(df):
    """
    Returns a list of column names that contain null values in a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame to check for null values
    
    Returns:
    list: List of column names that contain at least one null value
    """
    null_columns = df.isnull().any()
    columns_with_nulls = null_columns[null_columns].index.tolist()
    return columns_with_nulls

def fill_null_values(df, columns_to_check, numeric_fill=0, categorical_fill='Unknown'):
    """
    Fill null values in specified columns based on their data type.
    Only processes if columns_to_check is not empty.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to process
    columns_to_check (list): List of columns with null values from get_columns_with_nulls
    numeric_fill (numeric): Value to fill nulls in numeric columns (default: 0)
    categorical_fill (str): Value to fill nulls in categorical columns (default: 'Unknown')
    
    Returns:
    pandas.DataFrame: DataFrame with filled values
    """
    if not columns_to_check:
        print("No columns to process")
        return df
        
    df_filled = df.copy()
    
    # Split columns by data type
    numeric_nulls = [col for col in columns_to_check if df[col].dtype in ['int64', 'float64']]
    categorical_nulls = [col for col in columns_to_check if col not in numeric_nulls]
    
    # Fill numeric columns
    for col in numeric_nulls:
        df_filled[col] = df_filled[col].fillna(value=numeric_fill)
    
    # Fill categorical columns
    for col in categorical_nulls:
        df_filled[col] = df_filled[col].fillna(value=categorical_fill)
            
    return df_filled

def get_date_values_from_timestamp(df, field_name='timestamp', time_of_day_bins=[0, 6, 12, 18, 24], time_of_day_labels=['Night', 'Morning', 'Afternoon', 'Evening']):
    timestamp_dt =  df[field_name].dt
    df['hour'] = timestamp_dt.hour
    df['day'] = timestamp_dt.day
    df['month'] = timestamp_dt.month
    df['year'] = timestamp_dt.year
    df['dayofweek'] = timestamp_dt.dayofweek
    df['time_of_day'] = pd.cut(timestamp_dt.hour, bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    return df

def get_jaguar_movement_features(df,identifier='individual_id', timestamp_field_name = 'timestamp'):
    df['time_diff'] = df.groupby(identifier)[timestamp_field_name].diff()
    df['time_diff_hours'] = df['time_diff'].dt.total_seconds() / 3600
    df['distance'] = df.groupby(identifier, group_keys=False).apply(calculate_group_distances)
    df['speed'] = df['distance'] / df['time_diff_hours'].replace({0: np.nan})
    df['direction'] = df.groupby(identifier, group_keys=False).apply(calculate_group_directions)
    
    #Removes outliers
    df = df.replace([np.inf, -np.inf], np.nan)
    df = remove_outliers(df, 'speed')
    df = remove_outliers(df, 'distance')
    df['speed'] = df['speed'].fillna(method='ffill')
    df['distance'] = df['distance'].fillna(method='ffill')
    df['direction'] = df['direction'].fillna(method='ffill')
    return df

def process_dataset(df, columns_to_rename=None, columns_to_drop=None):
    """
    Process a pandas DataFrame by renaming and dropping specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to process
    columns_to_rename : dict, optional
        Dictionary where keys are original column names and values are new column names
        Example: {'old_name': 'new_name', 'old_name2': 'new_name2'}
    columns_to_drop : list, optional
        List of column names to drop from the DataFrame
        Example: ['column1', 'column2']
    
    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame with renamed and dropped columns
    """
    # Create a copy of the DataFrame to avoid modifying the original
    processed_df = df.copy()
    
    # Rename columns if specified
    if columns_to_rename:
        processed_df.rename(columns=columns_to_rename, inplace=True)
    
    # Drop columns if specified
    if columns_to_drop:
        processed_df.drop(columns_to_drop, axis=1, inplace=True)
    
    return processed_df