import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

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

