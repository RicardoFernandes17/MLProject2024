
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def get_speed_feature_plot(df, x='time_of_day', y='speed'):
    plt.figure(figsize=(15, 5))
    sns.boxplot(x=x, y=y, data=df)
    plt.title('Speed Distribution by Time of Day')
    plt.xticks(rotation=45)
    plt.ylabel('Speed (km/h)')
    return plt

def get_distance_distribution_plot(df,x='individual_id', y='distance'):
    plt.figure(figsize=(15, 5))
    sns.boxplot(x=x, y=y, data=df)
    plt.title('Distance Distribution by Jaguar')
    plt.xticks(rotation=45)
    plt.ylabel('Distance (km)')
    return plt

def get_movement_patterns_plot(df,hour_key='hour', speed_key='speed'):
    plt.figure(figsize=(15, 5))    
    hourly_speed = df.groupby(hour_key)[speed_key].mean()
    plt.plot(hourly_speed.index, hourly_speed.values)
    plt.title('Average Speed Throughout the Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Speed (km/h)')
    return plt

def get_direction_distribution_plot(df,column='direction'):
    plt.figure(figsize=(15, 5))    
    directions = df[column].dropna()
    plt.hist(np.deg2rad(directions), bins=36)
    plt.title('Movement Direction Distribution')
    return plt


def get_temporal_analysis_plot(df):
    monthly_distance = df.groupby(['year', 'month'])['distance'].mean().reset_index()
    monthly_distance['date'] = pd.to_datetime(monthly_distance[['year', 'month']].assign(day=1))
    
    plt.figure(figsize=(15, 5))
    plt.plot(monthly_distance['date'], monthly_distance['distance'])
    plt.title('Average Distance Traveled Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Distance (km)')
    plt.xticks(rotation=45)
    return plt


def get_added_features_plot(data):
    plt.figure(figsize=(20, 30))
    
    # Speed distribution (by time of the day)
    plt.subplot(4, 1, 1)  # Changed to 4 rows, 1 column, first plot
    sns.boxplot(x='time_of_day', y='speed', data=data)
    plt.title('Speed Distribution by Time of Day')
    plt.xticks(rotation=45)
    plt.ylabel('Speed (km/h)')

    # Distance distribution (by jaguar)
    plt.subplot(4, 1, 2)  # Second plot
    sns.boxplot(x='individual_id', y='distance', data=data)
    plt.title('Distance Distribution by Jaguar')
    plt.xticks(rotation=45)
    plt.ylabel('Distance (km)')

    # Movement patterns (over 24 hours)
    plt.subplot(4, 1, 3)  # Third plot
    hourly_speed = data.groupby('hour')['speed'].mean()
    plt.plot(hourly_speed.index, hourly_speed.values)
    plt.title('Average Speed Throughout the Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Speed (km/h)')

    # Direction distribution (Rose plot)
    plt.subplot(4, 1, 4, projection='polar')  # Fourth plot
    directions = data['direction'].dropna()
    plt.hist(np.deg2rad(directions), bins=36)
    plt.title('Movement Direction Distribution')

    # Plotting
    plt.tight_layout()
    return plt