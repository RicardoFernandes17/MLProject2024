import requests
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime

class WeatherService:
    """
    A service for retrieving and caching weather data for geographical locations.
    
    The class provides methods to:
    - Fetch weather data from Open-Meteo Archive API.
    - Caches weather information to reduce redundant API calls.
    - Process and annotate location data with weather information.
    
    Attributes:
        cache_dir (Path): Directory for storing cached weather data.
        base_url (str): Base URL for weather data API.
    """
    def __init__(self, cache_dir: str = 'data/weather_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    def get_weather_for_village(self, date: str, lat: float, lon: float):
        """
        Retrieve weather data for a specific location and date.
        
        Uses a caching mechanism to store and retrieve previously fetched data.
        Fetches hourly weather data including temperature, humidity, wind, etc.
        
        Args:
            date (str): Date for weather data in 'YYYY-MM-DD' format
            lat (float): Latitude of the location
            lon (float): Longitude of the location
        
        Returns:
            dict: Weather data for the specified location and date
        """
        cache_key = f"{lat:.2f}_{lon:.2f}_{date}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': date,
            'end_date': date,
            'hourly': 'temperature_2m,relativehumidity_2m,precipitation,windspeed_10m,winddirection_10m,cloudcover',
            'timezone': 'auto'
        }
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            time.sleep(0.1) 
            return data
            
        except Exception as e:
            print(f"Error fetching weather for {lat}, {lon}: {e}")
            return None
    
    def add_weather_data(self, unique_locations: pd.DataFrame):
        """
        Enrich a DataFrame of locations with corresponding weather information.
        
        Iterates through unique locations, fetches weather data, and adds 
        weather-related columns to the input DataFrame.
        
        Args:
            unique_locations (pd.DataFrame): DataFrame with location information
        
        Returns:
            pd.DataFrame: Original DataFrame with added weather columns
        """
        print("Adding weather data to villages...")
        data_with_weather = unique_locations.copy()
        
        weather_columns = ['temperature', 'humidity', 'wind_speed',
                         'wind_direction', 'precipitation', 'cloud_cover']
        for col in weather_columns:
            data_with_weather[col] = None
        
        total_rows = len(data_with_weather)
        for idx, row in data_with_weather.iterrows():
            if idx % 10 == 0:
                print(f"Processing {idx}/{total_rows} locations...")
                
            date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], datetime) else str(row['date'])
            
            weather_data = self.get_weather_for_village(
                date_str,
                row['village_latitude'],
                row['village_longitude']
            )
            
            if weather_data and 'hourly' in weather_data:
                hourly = weather_data['hourly']
                hour_idx = 12
                
                data_with_weather.loc[idx, 'temperature'] = hourly['temperature_2m'][hour_idx]
                data_with_weather.loc[idx, 'humidity'] = hourly['relativehumidity_2m'][hour_idx]
                data_with_weather.loc[idx, 'wind_speed'] = hourly['windspeed_10m'][hour_idx]
                data_with_weather.loc[idx, 'wind_direction'] = hourly['winddirection_10m'][hour_idx]
                data_with_weather.loc[idx, 'precipitation'] = hourly['precipitation'][hour_idx]
                data_with_weather.loc[idx, 'cloud_cover'] = hourly['cloudcover'][hour_idx]
        print("Weather data addition completed.")
        return data_with_weather