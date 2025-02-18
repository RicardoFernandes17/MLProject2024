import requests
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime
import logging

class WeatherService:
    """Simplified weather service for village-based locations."""
    
    def __init__(self, cache_dir: str = 'data/weather_cache'):
        """Initialize weather service."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_weather_for_village(self, date: str, lat: float, lon: float) -> dict:
        """Get weather data for a village location."""
        cache_key = f"{lat:.2f}_{lon:.2f}_{date}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Check cache first
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Prepare API request
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
            
            # Cache the response
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            time.sleep(0.1)  # Small delay between requests
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching weather for {lat}, {lon}: {e}")
            return None
    
    def add_weather_data(self, unique_locations: pd.DataFrame) -> pd.DataFrame:
        """Add weather data to unique village locations."""
        self.logger.info("Adding weather data to villages...")
        
        # Create copy of input data
        data_with_weather = unique_locations.copy()
        
        # Initialize weather columns
        weather_columns = ['temperature', 'humidity', 'wind_speed',
                         'wind_direction', 'precipitation', 'cloud_cover']
        for col in weather_columns:
            data_with_weather[col] = None
        
        # Process each unique date-location combination
        total_rows = len(data_with_weather)
        for idx, row in data_with_weather.iterrows():
            if idx % 10 == 0:
                self.logger.info(f"Processing {idx}/{total_rows} locations...")
                
            date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], datetime) else str(row['date'])
            
            weather_data = self.get_weather_for_village(
                date_str,
                row['village_latitude'],
                row['village_longitude']
            )
            
            if weather_data and 'hourly' in weather_data:
                hourly = weather_data['hourly']
                # Use noon (12:00) data as representative for the day
                hour_idx = 12
                
                data_with_weather.loc[idx, 'temperature'] = hourly['temperature_2m'][hour_idx]
                data_with_weather.loc[idx, 'humidity'] = hourly['relativehumidity_2m'][hour_idx]
                data_with_weather.loc[idx, 'wind_speed'] = hourly['windspeed_10m'][hour_idx]
                data_with_weather.loc[idx, 'wind_direction'] = hourly['winddirection_10m'][hour_idx]
                data_with_weather.loc[idx, 'precipitation'] = hourly['precipitation'][hour_idx]
                data_with_weather.loc[idx, 'cloud_cover'] = hourly['cloudcover'][hour_idx]
        
        self.logger.info("Weather data addition completed.")
        return data_with_weather