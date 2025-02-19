import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predictor import WeatherMovementPredictor
import pandas as pd
import matplotlib.pyplot as plt


def predict_new_data(data_path: str):
    """
    Process new weather data and predict movement states.
    
    Args:
        data_path: Path to CSV file with weather data
        
    Returns:
        DataFrame with original data plus movement predictions
    """
    # Load and process the data
    data = pd.read_csv(data_path)
    
    # Load model and make predictions
    predictor = WeatherMovementPredictor()
    predictions = predictor.predict_movement_state(data)
    
    return predictions

def main():
    """Load data, make predictions, and save results."""
    predictions = predict_new_data("data/predict/new_weather_data.csv")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    movement_counts = predictions['predicted_movement_state'].value_counts()
    movement_counts.plot(kind='bar')
    plt.title('Predicted Movement States Distribution')
    plt.xlabel('Movement State')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Save predictions
    output_path = "data/results/weather_movement_predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"\nFull predictions saved to: {output_path}")

if __name__ == "__main__":
    main()