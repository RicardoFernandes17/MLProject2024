import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.feature_engineering import FeatureEngineer
from predictor.prediction import BehaviorPredictor
import pandas as pd

def predict_new_data(data_path):
    """
    Process new data and predict jaguar behaviors.
    
    data_path: Path to CSV file with new movement data
        
    Returns:
        DataFrame with original data plus behavior predictions
    """
    # Load and process the data
    data = pd.read_csv(data_path)
    
    # Add required features
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = FeatureEngineer.add_time_features(data)
    
    data = FeatureEngineer.calculate_movement_features(data)
    
    # Create movement windows
    window_data = FeatureEngineer.create_movement_windows(data)
    
    # Load model and make predictions
    predictor = BehaviorPredictor()
    
    predictions = predictor.predict_behavior(window_data)
    
    return predictions

def main():
    predictions = predict_new_data("data/predict/new_jaguar_movement.csv")
    
    # Save predictions
    output_path = "data/results/movement_predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"\nFull predictions saved to: {output_path}")

if __name__ == "__main__":
    main()