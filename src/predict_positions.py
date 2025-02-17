import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predictor.prediction import PositionPredictor
import pandas as pd
import numpy as np


def predict_new_data(data_path: str):
    """
    Process new data and predict jaguar locations.
    
    Args:
        data_path: Path to CSV file with new movement data
        
    Returns:
        DataFrame with original data plus location predictions
    """
    # Load and process the data
    data = pd.read_csv(data_path)
    
    # Load model and make predictions
    predictor = PositionPredictor()
    predictions = predictor.predict_position(data)
    
    return predictions

def main():
    """Load data, make predictions, and save results."""
    predictions = predict_new_data("data/predict/new_jaguar_data.csv")
    
    print(predictions)

    # Save predictions
    output_path = "data/results/position_predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"\nFull predictions saved to: {output_path}")

if __name__ == "__main__":
    main()