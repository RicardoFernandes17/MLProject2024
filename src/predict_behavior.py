# src/predict_behavior.py
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from pipeline.prediction import BehaviorPredictor
import pandas as pd

def predict_new_data(data_path: str) -> pd.DataFrame:
    """
    Process new data and predict jaguar behaviors.
    
    Args:
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
    # Example usage
    print("Loading new data and making predictions...")
    
    # Replace with path to your new data
    new_data_path = "data/new_jaguar_movement.csv"
    
    try:
        predictions = predict_new_data(new_data_path)
        
        print("\nPrediction Summary:")
        print("\nBehavior State Distribution:")
        print(predictions['predicted_state'].value_counts())
        
        print("\nSample Predictions (first 5 rows):")
        preview_cols = [
            'timestamp', 'individual_id', 'speed_mean',
            'predicted_state'
        ] + [col for col in predictions.columns if col.startswith('probability_')]
        print(predictions[preview_cols].head())
        
        # Save predictions
        output_path = "results/predictions.csv"
        predictions.to_csv(output_path, index=False)
        print(f"\nFull predictions saved to: {output_path}")
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")

if __name__ == "__main__":
    main()