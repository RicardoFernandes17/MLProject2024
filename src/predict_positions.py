import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predictor.position_prediction import PositionPredictor
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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
    
    #print(predictions)
    
    # Create map visualization
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    for _i, row in predictions.iterrows():
        ax.plot(row['predicted_longitude'], 
                row['predicted_latitude'],
                'o-',
                markersize=10,
                alpha=0.6,
                label=f'Jaguar')

    ax.set_title('Jaguar Movement Patterns')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Save predictions
    output_path = "data/results/position_predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"\nFull predictions saved to: {output_path}")

if __name__ == "__main__":
    main()