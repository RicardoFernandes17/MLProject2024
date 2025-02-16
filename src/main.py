from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from pipeline.ml_pipeline import MLPipeline
from sklearn.model_selection import train_test_split
import pickle

def main():
    # Initialize data loader
    data_loader = DataLoader(
        'data/raw/jaguar_movement_data.csv',
        'data/raw/jaguar_additional_information.csv'
    )
    
    # Load and preprocess data
    print("Loading data...")
    data = data_loader.load_data()
    
    # Add features
    print("Adding time features...")
    data = FeatureEngineer.add_time_features(data)
    print("Calculating movement features...")
    data = FeatureEngineer.calculate_movement_features(data)
    
    # Create movement windows
    print("Creating movement windows...")
    window_data = FeatureEngineer.create_movement_windows(data)
    
    # Classify movement states
    print("Classifying movement states...")
    window_data = FeatureEngineer.classify_movement_state(window_data)
    
    # Remove rows with unknown states or NaN values
    window_data = window_data.dropna()
    window_data = window_data[window_data['movement_state'] != 'unknown']
    
    # Define feature columns for model
    feature_cols = [
        'speed_mean', 'speed_max', 'speed_std',
        'distance_sum', 'distance_mean',
        'direction_mean', 'direction_std',
        'area_covered', 'movement_intensity',
        'path_efficiency', 'direction_variability'
    ]
    
    # Initialize the ML pipeline
    print("Initializing ML pipeline...")
    ml_pipeline = MLPipeline()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        window_data[feature_cols],
        window_data['movement_state'],
        test_size=0.2,
        random_state=42,
        stratify=window_data['movement_state']
    )
    
    # Train and evaluate models
    print("Training and evaluating models...")
    results = ml_pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Print results
    print("\nModel Results:")
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"Best parameters: {result['best_params']}")
        print(f"Best CV score: {result['best_score']:.4f}")
        print(f"Test score: {result['test_score']:.4f}")
        print(f"CV scores mean ± std: {result['cv_scores'].mean():.4f} ± {result['cv_scores'].std():.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    ml_pipeline.feature_names = feature_cols
    plt = ml_pipeline.plot_results(results)
    plt.savefig('models/model_comparison.png')
    
    # Save best model
    print("\nSaving best model...")
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(ml_pipeline.best_model, f)

if __name__ == "__main__":
    main()
