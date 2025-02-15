from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.behavior_classifier import JaguarBehaviorClassifier
from pipeline.training_pipeline import ModelTrainingPipeline

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
    
    # Initialize classifier
    print("Initializing classifier...")
    classifier = JaguarBehaviorClassifier()
    
    # Create and run training pipeline
    print("Training model...")
    pipeline = ModelTrainingPipeline(classifier, feature_cols)
    results = pipeline.train(window_data, 'movement_state')
    
    # Save model
    print("Saving model...")
    pipeline.save_model('models/jaguar_behavior_model.pkl')
    
    print("\nTraining Results:")
    print(f"Train Score: {results['train_score']:.4f}")
    print(f"Test Score: {results['test_score']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Additional analysis
    print("\nMovement State Distribution:")
    print(window_data['movement_state'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
