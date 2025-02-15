from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

class ModelTrainingPipeline:
    """Pipeline for training jaguar behavior model."""
    
    def __init__(self, classifier, feature_cols):
        self.classifier = classifier
        self.feature_cols = feature_cols
        
    def train(self, data, target_col, test_size=0.2):
        """
        Train the model.
        """
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
            
        # Check for NaN values
        if data[self.feature_cols].isna().any().any():
            print("Warning: NaN values found in feature columns. Dropping rows with NaN values.")
            data = data.dropna(subset=self.feature_cols + [target_col])
            
        X = data[self.feature_cols]
        y = data[target_col]
        
        if len(X) == 0:
            raise ValueError("No valid data remaining after cleaning")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.classifier.score(X_train, y_train)
        test_score = self.classifier.score(X_test, y_test)
        
        y_pred = self.classifier.predict(X_test)
        classification_rep = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def save_model(self, filepath):
        """
        Saves the trained classifier to a file using pickle.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.classifier, f)
    
    @staticmethod
    def load_model(filepath):
        """
        Loads a trained classifier from a pickle file.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
