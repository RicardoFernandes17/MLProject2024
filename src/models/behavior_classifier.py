from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

class JaguarBehaviorClassifier(BaseEstimator, ClassifierMixin):
    """Classifier for jaguar movement behavior."""
    
    def __init__(self, threshold_speed=5, threshold_direction_var=45):
        self.threshold_speed = threshold_speed
        self.threshold_direction_var = threshold_direction_var
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        """Fit the classifier."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        """Predict movement behavior."""
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict behavior probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)
