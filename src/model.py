"""
Simple Linear Regression Model using scikit-learn
"""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


class DiabetesPredictor:
    """Simple diabetes progression predictor"""

    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False

    def load_data(self):
        """Load the diabetes dataset"""
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        """Train the model"""
        X_train, X_test, y_train, y_test = self.load_data()

        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'mse': mse,
            'r2': r2,
            'predictions': y_pred[:5].tolist(),  # First 5 predictions
            'actual': y_test[:5].tolist()        # First 5 actual values
        }

    def predict(self, features):
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)

        return self.model.predict(features)

    def save_model(self, filepath="model.pkl"):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        joblib.dump(self.model, filepath)
        return filepath

    def load_model(self, filepath="model.pkl"):
        """Load a trained model"""
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            self.is_trained = True
            return True
        return False


def main():
    """Main function to demonstrate the model"""
    predictor = DiabetesPredictor()

    print("Training diabetes progression model...")
    results = predictor.train()

    print("Model Performance:")
    print("Mean Squared Error: {:.2f}".format(results['mse']))
    print("R² Score: {:.3f}".format(results['r2']))
    print("Sample predictions: {}".format(results['predictions']))
    print("Actual values: {}".format(results['actual']))

    # Save the model
    model_path = predictor.save_model("models/diabetes_model.pkl")
    print("Model saved to: {}".format(model_path))


if __name__ == "__main__":
    main()
