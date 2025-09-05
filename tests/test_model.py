"""
Tests for the diabetes prediction model
"""
import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import DiabetesPredictor


class TestDiabetesPredictor:
    """Test cases for DiabetesPredictor"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        predictor = DiabetesPredictor()
        assert predictor.model is not None
        assert not predictor.is_trained
    
    def test_data_loading(self):
        """Test data loading functionality"""
        predictor = DiabetesPredictor()
        X_train, X_test, y_train, y_test = predictor.load_data()
        
        # Check data shapes
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        
        # Check feature dimensions (diabetes dataset has 10 features)
        assert X_train.shape[1] == 10
        assert X_test.shape[1] == 10
    
    def test_model_training(self):
        """Test model training"""
        predictor = DiabetesPredictor()
        results = predictor.train()
        
        # Check training results
        assert predictor.is_trained
        assert 'mse' in results
        assert 'r2' in results
        assert results['mse'] > 0  # MSE should be positive
        assert -1 <= results['r2'] <= 1  # R² should be between -1 and 1
        
        # Check predictions format
        assert len(results['predictions']) == 5
        assert len(results['actual']) == 5
    
    def test_prediction_before_training(self):
        """Test that prediction fails before training"""
        predictor = DiabetesPredictor()
        sample_features = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0]
        
        with pytest.raises(ValueError, match="Model must be trained"):
            predictor.predict(sample_features)
    
    def test_prediction_after_training(self):
        """Test prediction after training"""
        predictor = DiabetesPredictor()
        predictor.train()  # Train the model
        
        # Test prediction with sample features
        sample_features = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0]
        prediction = predictor.predict(sample_features)
        
        assert prediction is not None
        assert len(prediction) == 1  # Single prediction
        assert isinstance(prediction[0], (int, float, np.number))
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        predictor = DiabetesPredictor()
        predictor.train()
        
        # Test saving
        model_path = "test_model.pkl"
        saved_path = predictor.save_model(model_path)
        assert saved_path == model_path
        assert os.path.exists(model_path)
        
        # Test loading
        new_predictor = DiabetesPredictor()
        assert not new_predictor.is_trained
        
        success = new_predictor.load_model(model_path)
        assert success
        assert new_predictor.is_trained
        
        # Test prediction with loaded model
        sample_features = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0]
        prediction = new_predictor.predict(sample_features)
        assert prediction is not None
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
    
    def test_model_save_without_training(self):
        """Test that saving fails without training"""
        predictor = DiabetesPredictor()
        
        with pytest.raises(ValueError, match="Model must be trained"):
            predictor.save_model("test_model.pkl")
    
    def test_load_nonexistent_model(self):
        """Test loading a non-existent model"""
        predictor = DiabetesPredictor()
        success = predictor.load_model("nonexistent_model.pkl")
        assert not success
        assert not predictor.is_trained
