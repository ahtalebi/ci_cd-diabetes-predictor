# Diabetes Predictor

Simple linear regression model to predict diabetes progression.

## Usage
```python
from src.model import DiabetesPredictor

predictor = DiabetesPredictor()
results = predictor.train()
print(f"R² Score: {results['r2']:.3f}")
