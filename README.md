# 🏥 Diabetes Progression Predictor with CI/CD

A complete machine learning project that predicts diabetes progression using linear regression, featuring automated CI/CD pipeline and interactive web interface.

[![CI Pipeline](https://github.com/yourusername/diabetes-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/diabetes-predictor/actions)

## 🚀 [Live Demo - Try It Now!](https://your-app-name.streamlit.app)

## 🎯 What This Project Does

### The Machine Learning Model
- **Dataset**: 442 diabetes patients from scikit-learn's built-in dataset
- **Features**: 10 normalized health measurements (age, BMI, blood pressure, blood tests)
- **Prediction**: Diabetes progression score after one year (0-300+ scale)
- **Algorithm**: Linear Regression with 45% accuracy (R² = 0.453)

### Real-World Application
**Example**: Input a patient's health data → Get prediction like "progression score: 145" → Help doctors assess treatment needs.

## ✨ Key Features

### 🤖 Machine Learning
- ✅ Trained linear regression model
- ✅ 45% prediction accuracy on diabetes progression
- ✅ Handles 10 different health metrics
- ✅ Pre-trained model included (`models/diabetes_model.pkl`)

### 🌐 Interactive Web App
- ✅ **Streamlit web interface** - no coding required to use
- ✅ **Real-time predictions** as you adjust patient data
- ✅ **Color-coded risk levels** (green/yellow/red)
- ✅ **Professional UI** with medical icons and descriptions
- ✅ **Example patient presets** for quick testing

### 🔄 Professional CI/CD Pipeline
- ✅ **Automated testing** on every code push
- ✅ **Code quality checks** with flake8 linting
- ✅ **Python 3.11** compatibility testing
- ✅ **Model validation** - ensures model trains successfully
- ✅ **80% test coverage** with comprehensive unit tests

## 🎮 How to Use

### Option 1: Use the Live Web App (Easiest)
1. **Visit**: [Live Demo Link](https://your-app-name.streamlit.app)
2. **Adjust sliders** for patient health measurements
3. **Get instant predictions** with risk assessment
4. **Try example patients** with preset values

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/yourusername/diabetes-predictor.git
cd diabetes-predictor

# Set up environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

### Option 3: Use as Python Library
```python
from src.model import DiabetesPredictor

# Load pre-trained model
predictor = DiabetesPredictor()
predictor.load_model("models/diabetes_model.pkl")

# Make prediction for a patient
patient_data = [0.1, -0.2, 0.5, 0.3, -0.1, 0.2, -0.4, 0.1, 0.3, -0.2]
progression_score = predictor.predict(patient_data)[0]
print(f"Predicted diabetes progression: {progression_score:.1f}")
```

## 📁 Project Structure

```
diabetes-predictor/
├── 🌐 app.py                      # Streamlit web application
├── 📊 src/
│   └── model.py                   # Core ML model implementation
├── 🧪 tests/
│   └── test_model.py              # Comprehensive unit tests (8 tests)
├── 🤖 models/
│   └── diabetes_model.pkl         # Pre-trained model file
├── ⚙️ .github/workflows/
│   └── ci.yml                     # CI/CD pipeline configuration
├── 📋 requirements.txt            # Python dependencies
├── 🔧 pyproject.toml             # Modern project configuration
├── 🚫 .gitignore                 # Git ignore rules
└── 📖 README.md                  # This file
```

## 🛠️ Technologies Used

### Core ML Stack
- **Python 3.11** - Latest stable Python
- **scikit-learn** - Machine learning library
- **NumPy** - Numerical computations
- **joblib** - Model serialization

### Web Interface
- **Streamlit** - Interactive web app framework
- **Responsive design** - Works on desktop and mobile

### Development & CI/CD
- **pytest** - Testing framework with 80% coverage
- **flake8** - Code quality and style checking
- **GitHub Actions** - Automated CI/CD pipeline
- **Git** - Version control

## 🔄 CI/CD Pipeline Explained

### What Happens When You Push Code:

1. **🔍 Code Quality Check**
   ```bash
   flake8 src/  # Checks code style and syntax
   ```

2. **🧪 Automated Testing**
   ```bash
   pytest tests/ -v --cov=src/  # Runs 8 unit tests
   ```

3. **🤖 Model Validation**
   ```bash
   python src/model.py  # Trains model to ensure it works
   ```

4. **✅ Results** - Green checkmark if all pass, red X if anything fails

### Benefits of CI/CD:
- **Prevents bugs** from reaching production
- **Maintains code quality** automatically
- **Safe collaboration** - know code works before merging
- **Professional workflow** - industry standard practices

## 📊 Model Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| **R² Score** | 0.453 | Explains 45% of diabetes progression variance |
| **Mean Squared Error** | ~2900 | Average prediction error |
| **Features** | 10 | Health measurements used for prediction |
| **Training Data** | 442 patients | Real medical data from research |

### Model Interpretability
- **Higher scores** = Worse diabetes progression expected
- **Typical range** = 0 to 300+
- **Clinical relevance** = Helps prioritize patient care

## 🧪 Testing Coverage

**8 comprehensive tests** covering:
- ✅ Model initialization and training
- ✅ Data loading and validation  
- ✅ Prediction functionality
- ✅ Model saving and loading
- ✅ Error handling for edge cases
- ✅ File operations and permissions

```bash
# Run tests locally
pytest tests/ -v --cov=src/
# Result: 80% code coverage
```

## 🌟 What Makes This Project Special

### 🎓 Educational Value
- **Complete ML workflow** from data to deployment
- **Professional practices** - CI/CD, testing, documentation
- **Real-world applicable** - medical prediction use case
- **Beginner friendly** - well documented and explained

### 🏭 Production Ready
- **Automated quality gates** prevent bad code deployment
- **Comprehensive testing** ensures reliability
- **Easy deployment** with one-click Streamlit hosting
- **Maintainable code** with proper structure and documentation

### 👥 Collaboration Friendly
- **Clear documentation** for new developers
- **Automated testing** prevents integration issues
- **Standardized workflow** with CI/CD pipeline
- **Version control** with meaningful commit history

## 🚀 Deployment Options

| Platform | Difficulty | Cost | Best For |
|----------|------------|------|----------|
| **Streamlit Cloud** | Easy | Free | Quick demos, sharing with friends |
| **Hugging Face Spaces** | Easy | Free | ML model showcases |
| **Railway** | Medium | Free tier | Full web applications |
| **Render** | Medium | Free tier | Professional deployments |
| **AWS/GCP/Azure** | Hard | Pay-as-use | Enterprise production |

## 🔮 Future Enhancements

### Short Term
- [ ] Add more ML algorithms (Random Forest, XGBoost)
- [ ] Include feature importance visualization
- [ ] Add data validation and input sanitization
- [ ] Implement model performance monitoring

### Long Term
- [ ] Real-time model retraining pipeline
- [ ] Integration with hospital management systems
- [ ] Advanced security and compliance (HIPAA)
- [ ] Multi-model ensemble predictions
- [ ] A/B testing for model versions

## 🤝 Contributing

```bash
# Fork the repository
# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
pytest tests/
flake8 src/

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# Create Pull Request - CI will automatically test your changes!
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **scikit-learn team** for the diabetes dataset
- **Streamlit** for the amazing web app framework
- **GitHub Actions** for free CI/CD infrastructure
- **Open source community** for the tools and libraries

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/diabetes-predictor/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/yourusername/diabetes-predictor/discussions)
- **Email**: your.email@example.com

---

**⭐ If this project helped you learn ML or CI/CD, please star it on GitHub!**

Built with ❤️ using Python, scikit-learn, Streamlit, and GitHub Actions.
