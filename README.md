# Personality Type Prediction (Extrovert vs Introvert)

This project predicts whether a person is an **Introvert** or an **Extrovert** based on personality-related behavioral features using advanced **machine learning techniques**.

The solution is implemented in a Jupyter Notebook and uses an **ensemble of gradient boosting models** with **Optuna-based hyperparameter tuning** to achieve high prediction accuracy.

## Project Overview

This project solves a **binary classification problem** where the goal is to identify personality type using structured behavioral data.

The pipeline includes:
- Feature preprocessing
- Scaling & missing-value handling
- Model training & validation
- Ensemble learning
- Hyperparameter optimization

The final model is trained using:
- **XGBoost**
- **CatBoost**
- **LightGBM**

combined using a **soft-voting ensemble** classifier.

---

## Problem Statement

Given personality and behavioral features such as:

- Stage fear
- Feeling drained after socializing
- Other psychological indicators

Predict whether the individual is:
- `Introvert`
- `Extrovert`

---

## Dataset

The project uses:

- `train.csv` – labeled training data  
- `test.csv` – unlabeled test data  

The model uses all numerical and encoded behavioral features present in the dataset.

Categorical fields like `Yes / No` are encoded as binary values.

Target Variable Mapping:

| Label       | Encoding |
|-------------|----------|
| Extrovert   | 0        |
| Introvert   | 1        |

---

## Project Structure

Personality-Predictor/
│
├── personality_predictor.ipynb # Jupyter notebook (main pipeline)
├── train.csv # Training data
├── test.csv # Test data
└── README.md # Project documentation

yaml
Copy code

---

## Machine Learning Pipeline

### 1️. Data Preprocessing

- Convert categorical values (`Yes`, `No`) into numbers
- Scale numerical features using:
StandardScaler

sql
Copy code
- Handle missing values using:
KNN Imputer (k=5)

yaml
Copy code

---

### 2️. Model Training

The following models are trained:

| Model | Description |
|--------|-------------|
| XGBoost | High-performance gradient boosting |
| CatBoost | Robust to noise and categorical patterns |
| LightGBM | Efficient for large datasets |

These models are combined into:

VotingClassifier (Soft Voting)

yaml
Copy code

---

### 3. Hyperparameter Optimization

Hyperparameters are tuned using:

Optuna

yaml
Copy code

The tuning process optimizes:

- Learning rate
- Tree depth
- Number of estimators
- Regularization
- Sampling ratios

This ensures optimal ensemble performance.

---

### 4. Final Model

After tuning:

- Best parameters are selected automatically
- Final ensemble is trained on the entire dataset
- Model performance on validation data ≈ **96% accuracy**

---

## How to Run

### Option 1: Jupyter

Open and run:

personality_predictor.ipynb

yaml
Copy code

Run each cell in order.

---

### Option 2: Kaggle

Upload:
- `personality_predictor.ipynb`
- `train.csv`
- `test.csv`

Run the notebook cells.

---

## Evaluation Metric

Metric used:

Accuracy

yaml
Copy code

Train / Validation Split:

80% Training
20% Validation

yaml
Copy code

---

## Future Improvements

Possible enhancements:

- Feature importance (SHAP)
- Cross-validation
- Stacking ensemble
- Model explainability
- Dashboard visualization

---

## License

This project is built for:

- Learning
- Academic demonstration
- Portfolio use

You may reuse with attribution.

---

## Author  
**Sahil Vikas Gawade**  
B.Tech Chemical Engineering, IIT Guwahati  
Aspiring Data Scientist / Analyst  

---
Thank you for exploring this project! 😊