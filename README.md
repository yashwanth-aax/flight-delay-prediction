# ✈️ Flight Delay Prediction System

An end-to-end Machine Learning project that predicts whether a flight will arrive **more than 15 minutes late** using historical U.S. flight data. The project covers the complete ML workflow, including data preprocessing, feature engineering, model development, evaluation, feature importance analysis, and deployment through a Streamlit web application.

---

## Project Overview

Flight delays significantly impact airline operations and passenger experience. This project leverages historical flight information to build a predictive model capable of estimating the probability of an arrival delay using operational flight features.

The project compares multiple machine learning algorithms and deploys the best-performing model for real-time prediction.

---

## Dataset

- **Source:** U.S. Domestic Flight Dataset
- **Records Used:** 300,000+ sampled flights
- **Prediction Task:** Binary Classification
- **Target Variable:** Arrival Delay > 15 Minutes

---

## Features Used

- Month
- Airline
- Origin Airport
- Destination Airport
- Scheduled Departure Time
- Scheduled Flight Time
- Flight Distance
- Departure Delay
- Taxi-Out Time
- Departure Hour (Engineered)
- Morning Flight Indicator (Engineered)

---

## Feature Engineering

The following engineered features were created to improve model performance:

- Departure Hour
- Morning Flight Indicator

Feature importance analysis was performed using XGBoost to identify the most influential variables contributing to flight delays.

---

## Machine Learning Models

The following models were implemented and compared:

- Logistic Regression
- Random Forest
- XGBoost

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Best Model

**XGBoost**

| Metric | Score |
|---------|-------|
| Accuracy | 93.94% |
| Precision | 80.47% |
| Recall | 87.49% |
| F1-Score | 83.83% |
| ROC-AUC | **0.9696** |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

## Project Structure

```
Flight-Delay-Prediction/
│
├── data/
│   └── flights.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── encoders.pkl
│
├── src/
│   ├── app.py
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Workflow

```
Flight Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
Categorical Encoding
       │
       ▼
Train-Test Split
       │
       ▼
Model Training
(Logistic Regression,
Random Forest,
XGBoost)
       │
       ▼
Performance Evaluation
       │
       ▼
Feature Importance Analysis
       │
       ▼
Model Serialization
(Joblib)
       │
       ▼
Streamlit Web Application
```

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Models

```bash
python -m src.train
```

### Launch Streamlit Application

```bash
python -m streamlit run src/app.py
```

---

## Key Highlights

- End-to-end Machine Learning pipeline
- 300K+ historical flight records
- Feature engineering and importance analysis
- Comparison of multiple classification algorithms
- High-performing XGBoost model (ROC-AUC: **0.9696**)
- Interactive web application for real-time prediction
- Modular and reusable project architecture

---

## Future Improvements

- Hyperparameter optimization using GridSearchCV
- Bayesian posterior uncertainty estimation
- SHAP-based model explainability
- Real-time weather integration
- Flight route visualization dashboard
- REST API deployment using FastAPI

---

## Author

**Yashwanth Bunny**

Indian Institute of Technology Kanpur  
Department of Statistics and Data Science
