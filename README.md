# insurance-premium-prediction

## Overview
Machine learning project to predict insurance premium amounts using regression models. Built as part of DS340, this project includes a full ML pipeline — from data cleaning and feature engineering to model tuning and a Kaggle submission.

## Dataset
- 960,000 training rows with 22 features
- 240,000 validation rows (80/20 split)
- Features include age, BMI, smoking status, gender, occupation, education, location, and policy date information

## Approach

### Preprocessing
- Dropped ID column; converted policy dates to datetime
- Imputed missing values: mean for numerical, most frequent for binary categorical, constant ("missing") for multi-category

### Feature Engineering
- Extracted `policy_year` and `policy_month` from dates
- Cyclical encoding for month (sin/cos transformation)
- Binary encoding for gender and smoking status
- Target encoding for high-cardinality features (occupation, education, location)

### Target Transformation
- Applied log transformation (`log(1 + premium)`) to address skewed distribution and stabilize variance

## Models
| Model | Validation RMSE (Log Scale) |
|---|---|
| Ridge Regression | 1.0890 |
| XGBoost | 1.0533 |
| LightGBM | **1.0496** |

**Best model:** LightGBM, tuned via GridSearchCV  
**Final validation RMSE:** $923.97

## How to Run
1. Clone the repo
   ```bash
   git clone https://github.com/your-username/ds340-insurance-premium-prediction.git
   cd ds340-insurance-premium-prediction
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook in Jupyter
   ```bash
   jupyter notebook DS340_SaanviElaty_project.ipynb
   ```
4. Run all cells

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
category_encoders
joblib
```

## Results
The final LightGBM model was retrained on the full training set, predictions were inverse-log-transformed, and results were submitted to Kaggle as `submission.csv`.
