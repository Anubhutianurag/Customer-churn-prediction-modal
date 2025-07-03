# Customer Churn Prediction

This project focuses on predicting customer churn using machine learning techniques. The goal is to analyze customer data, identify patterns leading to churn, and build a robust predictive model to help businesses proactively retain their customers.

---

## Project Overview

Customer churn is a critical metric for subscription-based businesses. By leveraging data analysis and machine learning, this project aims to:

- Understand customer behavior
- Identify key factors influencing churn
- Predict which customers are likely to leave
- Support data-driven retention strategies

---

## Technologies & Tools

- Python: Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn: Model building and evaluation
- XGBoost / Random Forest: Ensemble methods
- SHAP: Model interpretability
- SMOTE: Handling class imbalance
- ELI5: Feature weight explanations
- SQL: Query-based aggregation
- Jupyter Notebook / Google Colab

---

## Features

- Data preprocessing: Missing values, type conversions, and label encoding
- Exploratory Data Analysis (EDA): Statistical summaries, visual insights, segment-wise churn analysis
- Feature engineering: Encoding categorical features, binning, transformation
- Model training: Decision Tree, Random Forest, XGBoost
- Model evaluation: Accuracy, confusion matrix, ROC-AUC, cross-validation
- Model interpretability: SHAP (SHapley Additive Explanations) for feature influence
- Customer segmentation: Based on churn risk
- Exported trained model and encoders for future use

---
##Modeling Summary

-Data was cleaned and preprocessed
-Categorical variables encoded using label encoding
-SMOTE was used to handle class imbalance
-Dataset split into 80% training and 20% testing
-Multiple models were tested; Random Forest gave the best results (~84% accuracy)
-SHAP values helped interpret key features influencing churn: Tenure, MonthlyCharges, Contract, InternetService

---

## Folder Structure
```plaintext
-├── customer (1).ipynb                             # Main analysis and model training notebook
-├── WA_Fn-UseC_-Telco-Customer-Churn.csv           # Input dataset
-├── encoders.pkl                                   # Encoders for categorical features
-├── customer_churn_model.pkl                       # Trained ML model (Random Forest)
-├── Customer-Churn-Analysis-Telecom-Industry.pdf   # Summary presentation
-├── Customer_Churn_Analysis_Full_Report.pdf        # Detailed project documentation

---

##References

Telco Customer Churn Dataset – Kaggle
SMOTE documentation – Imbalanced-learn
SHAP documentation – SHAP values for explainability

---
