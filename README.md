# Credit Risk Probability Model for Alternative Data

## Project Overview

This project builds an end-to-end credit risk probability model for Bati Bank. The goal is to help the bank support a buy-now-pay-later service by predicting customer credit risk using transaction behavior from an eCommerce platform.

The project includes data exploration, feature engineering, proxy target creation, model training, experiment tracking, API deployment, Docker containerization, and CI/CD automation.

## Project Structure

credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

## Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretability

The Basel II Accord emphasizes proper risk measurement, documentation, and model governance. In credit risk modeling, this means the bank must clearly understand how the model makes decisions and why a customer is classified as high-risk or low-risk.

Because credit decisions can affect customers financially, the model should not only be accurate but also explainable. A well-documented and interpretable model helps the bank justify lending decisions, monitor model behavior, and reduce the risk of unfair or unclear credit scoring.

For this reason, simple and transparent models such as Logistic Regression with Weight of Evidence may be useful in regulated financial settings because they make it easier to explain how each feature contributes to risk.

### 2. Why a Proxy Variable Is Necessary

The dataset does not contain a direct default label showing whether a customer failed to repay a loan. Since supervised machine learning needs a target variable, a proxy variable must be created to represent possible credit risk.

In this project, customer behavior such as Recency, Frequency, and Monetary value can be used to estimate risk. For example, customers with low transaction frequency, low monetary value, and long inactivity may be treated as higher-risk customers.

However, this introduces business risk because the proxy is not the same as actual loan default. A customer who transacts less is not always a bad borrower. This means the model may incorrectly classify some good customers as high-risk or some risky customers as low-risk. Therefore, the proxy target should be clearly documented as a modeling assumption, not ground truth.

### 3. Trade-offs Between Simple and Complex Models

A simple model such as Logistic Regression with Weight of Evidence is easier to explain, monitor, and defend in a regulated banking environment. It is useful when transparency and documentation are very important. However, it may not capture complex patterns in customer behavior.

A more complex model such as Gradient Boosting or Random Forest may produce better predictive performance because it can capture non-linear relationships and interactions between features. However, these models are harder to interpret and may be more difficult to justify to regulators, business stakeholders, and risk teams.

In a regulated credit risk context, the best approach is to compare both interpretable and high-performance models. The final model should balance accuracy, fairness, interpretability, and business usefulness.
"@ | Set-Content README.md

## Dataset Source

The dataset used in this project was provided as part of the 10 Academy Week 4 Credit Risk Modeling Challenge.

The dataset contains anonymized transaction-level financial data used for exploratory data analysis and credit risk modeling.