# Task 3 – Linear Regression

## Project Overview
This project is part of the AI & ML Internship Task 3.

The objective is to implement and understand Simple Linear Regression and Multiple Linear Regression using Python and Scikit-learn.

## Dataset
- Dataset: House Price Prediction
- Total Records: 4,600
- Total Features: 18
- Target Variable: `price`
- Missing Values: 0

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Statsmodels (for OLS regression analysis)

## Implementation

### 1. Simple Linear Regression
- Input Feature: `sqft_living`
- Target: `price`
- Train-Test Split: 80:20
- Random State: 42

### 2. Multiple Linear Regression
Input Features:
- bedrooms
- bathrooms
- sqft_living
- sqft_lot
- floors
- waterfront
- view
- condition
- yr_built

Target: `price`

### 3. Model Evaluation

The models were evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

| Metric | Simple Regression | Multiple Regression |
|---|---:|---:|
| MAE | 225375.25 | 210448.46 |
| MSE | 990204087727.14 | 987102629687.91 |
| R² Score | 0.0291 | 0.0321 |

### 4. Model Visualization
- Simple Linear Regression: Actual house prices and fitted regression line.
- Multiple Linear Regression: Actual vs Predicted house prices.

### 5. Model Interpretation
The coefficients and intercepts were extracted from both trained models to understand the relationship between input features and house prices.

## Conclusion
Both Simple and Multiple Linear Regression models were implemented successfully.

The Multiple Linear Regression model achieved a slightly better R² score and lower MAE and MSE than the Simple Linear Regression model.

The project provided practical implementation experience with regression modeling, model evaluation, and visualization.

## Project Structure
```text
Task-3-Linear-Regression/
│
├── data.csv
├── linear_regression.py
├── README.md
└── screenshots/
```
