# Real_Estate_Analysis_Delhi

# 🏡 Real Estate Price Prediction using Multiple Linear Regression

This project is a complete case study on predicting housing prices in Delhi, India using **Multiple Linear Regression**. The dataset contains various features such as property size, number of bedrooms/bathrooms, amenities, furnishing status, and more.

---

## 📌 Objective

- Understand key features affecting housing prices.
- Build a predictive model using multiple linear regression.
- Evaluate model performance and interpret results.

---

## 🧰 Tools and Libraries

- **Python**
- **Pandas, NumPy** – Data handling
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn** – Modeling & evaluation
- **Statsmodels** – Multicollinearity (VIF)

---

## 🧭 Project Workflow

### 1. 🧾 Data Loading & Overview
- Import dataset
- Basic exploration (`shape`, `head`, `info`)
- Check for missing values and duplicates

### 2. 📊 Exploratory Data Analysis (EDA)
- Descriptive statistics
- Frequency distribution for categorical variables
- Distribution plots and boxplots to identify skewness and outliers

### 3. 🧼 Data Cleaning & Transformation
- Convert binary categorical variables (e.g., "yes"/"no" → 1/0)
- One-hot encoding for categorical columns
- Outlier detection using IQR method
- Log transformation of the target variable (`price`)

### 4. ⚖️ Feature Scaling
Applied three normalization techniques:
- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`

### 5. 🔍 Multicollinearity Check
- Calculated **VIF** (Variance Inflation Factor) to detect correlated predictors.

### 6. 📈 Model Building & Evaluation
- Train-test split
- Fit a **Linear Regression model**
- Performance metrics:
  - `R² Score`
  - `MAE`, `MSE`, `RMSE`

### 7. 📊 Visual Analysis of Results
- Regression plots (feature vs price)
- Boxplots for categorical variables vs price
- Impact analysis of furnishing status

---

## 📁 Files

- `Housing.csv` – Original dataset
- `Real_Estate_Analysis.ipynb` – Main analysis notebook
- `README.md` – Project overview

---

## 📊 Sample Output

| Metric     | Value     |
|------------|-----------|
| R² Score   | 0.68+     |
| MAE        | ~0.21     |
| RMSE       | ~0.28     |

---

## 🚀 Future Improvements

- Try **regularization techniques** (Lasso/Ridge) to improve generalization.
- Use **non-linear models** or **tree-based regressors**.
- Add **feature engineering** to capture hidden interactions.

---
