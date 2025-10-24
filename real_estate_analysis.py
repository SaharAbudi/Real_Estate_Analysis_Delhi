# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Real Estate Price Analysis",
    page_icon="🏠",
    layout="wide"
)

st.title("🏙️ Real Estate Price Analysis Dashboard")
st.markdown("This dashboard explores the **Housing Dataset**, analyzes trends, and builds predictive models.")

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
file_path = "Housing.csv"

if not os.path.exists(file_path):
    st.error(f"❌ File '{file_path}' not found. Make sure it's in the repository root.")
    st.stop()

df = pd.read_csv(file_path)
st.success("✅ Dataset loaded successfully!")
st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

st.subheader("Preview of Dataset")
st.dataframe(df.head())

# ----------------------------------------------------
# DATA EXPLORATION
# ----------------------------------------------------
st.subheader("Data Overview")
col1, col2 = st.columns(2)
with col1:
    st.write("**Data Types & Non-Null Counts:**")
    buffer = df.dtypes.to_frame("Type").join(df.notnull().sum().rename("Non-Null Count"))
    st.dataframe(buffer)
with col2:
    st.write("**Missing Values:**")
    st.dataframe(df.isnull().sum())

st.write(f"**Duplicate Records:** {df.duplicated().sum()}")

# ----------------------------------------------------
# STATISTICS
# ----------------------------------------------------
st.subheader("Descriptive Statistics")
st.dataframe(df.describe())

# ----------------------------------------------------
# VISUALIZATION SECTION
# ----------------------------------------------------
st.subheader("📊 Price Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["price"], kde=True, ax=ax, color="skyblue")
ax.set_title("Distribution of House Prices")
ax.set_xlabel("Price")
ax.set_ylabel("Frequency")
st.pyplot(fig)

st.markdown("**Price Summary:**")
st.write({
    "Min": df["price"].min(),
    "Max": df["price"].max(),
    "Mean": round(df["price"].mean(), 2),
    "Median": round(df["price"].median(), 2),
    "Std Dev": round(df["price"].std(), 2),
    "Skewness": round(df["price"].skew(), 2)
})

# ----------------------------------------------------
# CORRELATION
# ----------------------------------------------------
st.subheader("🔗 Correlation Matrix")

df_numeric = df.copy()
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df_numeric[col] = df_numeric[col].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})

df_numeric = pd.get_dummies(df_numeric, columns=['furnishingstatus'], drop_first=True)

corr = df_numeric.corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix")
st.pyplot(fig)

st.write("**Top Correlations with Price:**")
st.dataframe(corr["price"].sort_values(ascending=False).to_frame("Correlation"))

# ----------------------------------------------------
# OUTLIER DETECTION
# ----------------------------------------------------
st.subheader("📈 Outlier Detection")

numerical_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    sns.boxplot(y=df[col], ax=axes[i], color="lightcoral")
    axes[i].set_title(f"{col}")
st.pyplot(fig)

# ----------------------------------------------------
# TRANSFORMATION
# ----------------------------------------------------
st.subheader("Log Transformation of Price")
df_log = df_numeric.copy()
df_log["price_log"] = np.log(df_log["price"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df_log["price"], kde=True, ax=axes[0])
axes[0].set_title("Original Price Distribution")
sns.histplot(df_log["price_log"], kde=True, ax=axes[1], color="orange")
axes[1].set_title("Log-Transformed Price Distribution")
st.pyplot(fig)

# ----------------------------------------------------
# SCALING METHODS
# ----------------------------------------------------
st.subheader("Feature Scaling Comparison")
num_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}

for name, scaler in scalers.items():
    df_scaled = df_log.copy()
    df_scaled[num_features] = scaler.fit_transform(df_log[num_features])
    st.markdown(f"**{name} Statistics:**")
    st.dataframe(df_scaled[num_features].describe().round(3).loc[['mean', 'std', 'min', 'max']])

# ----------------------------------------------------
# MULTICOLLINEARITY (VIF)
# ----------------------------------------------------
st.subheader("📉 Variance Inflation Factor (VIF)")
X = df_log.drop(["price", "price_log"], axis=1)
vif_data = pd.DataFrame({
    "Feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
st.dataframe(vif_data.sort_values("VIF", ascending=False))

# ----------------------------------------------------
# LINEAR REGRESSION MODEL
# ----------------------------------------------------
st.subheader("🏗️ Linear Regression Model")

X_std = df_log.drop(columns=["price", "price_log"])
y_std = df_log["price_log"]
X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
st.write("**Prediction Comparison:**")
st.dataframe(results.head())

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

st.write("### Model Performance")
st.metric("R² Score", f"{r2:.3f}")
st.metric("MAE", f"{mae:.3f}")
st.metric("RMSE", f"{rmse:.3f}")

st.balloons()
