import pandas as pd #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import mean_squared_error, r2_score #type: ignore
from catboost import CatBoostRegressor, Pool #type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
import numpy as np #type: ignore


# Load dataset
df = pd.read_csv('./Datasets/feature_engineered_dataset.csv')

# Drop unnecessary columns
drop_cols = ['Unnamed: 0', 'FL_DATE', 'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE',
             'ORIGIN_CITY', 'DEST_CITY', 'DEP_TIME', 'ARR_TIME', 'CANCELLED', 'DIVERTED', 'FL_NUMBER', 'TOTAL_KNOWN_DELAY_CAUSE', 'DEP_DELAY']
df = df.drop(columns=drop_cols)

# Define categorical columns (CatBoost can handle string labels directly)
categorical_cols = ['AIRLINE', 'ORIGIN', 'DEST', 'DEP_TIME_OF_DAY', 'ROUTE', 'HAUL_TYPE']

# Split data
X = df.drop(columns=['ARR_DELAY'])
y = df['ARR_DELAY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoost Pool for automatic categorical handling
train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
test_pool = Pool(X_test, y_test, cat_features=categorical_cols)

# Train CatBoost Regressor
model = CatBoostRegressor(verbose=100, random_seed=42)
model.fit(train_pool)

# Predict and evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
print(f"Train MSE: {mean_squared_error(y_train, y_train_pred):.2f}")
print(f"Train R^2: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, y_test_pred):.2f}")
print(f"Test R^2: {r2_score(y_test, y_test_pred):.4f}")


# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.5, edgecolor=None)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Arrival Delay')
plt.ylabel('Predicted Arrival Delay')
plt.title('Predicted vs Actual Arrival Delays (Test Set)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
