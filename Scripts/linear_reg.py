import pandas as pd #type: ignore
from sklearn.linear_model import LassoCV #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import mean_squared_error, r2_score #type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler #type: ignore
import matplotlib.pyplot as plt #type: ignore

df = pd.read_csv('./Datasets/feature_engineered_dataset.csv')

drop_cols = ['Unnamed: 0', 'FL_DATE', 'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE',
             'ORIGIN_CITY', 'DEST_CITY', 'DEP_TIME', 'ARR_TIME', 'CANCELLED', 'DIVERTED', 'FL_NUMBER', 'TOTAL_KNOWN_DELAY_CAUSE', 'DEP_DELAY']
df = df.drop(columns=drop_cols)

categorical_cols = ['AIRLINE', 'ORIGIN', 'DEST', 'DEP_TIME_OF_DAY', 'ROUTE', 'HAUL_TYPE']

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['ARR_DELAY'])
y = df['ARR_DELAY']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

print(f"Best alpha from CV: {lasso_cv.alpha_}")

# Predict and evaluate
y_train_pred = lasso_cv.predict(X_train_scaled)
y_test_pred = lasso_cv.predict(X_test_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse:.2f}")
print(f"Training R² Score: {train_r2:.4f}")
print(f"Testing MSE: {test_mse:.2f}")
print(f"Testing R² Score: {test_r2:.4f}")

# Optional: see which features were zeroed out
coeffs = pd.Series(lasso_cv.coef_, index=X.columns)
print("\nZeroed-out features by Lasso (feature selection):")
print(coeffs[coeffs == 0].index.tolist())

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Arrival Delay')
plt.ylabel('Predicted Arrival Delay')
plt.title('Lasso: Actual vs Predicted Arrival Delay')
plt.grid(True)
plt.tight_layout()
plt.show()