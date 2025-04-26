import pandas as pd #type: ignore
import xgboost as xgb #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import mean_squared_error, r2_score #type: ignore
from sklearn.preprocessing import LabelEncoder #type: ignore

df = pd.read_csv('./Datasets/feature_engineered_dataset.csv')

drop_cols = ['Unnamed: 0', 'FL_DATE', 'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE',
             'ORIGIN_CITY', 'DEST_CITY', 'DEP_TIME', 'ARR_TIME', 'CANCELLED', 'DIVERTED', 'FL_NUMBER']
df = df.drop(columns=drop_cols)

categorical_cols = ['AIRLINE', 'ORIGIN', 'DEST', 'DEP_TIME_OF_DAY', 'ROUTE', 'HAUL_TYPE']

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders if you want to inverse transform later

X = df.drop(columns=['ARR_DELAY'])
y = df['ARR_DELAY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xg_reg = xgb.XGBRegressor(
    objective='reg:squarederror',  # Use 'reg:squarederror' for regression
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1, # alpha
    random_state=42
)

print("Starting the training...")
xg_reg.fit(X_train, y_train)
print("Training done")

y_pred = xg_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100  # Accuracy in %

print(f"XGBoost MSE: {mse:.2f}")
print(f"XGBoost R² Score: {r2:.4f}")
print(f"XGBoost Accuracy: {accuracy:.2f}%")