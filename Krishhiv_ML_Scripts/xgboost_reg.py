import pandas as pd #type: ignore
import xgboost as xgb #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import mean_squared_error, r2_score #type: ignore
from sklearn.preprocessing import LabelEncoder #type: ignore
import matplotlib.pyplot as plt #type: ignore

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
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='rmse'
)


print("Starting the training...")
eval_set = [(X_train, y_train), (X_test, y_test)]  # Training and validation data
xg_reg.fit(X_train, y_train, eval_set=eval_set, verbose=True)
print("Training done")


# Retrieve evaluation results
results = xg_reg.evals_result()

# Plotting train and validation loss
epochs = range(1, len(results['validation_0']['rmse']) + 1)
plt.plot(epochs, results['validation_0']['rmse'], label='Train RMSE')
plt.plot(epochs, results['validation_1']['rmse'], label='Validation RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('XGBoost Training vs Validation RMSE')
plt.legend()
plt.grid(True)
plt.show()


y_pred = xg_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100

print(f"XGBoost MSE: {mse:.2f}")
print(f"XGBoost R² Score: {r2:.4f}")
print(f"XGBoost Accuracy: {accuracy:.2f}%")