import pandas as pd #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.ensemble import RandomForestRegressor #type: ignore
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

rf = RandomForestRegressor(
    n_estimators=100,    # Number of trees
    max_depth=15,        # Limit tree depth to avoid overfitting
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Starting training...")
rf.fit(X_train, y_train)
print("Training done.")

# --------------------------------
# Predictions
y_pred = rf.predict(X_test)

# --------------------------------
# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100  # Accuracy in %

print(f"Random Forest MSE: {mse:.2f}")
print(f"Random Forest R² Score: {r2:.4f}")
print(f"Random Forest Accuracy: {accuracy:.2f}%")