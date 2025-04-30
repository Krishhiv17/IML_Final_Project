import pandas as pd #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.ensemble import RandomForestRegressor #type: ignore
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

y_train_pred = rf.predict(X_train)

train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Random Forest Training MSE: {train_mse:.2f}")
print(f"Random Forest Training R² Score: {train_r2:.4f}")
print(f"Random Forest Training Accuracy: {train_r2 * 100:.2f}%")

y_pred = rf.predict(X_test)

test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"Random Forest Test MSE: {test_mse:.2f}")
print(f"Random Forest Test R² Score: {test_r2:.4f}")
print(f"Random Forest Test Accuracy: {test_r2 * 100:.2f}%")



# Metrics
datasets = ['Training', 'Test']
r2_scores = [train_r2 * 100, test_r2 * 100]  # Convert to %
mse_scores = [train_mse, test_mse]

# Plot R² Score (Accuracy)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(datasets, r2_scores, color=['skyblue', 'salmon'])
plt.ylim(0, 100)
plt.title('Random Forest Accuracy (R² %)')
plt.ylabel('Accuracy (%)')
for i, v in enumerate(r2_scores):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')

# Plot MSE
plt.subplot(1, 2, 2)
plt.bar(datasets, mse_scores, color=['skyblue', 'salmon'])
plt.title('Random Forest MSE')
plt.ylabel('Mean Squared Error')
for i, v in enumerate(mse_scores):
    plt.text(i, v + 0.5, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
