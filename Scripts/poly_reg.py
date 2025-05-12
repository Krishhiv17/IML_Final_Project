import pandas as pd #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler #type: ignore
from sklearn.linear_model import SGDRegressor #type: ignore
from sklearn.metrics import mean_squared_error, r2_score #type: ignore

# Load dataset
df = pd.read_csv('./Datasets/feature_engineered_dataset.csv')

# Drop unnecessary columns
drop_cols = ['Unnamed: 0', 'FL_DATE', 'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE',
             'ORIGIN_CITY', 'DEST_CITY', 'DEP_TIME', 'ARR_TIME', 'CANCELLED', 'DIVERTED', 'FL_NUMBER', 'TOTAL_KNOWN_DELAY_CAUSE', 'DEP_DELAY']
df = df.drop(columns=drop_cols)

# Categorical columns encoding
categorical_cols = ['AIRLINE', 'ORIGIN', 'DEST', 'DEP_TIME_OF_DAY', 'ROUTE', 'HAUL_TYPE']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target variable
X = df.drop(columns=['ARR_DELAY'])
y = df['ARR_DELAY']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Polynomial transformation on scaled data
poly = PolynomialFeatures(degree=2)  # You can adjust the degree as needed
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train SGDRegressor model (stochastic gradient descent)
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3)
sgd_regressor.fit(X_train_poly, y_train)

# Predict on both training and test sets
y_train_pred = sgd_regressor.predict(X_train_poly)
y_test_pred = sgd_regressor.predict(X_test_poly)

# Evaluate the model on training data
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Training MSE: {train_mse}')
print(f'Training R^2: {train_r2}')

# Evaluate the model on test data
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f'Test MSE: {test_mse}')
print(f'Test R^2: {test_r2}')
