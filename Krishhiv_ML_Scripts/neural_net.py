import pandas as pd #type: ignore
import numpy as np #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from keras.layers import Dropout #type: ignore
from sklearn.metrics import mean_squared_error, r2_score #type: ignore
from sklearn.preprocessing import LabelEncoder #type: ignore
from keras.callbacks import EarlyStopping #type: ignore
from keras import regularizers #type: ignore
import matplotlib.pyplot as plt #type: ignore

df = pd.read_csv('./Datasets/feature_engineered_dataset.csv')

# Drop unnecessary columns
drop_cols = ['Unnamed: 0', 'FL_DATE', 'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE', 
             'ORIGIN_CITY', 'DEST_CITY', 'DEP_TIME', 'ARR_TIME', 'CANCELLED', 'DIVERTED', 'FL_NUMBER']
df = df.drop(columns=drop_cols)

# Encode categorical variables
categorical_cols = ['AIRLINE', 'ORIGIN', 'DEST', 'DEP_TIME_OF_DAY', 'ROUTE', 'HAUL_TYPE']

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save the encoder for inverse transformation if needed

# Separate features and target
X = df.drop(columns=['ARR_DELAY'])
y = df['ARR_DELAY']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Define EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train, 
    epochs=100,          # Increased max epochs so early stopping can kick in
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stopping],
    verbose=1
)

y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100  # Accuracy in %

print(f"Neural Network MSE: {mse:.2f}")
print(f"Neural Network R² Score: {r2:.4f}")
print(f"Neural Network Accuracy: {accuracy:.2f}%")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()