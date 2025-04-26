import pandas as pd #type: ignore
import numpy as np #type: ignore

df = pd.read_csv('./Datasets/flight_delay_data_cleaned.csv')

df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

df['FL_MONTH'] = df['FL_DATE'].dt.month
df['FL_DAY'] = df['FL_DATE'].dt.day
df['FL_DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek  # Monday=0, Sunday=6
df['IS_WEEKEND'] = df['FL_DAY_OF_WEEK'].apply(lambda x: 1 if x >= 5 else 0)

df['CRS_DEP_HOUR'] = df['CRS_DEP_TIME'] // 100  # If time is 1345 → 13
df['CRS_ARR_HOUR'] = df['CRS_ARR_TIME'] // 100

def time_of_day(hour):
    if pd.isna(hour):
        return 'Unknown'
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['DEP_TIME_OF_DAY'] = df['CRS_DEP_HOUR'].apply(time_of_day)

df['ROUTE'] = df['ORIGIN'] + '_' + df['DEST']

# Flight Haul Type (Short, Medium, Long based on Distance)
def haul_type(distance):
    if distance < 500:
        return 'Short'
    elif distance < 1500:
        return 'Medium'
    else:
        return 'Long'

df['HAUL_TYPE'] = df['DISTANCE'].apply(haul_type)

# 3. (Optional) Airport Traffic Features
origin_counts = df['ORIGIN'].value_counts().to_dict()
df['ORIGIN_BUSY_SCORE'] = df['ORIGIN'].map(origin_counts)

dest_counts = df['DEST'].value_counts().to_dict()
df['DEST_BUSY_SCORE'] = df['DEST'].map(dest_counts)

# ----------------------------------------
# 4. Features List
features = [
    'FL_MONTH', 'FL_DAY', 'FL_DAY_OF_WEEK', 'IS_WEEKEND',
    'CRS_DEP_HOUR', 'CRS_ARR_HOUR', 'DEP_TIME_OF_DAY',
    'DISTANCE', 'HAUL_TYPE', 'ORIGIN_BUSY_SCORE', 'DEST_BUSY_SCORE',
    'CRS_ELAPSED_TIME', 'TAXI_OUT', 'TAXI_IN', 'AIR_TIME',
    'AIRLINE', 'ORIGIN', 'DEST', 'ROUTE'
]

# Your target variable
target = 'ARR_DELAY'

X = df[features]
y = df[target]

print(X.head())
print()
print()
print(y.head())

df.to_csv('./Datasets/feature_engineered_dataset.csv')