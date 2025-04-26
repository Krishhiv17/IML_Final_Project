import pandas as pd #type: ignore

# Load the dataset
df = pd.read_csv("./Datasets/flight_delay_data.csv")

# Standardize column names (optional but recommended)
df.columns = df.columns.str.strip().str.upper()

# 1. Filter out cancelled and diverted flights
df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]

# 2. Drop rows where ARR_DELAY (target) is missing
df = df[df['ARR_DELAY'].notnull()]

# 3. Drop columns not useful or with many nulls
drop_cols = ['CANCELLATION_CODE', 'WHEELS_OFF', 'WHEELS_ON']
df = df.drop(columns=drop_cols)

# 4. Fill missing values in delay cause columns with 0
delay_cols = [
    'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
    'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT'
]
df[delay_cols] = df[delay_cols].fillna(0)

# 5. Optional: Create a total known delay cause column
df['TOTAL_KNOWN_DELAY_CAUSE'] = df[delay_cols].sum(axis=1)

print(len(df))

# Done — inspect cleaned data
print(df.info())
print(df.head())

print(df.isnull().values.any())

df.to_csv('./Datasets/flight_delay_data_cleaned.csv')