import pandas as pd #type: ignore

df1 = pd.read_csv('./Datasets/flight_delay_data_cleaned.csv')
df2 = pd.read_csv('./Datasets/feature_engineered_dataset.csv')


print(len(df1))
print(len(df2))