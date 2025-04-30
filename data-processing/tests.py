import pandas as pd #type: ignore
import seaborn as sns #type: ignore
import matplotlib.pyplot as plt #type: ignore

df = pd.read_csv('./Datasets/feature_engineered_dataset.csv')

correlation_with_target = df.corr(numeric_only=True)['ARR_DELAY'].sort_values(ascending=False)

print("Correlation of features with ARR_DELAY:")
print(correlation_with_target)
