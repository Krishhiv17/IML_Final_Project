import pandas as pd #type: ignore
import seaborn as sns #type: ignore
import matplotlib.pyplot as plt #type: ignore

df = pd.read_csv('./Datasets/feature_engineered_dataset.csv')

numeric_df = df.select_dtypes(include=['number'])

# Now compute correlation
corr_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(14,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Numerical Features")
plt.show()