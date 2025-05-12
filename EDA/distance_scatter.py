import pandas as pd #type: ignore
import seaborn as sns #type: ignore
import matplotlib.pyplot as plt #type: ignore

df = pd.read_csv('./Datasets/feature_engineered_dataset.csv')

# Scatter plot of Distance vs Arrival Delay
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='DISTANCE', y='ARR_DELAY', alpha=0.4)
plt.title('Flight Distance vs Arrival Delay')
plt.xlabel('Distance (miles)')
plt.ylabel('Arrival Delay (minutes)')
plt.grid(True)
plt.show()
