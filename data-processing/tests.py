import pandas as pd #type: ignore
import seaborn as sns #type: ignore
import matplotlib.pyplot as plt #type: ignore

df = pd.read_csv('./Datasets/feature_engineered_dataset.csv')

plt.figure(figsize=(10, 6))
plt.hist(df['ARR_DELAY'].dropna(), bins=100, color='skyblue', edgecolor='black')
plt.title('Distribution of Arrival Delays')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Number of Flights')
plt.xlim(-100, 300)  # Focused range to ignore outliers (adjust if needed)
plt.grid(True)
plt.show()