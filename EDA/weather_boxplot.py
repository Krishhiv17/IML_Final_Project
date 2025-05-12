import pandas as pd #type: ignore
import seaborn as sns #type: ignore
import matplotlib.pyplot as plt #type: ignore

df = pd.read_csv('./Datasets/feature_engineered_dataset.csv')

df['WEATHER_DELAY_BIN'] = pd.cut(
    df['DELAY_DUE_WEATHER'],
    bins=[-1, 0, 5, 15, 60, float('inf')],
    labels=['No Delay', 'Light (1-5)', 'Moderate (6-15)', 'Heavy (16-60)', 'Extreme (>60)']
)

# Plot boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='WEATHER_DELAY_BIN', y='ARR_DELAY', palette='Set2')
plt.title('Arrival Delay by Weather Delay Severity')
plt.xlabel('Weather Delay Category')
plt.ylabel('Arrival Delay (minutes)')
plt.grid(True)
plt.show()
