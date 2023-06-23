import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("VFglobal.csv", sep='|', header=0, low_memory=False)

data = df[['Nature mutation', 'Valeur fonciere']]

mean_values = data.groupby('Nature mutation')['Valeur fonciere'].mean()

plt.figure(figsize=(12, 6))
mean_values.plot(kind='bar', color='blue')
plt.xlabel('Nature de mutation')
plt.ylabel('Valeur foncière moyenne')
plt.title('Comparaison de la valeur foncière par nature de mutation')
plt.xticks(rotation=45)
plt.show()
