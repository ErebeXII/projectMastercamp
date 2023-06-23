import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("VFglobal.csv", sep='|', header=0, low_memory=False)

data = df[['Code postal', 'Valeur fonciere']]

value_counts = data['Code postal'].value_counts()

top_categories = value_counts.head(10).index.tolist()

data.loc[~data['Code postal'].isin(top_categories), 'Code postal'] = 'Autre'

mean_values = data.groupby('Code postal')['Valeur fonciere'].mean()

plt.figure(figsize=(12, 6))
mean_values.plot(kind='bar', color='blue')
plt.xlabel('Code postal')
plt.ylabel('Valeur foncière moyenne')
plt.title('Comparaison de la valeur foncière par code postal')
plt.xticks(rotation=45)
plt.show()
