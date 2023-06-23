import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("VFglobal.csv", sep='|', header=0, low_memory=False)

data = df[['Type de voie', 'Valeur fonciere']]

value_counts = data['Type de voie'].value_counts()

top_categories = value_counts.head(10).index.tolist()

data.loc[~data['Type de voie'].isin(top_categories), 'Type de voie'] = 'Autre'

mean_values = data.groupby('Type de voie')['Valeur fonciere'].mean()

plt.figure(figsize=(12, 6))
mean_values.plot(kind='bar', color='blue')
plt.xlabel('Type de voie')
plt.ylabel('Valeur foncière moyenne')
plt.title('Comparaison de la valeur foncière par type de voie')
plt.xticks(rotation=45)
plt.show()
