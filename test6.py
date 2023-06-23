import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("VFglobal.csv", sep='|', header=0, low_memory=False)

data = df[['Surface reelle bati', 'Valeur fonciere']]

value_counts = data['Surface reelle bati'].value_counts()

top_categories = value_counts.head(10).index.tolist()

data.loc[~data['Surface reelle bati'].isin(top_categories), 'Surface reelle bati'] = 'Autre'

mean_values = data.groupby('Surface reelle bati')['Valeur fonciere'].mean()

plt.figure(figsize=(12, 6))
mean_values.plot(kind='bar', color='blue')
plt.xlabel('Surface reelle bati')
plt.ylabel('Valeur foncière moyenne')
plt.title('Comparaison de la valeur foncière par surface reelle bati')
plt.xticks(rotation=45)
plt.show()