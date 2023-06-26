import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt


df = pd.read_csv("VFglobal.csv", sep='|', header=0, low_memory=False)

# Filtrer les types de voie qui ont moins de 1000 occurrences
min_occurrences = 1
filtered_df = df.groupby('Nature mutation').filter(lambda x: len(x) >= min_occurrences)

# Création d'une table de contingence entre le type de voie et la valeur foncière
contingency_table = pd.crosstab(df['Nature mutation'], df['Valeur fonciere'])

# Test du chi carré
chi2, p_value, _, _ = chi2_contingency(contingency_table)

alpha = 0.05  # Seuil de significativité

if p_value < alpha:
    print("Il existe une corrélation entre la nature mutation et la valeur foncière.")
else:
    print("Il n'existe pas de corrélation significative entre la nature mutation et la valeur foncière.")

# Calcul des moyennes de valeur foncière pour chaque type de voie
mean_values = df.groupby('Nature mutation')['Valeur fonciere'].median()

# Création du diagramme en barres
plt.bar(mean_values.index, mean_values.values)
plt.xlabel('Nature mutation')
plt.ylabel('Valeur foncière moyenne')
plt.title('Corrélation entre la nature mutation et la valeur foncière')
plt.show()
