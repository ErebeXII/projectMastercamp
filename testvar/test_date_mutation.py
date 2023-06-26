import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt


df = pd.read_csv("VFglobal.csv", sep='|', header=0, low_memory=False)

# Création d'une table de contingence entre le type de voie et la valeur foncière
contingency_table = pd.crosstab(df['Date mutation'], df['Valeur fonciere'])

# Test du chi carré
chi2, p_value, _, _ = chi2_contingency(contingency_table)

alpha = 0.05  # Seuil de significativité

if p_value < alpha:
    print("Il existe une corrélation entre la date de mutation et la valeur foncière.")
else:
    print("Il n'existe pas de corrélation significative entre la date de mutation et la valeur foncière.")

# Calcul des moyennes de valeur foncière pour chaque type de voie
mean_values = df.groupby('Date mutation')['Valeur fonciere'].median()

# Création du diagramme en barres
plt.bar(mean_values.index, mean_values.values)
plt.xlabel('Date de mutation')
plt.ylabel('Valeur foncière moyenne')
plt.title('Corrélation entre la date de mutation et la valeur foncière')
plt.show()
