import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

df = pd.read_csv("VFglobal.csv", sep='|', header=0, low_memory=False)

# Create a subset of the DataFrame with street numbers that appear at least 1000 times
street_counts = df['No voie'].value_counts()
valid_street_numbers = street_counts[street_counts >= 1000].index
filtered_df = df[df['No voie'].isin(valid_street_numbers)]

# Create the contingency table between street number and property value
contingency_table = pd.crosstab(filtered_df['No voie'], filtered_df['Valeur fonciere'])

# Perform the chi-square test
chi2, p_value, _, _ = chi2_contingency(contingency_table)

alpha = 0.05  # Significance level

if p_value < alpha:
    print("Il existe une corrélation entre le numéro voie et la valeur foncière.")
else:
    print("Il n'existe pas de corrélation significative entre le numéro voie et la valeur foncière.")

# Calculate the median property value for each street number
mean_values = filtered_df.groupby('No voie')['Valeur fonciere'].median()

# Create the bar chart
plt.bar(mean_values.index, mean_values.values)
plt.xlabel('No voie')
plt.ylabel('Valeur foncière moyenne')
plt.title('Corrélation entre le numéro voie et la valeur foncière')
plt.show()
