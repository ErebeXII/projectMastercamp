import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("VFglobal.csv", sep='|', header=0, low_memory=False)

# Create a subset of the DataFrame with street numbers that appear at least 1000 times
street_counts = df['1er lot'].value_counts()
valid_street_numbers = street_counts[street_counts >= 1000].index
filtered_df = df[df['1er lot'].isin(valid_street_numbers)]

# Normalize the property values using Min-Max scaling
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(filtered_df['Valeur fonciere'].values.reshape(-1, 1))

# Create the contingency table between street number and normalized property value
contingency_table = pd.crosstab(filtered_df['1er lot'], normalized_values.flatten())

# Perform the chi-square test
chi2, p_value, _, _ = chi2_contingency(contingency_table)

print(chi2)
print(p_value)

alpha = 0.005  # Significance level

if p_value < alpha:
    print("Il existe une corrélation entre le 1er lot et la valeur foncière (après normalisation).")
else:
    print("Il n'existe pas de corrélation significative entre le 1er lot et la valeur foncière (après normalisation).")

# Calculate the median property value for each street number
mean_values = filtered_df.groupby('1er lot')['Valeur fonciere'].median()

# Create the bar chart
plt.bar(mean_values.index, mean_values.values)
plt.xlabel('1er lot')
plt.ylabel('Valeur foncière moyenne')
plt.title('Corrélation entre le 1er lot et la valeur foncière (après normalisation)')
plt.show()
