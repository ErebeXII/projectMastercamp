import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("VFglobal.csv", sep='|', header=0, low_memory=False)

data = df[['Type de voie', 'Valeur fonciere']]

value_counts = data['Type de voie'].value_counts()

# create a new category called 'other' for all categories with less than 1000 values
# data['Type de voie'] = data['Type de voie'].apply(lambda x: x if value_counts[x] > 1000 else '-1')

median = data.groupby('Type de voie')['Valeur fonciere'].median()
mean_values = data.groupby('Type de voie')['Valeur fonciere'].mean()

# order mean_values by descending order
mean_values = mean_values.sort_values(ascending=False)
median = median.sort_values(ascending=False)
print(mean_values)
print(median)

# create a dictionary with the index as keys and the values as values for the mean and median values
mean_dict = mean_values.to_dict()
median_dict = median.to_dict()

#replace the translation in the json file


# replace the values in the data frame with the median values
data['Type de voie'] = data['Type de voie'].replace(median_dict)

# correlation matrix between "Type de voie" and "Valeur fonciere"
corr_matrix = data.corr()
print(corr_matrix)

print(data)

# plot the correlation matrix
plt.plot(corr_matrix)


plt.show()
