import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

shapefile_path = 'C:\\Users\\nothy\\Downloads\\departements-et-collectivites-doutre-mer-france@toursmetropole\\georef-france-departement-millesime.shp'
map = gpd.read_file(shapefile_path)

# Charger les données du prix de surface par département
data = pd.read_csv("../VF2022.csv", delimiter="|")

print(data.columns)

index = []

del_codes = ['971', '972', '973', '974', '976']

for i, code in map['dep_code'].items():
    code_str = code.strip("[]'")  # Supprimer les crochets et les apostrophes
    print(code_str)
    if code_str in del_codes:
        index.append(i)

# Calculer le prix moyen du mètre carré par département
prix_moyen_par_departement = data.groupby(data['Code departement'].astype(str))["Prix Surface Carre"].mean().reset_index()

# Convertir la colonne "Code departement" en valeurs numériques
prix_moyen_par_departement['Code departement'] = pd.to_numeric(prix_moyen_par_departement['Code departement'])

# Trier par ordre croissant du code département
prix_moyen_par_departement = prix_moyen_par_departement.sort_values('Code departement')

print(prix_moyen_par_departement)

map = map.drop(index)
map = map.sort_values('dep_code')

print(map['dep_code'])

# Fusionner les données géographiques avec les données du prix de surface par département
merged = map.merge(prix_moyen_par_departement, left_on='dep_code', right_on='Code departement', how='left')

print(merged)

"""""
map = map.drop(index)
print(map)
map.plot()
plt.show()
"""