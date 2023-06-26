import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

shapefile_path = 'C:\\Users\\nothy\\Downloads\\departements-et-collectivites-doutre-mer-france@toursmetropole\\georef-france-departement-millesime.shp'
map_df = gpd.read_file(shapefile_path)

# Charger les données du prix de surface par département
data = pd.read_csv("../VF2022.csv", delimiter="|")

index = []

del_codes = ['971', '972', '973', '974', '976']

for i, code in map_df['dep_code'].items():
    code_str = code.strip("[]'")  # Supprimer les crochets et les apostrophes
    if code_str in del_codes:
        index.append(i)

# Supprimer les lignes avec les codes de département à supprimer
map_df = map_df.drop(index)

# Calculer le prix moyen du mètre carré par département
prix_moyen_par_departement = data.groupby(data['Code departement'].astype(str))["Prix Surface Carre"].mean().reset_index()

# Fusionner les données géographiques avec les données du prix de surface par département
merged = map_df.merge(prix_moyen_par_departement, left_on='dep_code', right_on='Code departement', how='left')

# Afficher le DataFrame fusionné
print(merged)

# Vérifier les colonnes disponibles dans le DataFrame fusionné
print(merged.columns)

# Tracer la géo carte
merged.plot(column='Prix Surface Carre', cmap='Blues', linewidth=0.8, edgecolor='0.8', legend=True)
plt.show()
