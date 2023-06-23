import pandas as pd
import numpy as np
from clear_data import string_to_float_number

df = pd.read_csv("VFglobal.csv", sep='|', header=0, low_memory=False)

df['Surface Carrez du 1er lot'] = df['Surface Carrez du 1er lot'].fillna(0)
df['Surface Carrez du 2eme lot'] = df['Surface Carrez du 1er lot'].fillna(0)
df['Surface Carrez du 3eme lot'] = df['Surface Carrez du 1er lot'].fillna(0)
df['Surface Carrez du 4eme lot'] = df['Surface Carrez du 1er lot'].fillna(0)
df['Surface Carrez du 5eme lot'] = df['Surface Carrez du 1er lot'].fillna(0)

df['Sum Surface Carrez'] = df['Surface Carrez du 1er lot'] + df['Surface Carrez du 2eme lot'] + df['Surface Carrez du 3eme lot'] + df['Surface Carrez du 4eme lot'] + df['Surface Carrez du 5eme lot']

print(df['Surface Carrez du 1er lot'])
print(df['Surface Carrez du 2eme lot'])
print(df['Surface Carrez du 3eme lot'])
print(df['Surface Carrez du 4eme lot'])
print(df['Surface Carrez du 5eme lot'])
print(df['Sum Surface Carrez'])
print(df['Surface reelle bati'])

sum_surface_reelle = df['Sum Surface Carrez'].fillna(0) + df['Surface reelle bati'].fillna(0)

df.loc[df['Sum Surface Carrez'].ne(0), 'Prix Surface Carre'] = df['Valeur fonciere'] / df['Sum Surface Carrez']
df.loc[df['Sum Surface Carrez'].eq(0), 'Prix Surface Carre'] = df['Valeur fonciere'] / df['Surface reelle bati']
df['Prix Surface Carre'] = np.where(np.isinf(df['Prix Surface Carre']), 0, df['Prix Surface Carre'])

print(df['Prix Surface Carre'])
