import pandas as pd
from clear_data import string_to_float_number

df = pd.read_csv("VFglobal.csv", sep='|', header=0, low_memory=False)

#print(df['Surface Carrez du 1er lot'])
df['Surface Carrez du 1er lot'] = df['Surface Carrez du 1er lot'].fillna(0)
df['Surface Carrez du 2eme lot'] = df['Surface Carrez du 1er lot'].fillna(0)
df['Surface Carrez du 3eme lot'] = df['Surface Carrez du 1er lot'].fillna(0)
df['Surface Carrez du 4eme lot'] = df['Surface Carrez du 1er lot'].fillna(0)
df['Surface Carrez du 5eme lot'] = df['Surface Carrez du 1er lot'].fillna(0)

df['Sum Surface Carrez'] = df['Surface Carrez du 1er lot'] + df['Surface Carrez du 2eme lot'] + df['Surface Carrez du 3eme lot'] +df['Surface Carrez du 4eme lot'] + df['Surface Carrez du 5eme lot']

print(df['Surface Carrez du 1er lot'])
print(df['Surface Carrez du 2eme lot'])
print(df['Surface Carrez du 3eme lot'])
print(df['Surface Carrez du 4eme lot'])
print(df['Surface Carrez du 5eme lot'])
print(df['Sum Surface Carrez'])
print(df['Surface reelle bati'])

if (df['Sum Surface Carrez'] + df['Surface reelle bati']).eq(0).all():
    df['Prix Surface Carre'] = 0
else:
    df['Prix Surface Carre'] = 1

print(df['Prix Surface Carre'])


"""
cpt = 0
if (df['Valeur fonciere']).eq(0).all():
    cpt = cpt + 1

print(cpt)
"""

"""
if (df['Sum Surface Carrez'] + df['Surface reelle bati']).eq(0).all():
    df['Prix Surface Carre'] = 0
elif df['Sum Surface Carrez'].eq(0).all():
    df['Prix Surface Carre'] = df['Valeur fonciere'] / df['Surface reelle bati']
else:
    df['Prix Surface Carre'] = df['Valeur fonciere'] / df['Sum Surface Carrez']

print(df['Prix Surface Carre'])
"""


"""
sum = df['Surface Carrez du 1er lot'] + df['Surface Carrez du 2eme lot'] + df['Surface Carrez du 3eme lot'] + df['Surface Carrez du 4eme lot'] + df['Surface Carrez du 5eme lot']

if sum.sum() == 0:
    df['Prix Surface Carre'] = df['Valeur fonciere'] / df['Surface reelle bati']
else:
    df['Prix Surface Carre'] = df['Valeur fonciere'] / sum

print(df['Prix Surface Carre'])
"""
