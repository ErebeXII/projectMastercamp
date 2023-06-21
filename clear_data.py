import numpy as np
from main import df


def remplace_text(col):
    df[col] = df[col].fillna(0)
    df[col] = df[col].replace(np.nan, 0)
    t = list(df[col].unique())
    if 0 in t:
        t[t.index(0)], t[0] = t[0], t[t.index(0)]
    print(t)
    df[col] = df[col].map(t.index)
    return df


def remplace_date():
    df['Date mutation'] = pd.to_datetime(df['Date mutation'], format='%d/%m/%Y')
    df['Date mutation'] = df['Date mutation'].dt.strftime('%Y%m')
    df['Date mutation'] = df['Date mutation'].astype(int)
    return df

"""
df = remplace_text('Type de voie')
df = remplace_text('Nature mutation')
df = remplace_text('Nature culture')
df = remplace_text('Nature culture speciale')
df = remplace_text('B/T/Q')"""

print(df.info())
#remplace_date()
