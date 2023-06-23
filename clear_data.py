import os
import json

import numpy as np
import pandas as pd


def remplace_text(df,col):
    dict = {}
    t = list(df[col].unique())
    if 0 in t:
        t[t.index(0)], t[0] = t[0], t[t.index(0)]
    for i in t:
        dict[t.index(i)]=i
    json_file(col, dict)
    df[col] = df[col].map(t.index)

    return df


def remplace_date(df):
    df['Date mutation'] = pd.to_datetime(df['Date mutation'], format='%d/%m/%Y')
    df['Date mutation'] = df['Date mutation'].dt.strftime('%Y%m')
    df['Date mutation'] = df['Date mutation'].astype(int)

    return df


def string_to_float_number(df, col):
    df[col] = df[col].fillna("0,0")
    print(df[col])
    df[col] = df[col].str.replace(',', '.')
    df[col] = df[col].apply(lambda x: float(x) if x != '0' else 0)
    print(df[col])
    return df


def string_to_int(df, col):
    df[col] = df[col].fillna(0)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(0)
    df[col] = df[col].astype(int)
    return df
"""
df = remplace_text('Type de voie')
df = remplace_text('Nature mutation')
df = remplace_text('Nature culture')
df = remplace_text('Nature culture speciale')
df = remplace_text('B/T/Q')"""

#print(df.info())
#remplace_date()


def json_file(col, data):
    with open(col+'.json', 'w', encoding='utf-8') as fichier:
        json.dump(data, fichier, indent=4, ensure_ascii=False)
