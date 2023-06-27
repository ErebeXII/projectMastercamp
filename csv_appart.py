import pandas as pd
import json

def appart(file, path_appart="VFappart.csv"):
    print("Creating appart csv from "+ file.split("\\")[-1] + "...")
    df = pd.read_csv(file, sep='|', header=0, low_memory=False)

    with open('Type local.json') as file:
        data = json.load(file)
    key_number = 0
    for key, val in data.items():
        if val == "Appartement":
            key_number = key

    df2 = df[df["Type local"] == int(key_number)]
    df2.to_csv(path_appart, sep='|', encoding='utf-8', header=True, index=False)
    print("Appart csv created !")

