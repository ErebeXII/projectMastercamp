import pandas as pd
import numpy as np
from clear_data import remplace_text, remplace_date, string_to_float_number, string_to_int
import os
import graphics
import create_square_meter_price
import csv_appart

def create_global_csv(path, txt_path=""):
    print("Creating global csv...")
    # Initialize an empty list
    chunks = []

    # Iterate over the files and append chunks to the list
    for i in range(18, 23):
        print(f"Processing file: valeursfoncieres-20{i}.txt")

        filename = f"{txt_path}valeursfoncieres-20{i}.txt"

        columns_to_drop = ['Identifiant de document', 'Reference document', '1 Articles CGI', '2 Articles CGI',
                           '3 Articles CGI', '4 Articles CGI', '5 Articles CGI', 'Code voie',
                           'Voie', 'Commune', 'Code postal', 'Code commune', 'Code type local',
                           '2eme lot', '3eme lot', '4eme lot', '5eme lot', 'B/T/Q', 'No plan']
        chunk = pd.read_csv(filename, sep='|', header=0, low_memory=False)
        # drop unwanted columns
        chunk.drop(columns_to_drop, axis=1, inplace=True)
        # drop rows with all NaN values
        chunk.dropna(how='all', inplace=True)
        # drop duplicates
        chunk.drop_duplicates(inplace=True)

        chunks.append(chunk)

    # Concatenate the chunks into a single DataFrame
    print("Cleaning non numeric values...")
    df = pd.concat(chunks)
    # df = df.fillna(0)

    for col in ['Type de voie', 'Nature mutation', 'Nature culture',
                'Nature culture speciale', 'Section', 'Type local', '1er lot',
                'Prefixe de section', 'Code departement']:
        df = remplace_text(df, col)

    for col in ['No Volume', 'No voie']:
        df = string_to_int(df, col)

    for col in ['Valeur fonciere', 'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot',
                'Surface Carrez du 3eme lot', 'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot']:
        df = string_to_float_number(df, col)

    df = remplace_date(df)
    df = df.fillna(0)

    create_square_meter_price.create(df)

    df.to_csv(path, sep='|', encoding='utf-8', header=True, index=False)
    print("Global csv created !")


def create_year_csv(path, year, txt_path=""):
    # Initialize an empty list
    chunks = []

    print(f"Processing file: valeursfoncieres-{year}.txt")

    filename = f"{txt_path}valeursfoncieres-{year}.txt"

    columns_to_drop = ['Identifiant de document', 'Reference document', '1 Articles CGI', '2 Articles CGI',
                       '3 Articles CGI', '4 Articles CGI', '5 Articles CGI', 'Code voie',
                       'Voie', 'Code postal', 'Code commune', 'Code type local',
                       '2eme lot', '3eme lot', '4eme lot', '5eme lot', 'B/T/Q', 'No plan', 'No voie', 'Section',
                       '1er lot', 'Prefixe de section']
    df = pd.read_csv(filename, sep='|', header=0, low_memory=False)
    # drop unwanted columns
    df.drop(columns_to_drop, axis=1, inplace=True)
    # drop rows with all NaN values
    df.dropna(how='all', inplace=True)
    # drop duplicates
    df.drop_duplicates(inplace=True)

    # Concatenate the chunks into a single DataFrame
    print("Cleaning non numeric values...")

    for col in ['Type de voie', 'Nature mutation', 'Nature culture',
                'Nature culture speciale','Type local', 'Code departement', 'Commune']:
        df = remplace_text(df, col)

    for col in ['No Volume']:
        df = string_to_int(df, col)

    for col in ['Valeur fonciere', 'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot',
                'Surface Carrez du 3eme lot', 'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot']:
        df = string_to_float_number(df, col)

    df = remplace_date(df)
    df = df.fillna(0)

    create_square_meter_price.create(df)

    df.to_csv(path, sep='|', encoding='utf-8', header=True, index=False)
    print(f"{year} csv created !")


# path = r'C:\Users\meder\PycharmProjects\projectMastercamp2\VFglobal.csv'
#path = r'C:\Users\nothy\PycharmProjects\projectMastercamp\VFglobal.csv'
path = r'C:\Users\timot\Documents\Python\Project_Mastercamp_DS\VFglobal.csv'

if not os.path.exists(path):
    # txt_path is only needed if the txt files are not in the same folder as the script
    # txt_path="C:\\Users\\timot\\Documents\\Python\\Project_Mastercamp_DS\\
    #create_global_csv(path, txt_path="C:\\Users\\timot\\Documents\\Python\\Project_Mastercamp_DS\\")
    create_global_csv(path, txt_path="C:\\Users\\timot\\Documents\\Python\\Project_Mastercamp_DS\\")

path2 = r'C:\Users\timot\Documents\Python\Project_Mastercamp_DS\2022.csv'
#path2 = r'C:\Users\nothy\PycharmProjects\projectMastercamp\VF2022.csv'
#
# if not os.path.exists(path2):
#     create_year_csv(path, 2022, txt_path="C:\\Users\\timot\\Documents\\Python\\Project_Mastercamp_DS\\")
#     #create_year_csv(path, 2022)

path_appart = r'C:\Users\timot\Documents\Python\Project_Mastercamp_DS\appart.csv'

if not os.path.exists(path_appart):
    csv_appart.appart(path, path_appart=path_appart)

# df = pd.read_csv(path, sep='|', header=0, low_memory=False)
# graphics.plot_corr(df, 0.3)

# print(df.info())
# df_corr = df.corr()
# print("grande correlation", df_corr.unstack()[(df_corr.unstack() < 1)].nlargest(20))
#
# print(df_corr)
# print(df.shape)
#
# print(df_corr.loc['Valeur fonciere'])
path3 = r"C:\Users\timot\Documents\Python\Project_Mastercamp_DS\appart_predicted_51.csv"
if os.path.exists(path3):
    graphics.plotPredictedValues(path3, 1000000)

path4 = r"C:\Users\timot\Documents\Python\Project_Mastercamp_DS\appart_predicted_03.csv"
if os.path.exists(path4):
    graphics.plotPredictedValues(path4, 1000000)

path5 = r"C:\Users\timot\Documents\Python\Project_Mastercamp_DS\appart_predicted_59.csv"
if os.path.exists(path5):
    graphics.plotPredictedValues(path5, 1000000)