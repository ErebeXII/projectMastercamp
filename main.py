import pandas as pd
from clear_data import remplace_text, remplace_date, string_to_float_number, string_to_int


def create_global_csv(path):
    # Initialize an empty list
    chunks = []

    # Iterate over the files and append chunks to the list
    for i in range(18, 23):
        print(i)
        filename = f"valeursfoncieres-20{i}.txt"
        columns_to_drop = ['Identifiant de document', 'Reference document', '1 Articles CGI', '2 Articles CGI',
                           '3 Articles CGI',
                           '4 Articles CGI', '5 Articles CGI', 'Code voie', 'Voie', 'Commune','Code departement',
                           'Identifiant local','Type local']
        chunk = pd.read_csv(filename, sep='|', header=0, low_memory=False)
        # drop unwanted columns
        chunk.drop(columns_to_drop, axis=1, inplace=True)
        # drop rows with all NaN values
        chunk.dropna(how='all', inplace=True)
        # drop duplicates
        chunk.drop_duplicates(inplace=True)

        chunks.append(chunk)

    # Concatenate the chunks into a single DataFrame

    df = pd.concat(chunks)
    df = df.fillna(0)
    df = remplace_text(df, 'Type de voie')
    df = remplace_text(df, 'Nature mutation')
    df = remplace_text(df, 'Nature culture')
    df = remplace_text(df, 'Nature culture speciale')
    df = remplace_text(df, 'B/T/Q')
    df = remplace_text(df, 'Section')
    df = string_to_int(df, 'No Volume')
    df = string_to_int(df, '1er lot')
    df = string_to_int(df, '2eme lot')
    df = string_to_int(df, '3eme lot')
    df = string_to_int(df, '4eme lot')
    df = string_to_int(df, '5eme lot')
    df = remplace_date(df)
    df = string_to_float_number(df, 'Valeur fonciere')
    df = string_to_float_number(df, 'Surface Carrez du 1er lot')
    df = string_to_float_number(df, 'Surface Carrez du 2eme lot')
    df = string_to_float_number(df, 'Surface Carrez du 3eme lot')
    df = string_to_float_number(df, 'Surface Carrez du 4eme lot')
    df = string_to_float_number(df, 'Surface Carrez du 5eme lot')

    df.to_csv(path, sep='|', encoding='utf-8', header=True, index=False)

create_global_csv(r'C:\Users\nothy\PycharmProjects\projectMastercamp\VFglobal.csv')

df = pd.read_csv("C:\\Users\\nothy\\PycharmProjects\\projectMastercamp\\VFglobal.csv", sep='|', header=0, low_memory=False)
print(df.info())
df_corr = df.corr()
print("grande correlation", df_corr.unstack()[(df_corr.unstack() < 1)].nlargest(20))

print(df_corr)
print(df.shape)

print(df_corr.loc['Valeur fonciere'])