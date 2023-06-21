import pandas as pd


def create_global_csv(path):
    # Initialize an empty list
    chunks = []

    # Iterate over the files and append chunks to the list
    for i in range(18, 23):
        filename = f"valeursfoncieres-20{i}.txt"
        columns_to_drop = ['Identifiant de document', 'Reference document', '1 Articles CGI', '2 Articles CGI',
                           '3 Articles CGI',
                           '4 Articles CGI', '5 Articles CGI', 'Code voie', 'Voie', 'Commune',
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
    df.to_csv(path, sep='|', encoding='utf-8', header=True, index=False)

create_global_csv(r'C:\Users\timot\OneDrive\Documents\L3\machine_learning\projectMastercamp\VFglobal.csv')

df = pd.read_csv("VFglobal.csv", sep='|', header=0, low_memory=False)
print(df.shape)