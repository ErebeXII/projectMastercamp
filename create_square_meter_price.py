import numpy as np

def create(df):
    df['Sum Surface Carrez'] = df['Surface Carrez du 1er lot'] + df['Surface Carrez du 2eme lot'] + df[
        'Surface Carrez du 3eme lot'] + df['Surface Carrez du 4eme lot'] + df['Surface Carrez du 5eme lot']

    df['Prix Surface Carre'] = np.where(df['Sum Surface Carrez'].ne(0),
                                        df['Valeur fonciere'] / df['Sum Surface Carrez'],
                                        df['Valeur fonciere'] / df['Surface reelle bati'])
    df['Prix Surface Carre'] = np.where(np.isinf(df['Prix Surface Carre']), 0, df['Prix Surface Carre'])

    df.drop(['Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot', 'Surface Carrez du 3eme lot',
             'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot'], axis=1, inplace=True)
