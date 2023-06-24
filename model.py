import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats


def linearRegression(path, cols):
    # we create df with the columns we want
    # timothé path : C:\Users\timot\Documents\Python\Project_Mastercamp_DS\VFglobal.csv
    df = pd.read_csv(path, usecols=cols, sep='|', header=0, low_memory=False)

    # Handling missing values
    df = df.dropna()  # Remove rows with missing values

    # we create the model
    model = LinearRegression()

    # we split the data with "valeur-fonciere" as target
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Valeur fonciere', axis=1),
                                                        df['Valeur fonciere'], test_size=0.2, random_state=0)


    # Handling missing values in training set
    X_train = X_train.fillna(X_train.mean())
    y_train = y_train.fillna(y_train.median())

    # Handling missing values in test set
    X_test = X_test.fillna(X_train.mean())
    y_test = y_test.fillna(y_train.median())


    model.fit(X_train, y_train)

    # we predict the data
    y_pred = model.predict(X_test)

    # we calculate R^2, RMSE, Speraman's correlation
    score = model.score(X_test, y_test)
    # the best possible score is 1.0
    print("R^2: ", score)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # the best possible score is 0.0
    print("RMSE: ", rmse)
    spearman = stats.spearmanr(y_test, y_pred)
    # the best possible score is 1.0 or -1.0
    print("Spearman's correlation: ", spearman)

    # we return a dict with the results
    return {'R^2': score, 'RMSE': rmse, 'Spearman': spearman}

# we try with the following columns : No disposition|Date mutation|Nature mutation|Valeur fonciere|No voie|Type de voie|Code postal|Code departement|Section|No Volume|1er lot|Nombre de lots|Type local|Identifiant local|Surface reelle bati|Nombre pieces principales|Nature culture|Nature culture speciale|Surface terrain|Sum Surface Carrez|Prix Surface Carre
cols = ['No disposition', 'Date mutation', 'Nature mutation', 'Valeur fonciere', 'No voie', 'Type de voie',
         'Code departement', 'Section', 'No Volume', '1er lot', 'Nombre de lots', 'Type local',
        'Identifiant local', 'Surface reelle bati', 'Nombre pieces principales', 'Nature culture',
        'Nature culture speciale', 'Surface terrain', 'Sum Surface Carrez', 'Prix Surface Carre']


# r'C:\Users\timot\Documents\Python\Project_Mastercamp_DS\VFglobal.csv'

linearRegression(r'C:\Users\timot\Documents\Python\Project_Mastercamp_DS\VFglobal.csv',cols)
