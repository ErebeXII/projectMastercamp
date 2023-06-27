import json
import os.path

import sklearn as sk
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
# decision tree
from sklearn.tree import DecisionTreeRegressor
# random forest
from sklearn.ensemble import RandomForestRegressor
# gradient boosting
from sklearn.ensemble import GradientBoostingRegressor
# support vector regression
from sklearn.svm import SVR
# neural network
from sklearn.neural_network import MLPRegressor
# ridge regression
from sklearn.linear_model import Ridge

# clustering models
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt


def modelTraining(path, cols):

    # we create df with the columns we want
    # print the part after the last \ of the path
    print("Training model on " + path.split("\\")[-1] + " ...")
    df = pd.read_csv(path, usecols=cols, sep='|', header=0, low_memory=False)

    # Handling missing values
    df = df.dropna()  # Remove rows with missing values

    # we create the model
    # SVR() takes too much time to run
    # LinearRegression(), MLPRegressor() and Ridge() are bad models
    # DecisionTreeRegressor(), GradientBoostingRegressor() are good models | GradientBoostingRegressor() is faster
    # RandomForestRegressor() is the best model but it takes a lot of time to run

    # limit the time of training for random forest
    rf = RandomForestRegressor(n_estimators=100,random_state=42)

    # cluster models, very poor prediction
    # KMeans(n_clusters=5)

    models = [DecisionTreeRegressor(), GradientBoostingRegressor(), LinearRegression(), MLPRegressor(), Ridge(), SVR(), rf]


    # we split the data with "valeur-fonciere" as target
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Valeur fonciere', axis=1),
                                                        df['Valeur fonciere'], test_size=0.3, random_state=0)

    # Handling missing values in training set
    X_train = X_train.fillna(X_train.mean())
    y_train = y_train.fillna(y_train.median())

    # Handling missing values in test set
    X_test = X_test.fillna(X_train.mean())
    y_test = y_test.fillna(y_train.median())

    for model in models:

        print("--------------------------------------------------")
        print("Model: ", model)
        print("Fitting model...")
        start_time = time.time()

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

        # print the time of training
        print("Training time: %s seconds" % (time.time() - start_time))

    # path to save the csv file is made from path but without the .csv
    path_to_save = path[:-4] + "_predicted_"
    # add the name of the model to the path without the extension
    path_to_save = path_to_save + str(model).split('(')[0] + '.csv'

    # if the file at path_to_save exists, we delete it
    if os.path.exists(path_to_save):
        os.remove(path_to_save)

    # create a new dataframe with the "Valeur fonciere" of the test set, the actual values and the predicted values
    df_pred = pd.DataFrame({'Valeur fonciere': y_test, 'Predicted': y_pred, 'Difference': y_test - y_pred})

    # create csv file with the predicted values with the name of the model
    df_pred.to_csv(path_to_save, index=False)


def determineClusterNumber(path, cols):
    print(f"Training on {path}")
    df = pd.read_csv(path, usecols=cols, sep='|', header=0, low_memory=False)

    # Handling missing values
    df = df.dropna()  # Remove rows with missing values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Valeur fonciere', axis=1),
                                                        df['Valeur fonciere'], test_size=0.2, random_state=0)

    # Elbow method
    max_clusters = 10
    elbow_scores = []
    silhouette_scores = []

    for n_clusters in range(1, max_clusters + 1):
        start_time = time.time()
        model = KMeans(n_clusters=n_clusters)
        model.fit(X_train)

        # Calculate WCSS (Within-Cluster Sum of Squares)
        wcss = model.inertia_
        elbow_scores.append(wcss)
        print(f"{model}\nTraining for {n_clusters} clusters\nTraining time: {time.time() - start_time} seconds")

    # Plot Elbow method
    plt.plot(range(1, max_clusters + 1), elbow_scores)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS (lower is better)')
    plt.show()


def modelTrainingPerDepartement(path, cols):
    print("Training model on " + path.split("\\")[-1] + " ...")
    df = pd.read_csv(path, usecols=cols, sep='|', header=0, low_memory=False)
    # Handling missing values
    df = df.dropna()  # Remove rows with missing values
    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100,random_state=42)
    # Determine which departements have the best prediction by R^2 score
    departements = df['Code departement'].unique()
    # drop the departement with less than 10000 rows
    for departement in departements:
        if len(df[df['Code departement'] == departement]) < 10000:
            df.drop(df[df['Code departement'] == departement].index, inplace=True)
    departements = df['Code departement'].unique()

    best_R2_per_dpt = [] # [(departement, R^2)]
    with open('./Code departement.json') as json_file: # to translate the departement code to the name
        data = json.load(json_file)

    for departement in departements:
        print("Training model on departement " + str(data[str(departement)]) + " ...")
        # create a dataframe with only the departement
        df_dpt = df[df['Code departement'] == departement]
        # we split the data with "valeur-fonciere" as target
        X_train, X_test, y_train, y_test = train_test_split(df_dpt.drop('Valeur fonciere', axis=1),
                                                            df_dpt['Valeur fonciere'], test_size=0.2, random_state=0)
        # we fit the model
        rf.fit(X_train, y_train)
        # we calculate R^2
        score = rf.score(X_test, y_test)
        # if the R^2 is better than the previous one, we keep it
        best_R2_per_dpt.append((str(departement), score))

    # we sort the list by R^2 score
    best_R2_per_dpt.sort(key=lambda x: x[1], reverse=True)
    print(best_R2_per_dpt)
    dpt = best_R2_per_dpt[0][0]

    print("Best R^2 score is for departement " + str(dpt) + " with a score of " + str(best_R2_per_dpt[0][1]) +"\n"
                                                    + "With " + str(len(df[df['Code departement'] == dpt])) + " rows")


    # Redo the training with the best departement to store in a csv file
    df_dpt = df[df['Code departement'] == best_R2_per_dpt[0][0]]
    print(df_dpt)
    # we split the data with "valeur-fonciere" as target
    X_train, X_test, y_train, y_test = train_test_split(df_dpt.drop('Valeur fonciere', axis=1),
                                                        df_dpt['Valeur fonciere'], test_size=0.2, random_state=0)
    # we fit the model
    rf.fit(X_train, y_train)
    # we predict the data
    y_pred = rf.predict(X_test)
    # we save in a csv file
    path_to_save = path[:-4] + "_predicted_" + str(dpt) + '.csv'
    # if the file at path_to_save exists, we delete it
    if os.path.exists(path_to_save):
        os.remove(path_to_save)
    # create a new dataframe with the "Valeur fonciere" of the test set, the actual values and the predicted values
    df_pred = pd.DataFrame({'Valeur fonciere': y_test, 'Predicted': y_pred, 'Difference': y_test - y_pred})



# we try with the following columns : 'No disposition', 'Date mutation', 'Nature mutation', 'Valeur fonciere', 'Code departement', 'No Volume', 'Type local', 'Surface reelle bati', 'Nature culture', 'Nature culture speciale', 'Surface terrain', 'Sum Surface Carrez', 'Prix Surface Carre'
cols = ['No disposition', 'Date mutation', 'Nature mutation', 'Valeur fonciere', 'Code departement', 'No Volume',
        'Type local', 'Surface reelle bati', 'Nature culture', 'Nature culture speciale', 'Surface terrain',
        'Sum Surface Carrez', 'Prix Surface Carre']


# r'C:\Users\timot\Documents\Python\Project_Mastercamp_DSVFglobal.csv'
# r'C:\Users\timot\Documents\Python\Project_Mastercamp_DS\2022.csv'

#modelTraining(r'C:\Users\timot\Documents\Python\Project_Mastercamp_DS\appart.csv',cols)

#determineClusterNumber(r'C:\Users\timot\Documents\Python\Project_Mastercamp_DS\2022.csv',cols )

modelTrainingPerDepartement(r'C:\Users\timot\Documents\Python\Project_Mastercamp_DS\appart.csv',cols)