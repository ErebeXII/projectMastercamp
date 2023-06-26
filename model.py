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
    # timothé path : C:\Users\timot\Documents\Python\Project_Mastercamp_DS\VFglobal.csv
    print(f"Training on {path}")
    df = pd.read_csv(path, usecols=cols, sep='|', header=0, low_memory=False)

    # Handling missing values
    df = df.dropna()  # Remove rows with missing values

    # we create the model
    # SVR() takes too much time to run
    # LinearRegression(), MLPRegressor() and Ridge() are bad models
    # DecisionTreeRegressor(), GradientBoostingRegressor() are good models | GradientBoostingRegressor() is faster
    # RandomForestRegressor() is the best model but it takes a lot of time to run

    # limit the time of training for random forest
    rf = RandomForestRegressor(n_estimators=10, random_state=42)

    # cluster models, very poor prediction
    # KMeans(n_clusters=5)

    models = [rf]


    # we split the data with "valeur-fonciere" as target
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Valeur fonciere', axis=1),
                                                        df['Valeur fonciere'], test_size=0.2, random_state=0)

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

    return 0


def determineClusterNumber(path, cols):
    print(f"Training on {path}")
    df = pd.read_csv(path, usecols=cols, sep='|', header=0, low_memory=False)

    # Handling missing values
    df = df.dropna()  # Remove rows with missing values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Valeur fonciere', axis=1),
                                                        df['Valeur fonciere'], test_size=0.2, random_state=0)

    # Elbow method
    max_clusters = 20
    elbow_scores = []
    silhouette_scores = []

    for n_clusters in range(1, max_clusters + 1):
        start_time = time.time()
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        kmeans.fit(X_train)

        # Calculate WCSS (Within-Cluster Sum of Squares)
        wcss = kmeans.inertia_
        elbow_scores.append(wcss)
        print(f"Training for {n_clusters} clusters, training time: {time.time() - start_time} seconds")

    # Plot Elbow method
    plt.plot(range(1, max_clusters + 1), elbow_scores)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS (lower is better)')
    plt.show()


# we try with the following columns : 'No disposition', 'Date mutation', 'Nature mutation', 'Valeur fonciere', 'Code departement', 'No Volume', 'Type local', 'Surface reelle bati', 'Nature culture', 'Nature culture speciale', 'Surface terrain', 'Sum Surface Carrez', 'Prix Surface Carre'
cols = ['No disposition', 'Date mutation', 'Nature mutation', 'Valeur fonciere', 'Code departement', 'No Volume',
        'Type local', 'Surface reelle bati', 'Nature culture', 'Nature culture speciale', 'Surface terrain',
        'Sum Surface Carrez', 'Prix Surface Carre']


# r'C:\Users\timot\Documents\Python\Project_Mastercamp_DSVFglobal.csv'
# r'C:\Users\timot\Documents\Python\Project_Mastercamp_DS\2022.csv'

modelTraining(r'C:\Users\timot\Documents\Python\Project_Mastercamp_DS\2022.csv',cols)

#determineClusterNumber(r'C:\Users\timot\Documents\Python\Project_Mastercamp_DS\2022.csv',cols )
