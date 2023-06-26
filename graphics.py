import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_corr(df, prec=0.8):
    corr = df.corr()
    # list the variables with a correlation greater than 0.8 or less than -0.8 and not equal to 1
    high_corr_var = np.where(np.logical_and(np.logical_or(corr > prec, corr < -prec), corr != 1))
    high_corr_var = [(corr.columns[x], corr.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y]
    # List of the columns that appear in high_corr_var
    columns = np.unique(np.array(high_corr_var).ravel())
    print(columns)
    # create a new dataframe with only the columns that appear in high_corr_var
    hdf = df[columns]
    # create a new correlation matrix
    corr = hdf.corr()
    # plot the heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation matrix')
    # add the column names as labels
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    # make the plot more readable
    plt.tight_layout()
    plt.show()

def plotPredictedValues(path, n_samples=1000):
    df = pd.read_csv(path, sep=',', header=0, low_memory=False)

    # take random n_samples rows from the dataframe
    # if the dataframe has less than n_samples rows, we take all the rows
    if df.shape[0] < n_samples:
        n_samples = df.shape[0]
    else:
        df = df.sample(n=n_samples)

    # plot the "Valeur fonciere" on the x-axis and the predicted values on the y-axis with a red line
    plt.figure(figsize=(20, 10))
    plt.plot(df['Valeur fonciere'], df['Predicted'], 'ro')
    plt.xlabel('Valeur fonciere')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted ({n_samples} samples)')
    # plot a regression line
    plt.plot(np.unique(df['Valeur fonciere']), np.poly1d(np.polyfit(df['Valeur fonciere'], df['Predicted'], 1)) \
    (np.unique(df['Valeur fonciere'])), color='black')
    # plot a dashed line of slope 1
    plt.plot(np.unique(df['Valeur fonciere']), np.unique(df['Valeur fonciere']), linestyle='--', color='blue', label='Ideal model')
    # add the legend
    plt.legend(['Predicted', 'Regression line', 'Ideal model'])


    # plot the difference between the two

    plt.figure(figsize=(20, 10))
    plt.plot(df['Valeur fonciere'], df['Difference'], 'ro')
    plt.xlabel('Valeur fonciere')
    plt.ylabel('Difference')
    plt.title(f'Difference between Actual and Predicted ({n_samples} samples)')

    # plot regression line
    plt.plot(np.unique(df['Valeur fonciere']), np.poly1d(np.polyfit(df['Valeur fonciere'], df['Difference'], 1)) \
    (np.unique(df['Valeur fonciere'])), color='black')
    # plot a dashed line of slope 0
    plt.plot(np.unique(df['Valeur fonciere']), np.zeros(len(np.unique(df['Valeur fonciere']))), linestyle='--', color='blue', label='Ideal model')
    # add the legend
    plt.legend(['Difference', 'Regression line', 'Ideal model'])

    plt.show()


