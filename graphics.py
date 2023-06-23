import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
