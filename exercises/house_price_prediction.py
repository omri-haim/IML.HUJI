import sys

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)

    # Remove rows with prices that are non-positive and nan
    df = df.drop(df[df["price"] <= 0].index)
    df.dropna(subset=["price"], inplace=True)

    # Zip the two "yr_" columns into one column
    years = df.loc[:,"yr_built": "yr_renovated"]
    year = years.max(axis=1)
    df.drop(["yr_built", "yr_renovated"], axis=1, inplace=True)
    df.insert(14, "year", year, True)

    # Remove houses with exaggerated number of bedrooms or 0
    df = df.drop(df[df["bedrooms"] > 11].index)
    df = df.drop(df[df["bedrooms"] < 1].index)

    # Remove houses with non-integer number of bathrooms
    df = df.drop(df[df["bathrooms"] % 1 != 0].index)

    df.drop(["zipcode"], axis=1, inplace=True)

    design_mat = df.iloc[:, 3:]
    response = df.iloc[:, 2]
    return (design_mat, response)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    titles = X.columns.values.tolist()

    y = y.to_numpy()
    sy = np.std(y)
    muy = np.mean(y)
    y_ = y - muy

    cols = len(X.columns)
    rows = len(X)
    for i in range(cols):
        x = X.iloc[:, i].to_numpy()
        sx = np.std(x)
        mux = np.mean(x)
        x_ = x - mux
        cov_xy = y_.transpose() @ x_ / (rows-1)
        pearson = cov_xy/(sx*sy)
        plot_title = r""+ titles[i] + "$,\\quad \\rho = " + str(pearson) + "$"

        plt.scatter(x,y)
        plt.title(plot_title)
        plt.xlabel(titles[i])
        plt.ylabel("Prices")
        # plt.show()
        filename = output_path + titles[i] + "_vs_prices.png"
        plt.savefig(filename)
        plt.close()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    dataset_path = "Z:\My Drive\Courses\IML\IML.HUJI\datasets\house_prices.csv"
    design_mat, response = load_data(dataset_path)


    # Question 2 - Feature evaluation with respect to response
    save_path = "Z:\My Drive\Courses\IML\IML.HUJI\exercises\ex2\q2\\"
    feature_evaluation(design_mat, response, save_path)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(design_mat, response, 0.75)


    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    mse_loss = np.zeros((91,10))
    mse_means = np.zeros((91, 1))
    mse_stds = np.zeros((91, 1))
    train_X.insert(0, "response", train_y)

    for i in range(91):
        p = (i+10)/100
        for j in range(10):
            sample = train_X.sample(frac=p, axis=0)
            y = sample.iloc[:, 0]
            X = sample.iloc[:, 1:]
            reg = LinearRegression()
            reg.fit(X.to_numpy(), y.to_numpy())
            mse_loss[i, j] = reg.loss(test_X.to_numpy(), test_y.to_numpy())
        mse_means[i] = np.mean(mse_loss[i,:])
        mse_stds[i] = np.std(mse_loss[i, :])

    conf_int_up = mse_means + 2*mse_stds
    conf_int_down = mse_means - 2 * mse_stds

    p = np.arange(10, 101, 1)
    plt.plot(p, mse_means)
    plt.fill_between(p, conf_int_down.flatten(), conf_int_up.flatten(),
                     color='b', alpha=0.1)
    plt.xlabel("Percentage of Training Set")
    plt.ylabel("MSE Loss of Test Set Sampled")
    plt.title("Mean Loss as a function of\nTraining Set Sample Percentage")
    plt.grid(visible=True, which='both', alpha=0.5)
    plt.show()


