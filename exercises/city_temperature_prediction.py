import matplotlib.cm

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=True)
    df = df.drop(df[df["Temp"] < -30].index)
    DayOfYear = df.loc[:,"Date"]
    df.loc[:,"Date"] = pd.to_datetime(df.loc[:,"Date"])
    df["DayOfYear"] = (df.loc[:,"Date"]).dt.dayofyear
    response = df["Temp"]
    design = df.drop(["Temp"], axis=1)
    return design, response



if __name__ == '__main__':
    np.random.seed(0)

    #%% Question 1 - Load and preprocessing of city temperature dataset
    file_path = "Z:\My Drive\Courses\IML\IML.HUJI\datasets\City_Temperature.csv"
    design_mat, response = load_data(file_path)


    #%% Question 2 - Exploring data for specific country
    design_mat.insert(0, "response", response)
    design_with_resp = design_mat.drop(design_mat[design_mat["Country"] !=
                                                  "Israel"].index)

    # response = response.loc[response.index.isin(design_with_resp.index)]
    response_il = design_with_resp.iloc[:, 0]
    design_il = design_with_resp.iloc[:, 1:]

    # First Sub-question
    years = design_il["Year"].unique()
    cmap = matplotlib.cm.get_cmap("jet", len(years))
    plt.scatter(design_il["DayOfYear"], response_il, c=design_il["Year"], cmap=cmap,
                alpha=0.5, s=2, zorder=2)
    cbar = plt.colorbar(ticks=years)
    plt.xlabel("Day of the Year")
    plt.ylabel("Temperature [C$\degree$]")
    plt.title("Measured Temperature in Tel Aviv, Israel\nas a function of the day of "
              "the year")
    plt.grid(visible=True, which='both', alpha=0.5, zorder=0)
    plt.show()

    # Second Sub-question
    monthly_std = design_with_resp.groupby('Month').agg('std')
    monthly_std["Month"] = np.arange(1,13)
    plt.bar(x="Month", height="response", data=monthly_std, tick_label=monthly_std[
        "Month"], zorder=2)
    plt.title("Standard Deviation of Daily Temperature\nin Tel Aviv, Israel by Month")
    plt.xlabel("Month")
    plt.ylabel("STD of Daily Temperature [C$\degree$]")
    plt.grid(visible=True, which='both', alpha=0.5, zorder=0)
    plt.show()


    #%% Question 3 - Exploring differences between countries
    countries = design_mat["Country"].unique()
    country_monthly = design_mat.groupby(['Country', 'Month']).agg({
        'response': ['mean', 'std']})
    months = np.arange(1, 13)

    for country in countries:
        means = country_monthly["response"].T[country].loc["mean",:]
        stds = country_monthly["response"].T[country].loc["std",:]
        plt.errorbar(x=months, y=means, yerr=stds, label=country, capsize=3, alpha=0.5,
                     zorder=2)

    plt.xticks(ticks=months, labels=months)
    plt.ylim(-1,35)
    plt.legend(loc="best")
    plt.title("Average Daily Temperature by Month\nin Different Countries")
    plt.xlabel("Month")
    plt.ylabel("Average Daily Temperature [C$\degree$]")
    plt.grid(visible=True, which='both', alpha=0.5, zorder=0)
    plt.show()


    #%% Question 4 - Fitting model for different values of `k`
    X_il = pd.DataFrame(design_il["DayOfYear"])

    train_il_X, train_il_y, test_il_X, test_il_y \
        = split_train_test(X_il, response_il, train_proportion=0.75)
    losses = np.zeros(10)
    for k in range(1,11):
        poly_fit = PolynomialFitting(k=k)
        poly_fit.fit(train_il_X.to_numpy().flatten(), train_il_y.to_numpy().flatten())
        losses[k-1] = np.round(poly_fit.loss(test_il_X.to_numpy().flatten(),
                                    test_il_y.to_numpy().flatten()), 2)
        print("k =", str(k), ", loss =", losses[k - 1])


    plt.bar(x=np.arange(1,11), height=losses, tick_label=np.arange(1,11), zorder=2)
    plt.title("Test Error as a Function of Polynomial Fit Degree\nof Temperature vs. "
              "Day of the Year")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE of fit")
    plt.grid(visible=True, which='both', alpha=0.5, zorder=0)
    plt.show()

    #%% Question 5 - Evaluating fitted model on different countries
    k = 5
    countries = countries[countries != "Israel"]
    train_X = (design_il.loc[:, "DayOfYear"]).to_numpy().flatten()
    train_y = response_il.to_numpy().flatten()
    country_err = np.zeros(3)
    poly_fit = PolynomialFitting(k=k)
    poly_fit.fit(train_X, train_y)
    n = 0
    for country in countries:
        country_X = \
            design_mat.drop(design_mat[design_mat["Country"] != country].index)
        country_y = (country_X.loc[:, "response"]).to_numpy().flatten()
        country_X = (country_X.loc[:, "DayOfYear"]).to_numpy().flatten()
        country_err[n] = poly_fit.loss(country_X, country_y)
        n += 1

    plt.bar(x=countries, height=country_err, tick_label=countries, zorder=2)
    plt.title("Polynomial Regression MSE Error for\nvarious countries as test set")
    plt.xlabel("Country")
    plt.ylabel("MSE Loss")
    plt.grid(visible=True, which='both', alpha=0.5, zorder=0)
    plt.show()
