import functools
import pickle

import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from IMLearn.metrics.loss_functions import accuracy, misclassification_error


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    train_err = []
    test_err = []
    learners_vec = np.arange(1, n_learners + 1)

    for t in learners_vec:
        train_err.append(adaboost.partial_loss(train_X, train_y, t))
        test_err.append(adaboost.partial_loss(test_X, test_y, t))


    plt.plot(learners_vec, train_err, label='Train Error')
    plt.plot(learners_vec, test_err, label='Test Error')
    plt.legend()

    plt.xlabel("Number of Learners")
    plt.ylabel("Misclassification Error")
    plt.title("Misclassification Error of Training and Test Set\nas function of the "
              "number of weak learners")
    plt.grid()
    if noise == 0:
        plt.savefig('ex4/q1.png')
    else:
        plt.savefig('ex4/q5_1.png')
    plt.show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    if noise == 0:
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=[rf"$\text{{Ensemble Size = }}\textbf{{{m}}}$"
                                            for m in T],
                            horizontal_spacing=0.01, vertical_spacing=.03)

        for i, t in enumerate(T):
            prediction = lambda X0: adaboost.partial_predict(X0, T=t)
            fig.add_traces(
                [decision_surface(prediction, lims[0], lims[1], showscale=False),

                 go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                            showlegend=False,
                            marker=dict(color=test_y,
                                        symbol=class_symbols[np.int64((test_y + 1) / 2)],
                                        colorscale=[custom[0], custom[-1]],
                                        line=dict(color="black", width=1)))
                 ],
                rows=(i // 2) + 1, cols=(i % 2) + 1)

        fig.update_layout(
            title=rf"$\textbf{{Decision Boundaries Of AdaBoost Ensembles}}$",
            margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)

        fig.show()
        fig.write_image("ex4/q2.png", scale=2)

        # Question 3: Decision surface of best performing ensemble
        test_error = []
        num_classifiers_list = np.arange(1, 251)
        for T in num_classifiers_list:
            test_error.append(adaboost.partial_loss(test_X, test_y, T))
        T_hat = np.argmin(test_error)
        accu = accuracy(test_y, adaboost.partial_predict(test_X, T_hat))
        fig1 = make_subplots(1, 1)

        prediction = lambda X0: adaboost.partial_predict(X0, T=T_hat)
        fig1.add_traces([decision_surface(prediction, lims[0], lims[1], showscale=False),

                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                    showlegend=False,
                                    marker=dict(color=test_y,
                                                symbol=class_symbols[
                                                    np.int64((test_y + 1) / 2)],
                                                colorscale=[custom[0], custom[-1]],
                                                line=dict(color="black", width=1)))
                         ], rows=1, cols=1)

        fig1.update_layout(
            title=rf"$\textbf{{Decision Boundaries of Best AdaBoost Ensemble}} \\ \
                       \textbf{{Ensemble Size = }}\textbf{{{T_hat}}}\textbf{{, Accuracy =}} \textbf{{{accu}}}$",
            margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)

        fig1.show()
        fig1.write_image("ex4/q3.png", scale=2)

    # Question 4: Decision surface with weighted samples
    D = adaboost.D_
    D = D / np.max(D) * 5

    fig2 = make_subplots(1, 1)
    fig2.add_traces([decision_surface(adaboost.predict, lims[0], lims[1],
                                      showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                                showlegend=False,
                                marker=dict(color=train_y,
                                            size=D,
                                            symbol=class_symbols[
                                                np.int64((train_y + 1) / 2)],
                                            colorscale=[custom[0], custom[-1]],
                                            line=dict(color="black", width=1)))
                     ], rows=1, cols=1)

    fig2.update_layout(
        title=rf"$\textbf{{Decision Boundaries of full}} \\ \
                   \textbf{{AdaBoost Ensemble of trainings set}}$",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)

    fig2.show()
    if noise == 0:
        fig2.write_image("ex4/q4.png", scale=2)
    else:
        fig2.write_image("ex4/q5_4.png", scale=2)




if __name__ == '__main__':
    np.random.seed(0)

    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
