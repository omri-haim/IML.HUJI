import matplotlib.pyplot as plt
import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
from matplotlib.patches import Ellipse
from IMLearn.metrics import accuracy
from matplotlib import cm
import matplotlib
from matplotlib.legend_handler import HandlerBase


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly
    separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    abs_path = "Z:/My Drive/Courses/IML/IML.HUJI/datasets/"
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(abs_path+f)

        # Fit Perceptron and record loss in each fit iteration

        losses = []
        callback_func = lambda fit, a, b: losses.append(fit.loss(a,b))
        model = Perceptron(callback=callback_func)
        model.fit(X, y)


        # Plot figure of loss as function of fitting iteration
        iters = np.arange(1, len(losses)+1)
        plt.plot(iters, losses)

        plt.xlabel("Iteration #")
        plt.ylabel("Misclassification Loss")
        plt.title(n + " Data\nMisclassification Loss vs Iteration #")
        plt.xlim((1, len(iters)))
        plt.show()


def draw_ellipse(position, covariance, ax=None, color=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    covariance = covariance[0:2,0:2]
    position = position[0:2]

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(2, 3):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs, color=color))


class MarkerHandler(HandlerBase):
    def create_artists(self, legend, tup,xdescent, ydescent,
                        width, height, fontsize,trans):
        return [plt.Line2D([width/2], [height/2.],ls="",
                       marker=tup[1],color=tup[0], transform=trans)]


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    abs_path = "Z:/My Drive/Courses/IML/IML.HUJI/datasets/"
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(abs_path+f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on
        # the left and LDA predictions on the right. Plot title should specify dataset
        # used and subplot titles should specify algorithm and accuracy
        # Create subplots

        markers = ['o', 's', '^']
        colors = np.array(["teal", "red", "purple"])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7))
        yy1 = lda.predict(X)
        yy2 = gnb.predict(X)

        for idx,k in enumerate(lda.classes_):
            xx = X[y==k]
            yyy1 = yy1[y == k]
            yyy2 = yy2[y == k]

            s1 = ax1.scatter(xx[:,0], xx[:,1], c=colors[yyy1], cmap=colors,
                             marker=markers[idx], alpha=0.5, label=k)
            ax1.scatter(lda.mu_[idx, 0], lda.mu_[idx, 1], c='k', marker='x')

            s2 = ax2.scatter(xx[:,0], xx[:,1], c=colors[yyy2], cmap=colors,
                             marker=markers[idx], alpha=0.5, label=k)
            ax2.scatter(gnb.mu_[idx, 0], gnb.mu_[idx, 1], c='k', marker='x')

        legend11 = ax1.legend(loc="lower left", title="True", title_fontsize=16)
        legend21 = ax2.legend(loc="lower left", title="True", title_fontsize=16)

        LH1 = legend11.legendHandles
        LH2 = legend21.legendHandles
        for i in range(len(LH1)):
            LH1[i].set_color('k')
            LH2[i].set_color('k')

        ax1.add_artist(legend11)
        ax2.add_artist(legend21)

        ax1.legend(list(zip(colors, ['s']*3)), lda.classes_,
                  handler_map={tuple: MarkerHandler()}, loc="lower right",
                   title="Predicted", title_fontsize=16)
        ax2.legend(list(zip(colors, ['s'] * 3)), lda.classes_,
                   handler_map={tuple: MarkerHandler()}, loc="lower right",
                   title="Predicted", title_fontsize=16)

        ax1.set_xlabel("feature 1", fontsize=16)
        ax2.set_xlabel("feature 1", fontsize=16)
        ax1.set_ylabel("feature 2", fontsize=16)
        ax2.set_ylabel("feature 2", fontsize=16)

        lda_accur = accuracy(y, lda.predict(X))
        gnb_accur = accuracy(y, gnb.predict(X))
        lda_txt = "LDA\nAccuracy = " + str(round(lda_accur,3))
        gnb_txt = "Gaussian Naive Bayes\nAccuracy = " + str(round(gnb_accur,3))
        ax1.set_title(lda_txt, fontsize=18)
        ax2.set_title(gnb_txt, fontsize=18)

        for k in range(len(lda.classes_)):
            mu = lda.mu_[k,:]
            draw_ellipse(mu, lda.cov_, ax=ax1, color='k', alpha=0.2, zorder=0)

        for k in range(len(gnb.classes_)):
            cov = np.diag(gnb.vars_[k,:])
            mu = gnb.mu_[k,:]
            draw_ellipse(mu, cov, ax=ax2, color='k', alpha=0.2, zorder=0)

        plt.suptitle(f, fontsize=20)
        plt.show()
        del lda
        del gnb


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    # compare_gaussian_classifiers()

