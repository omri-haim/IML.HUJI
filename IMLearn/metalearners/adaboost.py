import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations

        self.models_ = [None] * iterations
        self.D_ = None
        self.weights_ = np.zeros(iterations)


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = y.shape[0]
        self.D_ = np.ones(m) / m
        X_samp = X
        y_samp = y
        idxs = np.arange(0, m)

        for t in range(self.iterations_):
            if t == 0:
                print(t, end=" ")
            elif t % 25 == 0:
                print(t)
            else:
                print(t, end=" ")

            self.models_[t] = self.wl_().fit(X_samp, y_samp)
            ht = self.models_[t].predict(X)
            epsilon = np.sum( (y != ht) * self.D_)
            self.weights_[t] = 0.5*np.log(1/epsilon - 1)
            self.D_ *= np.exp(- y * self.weights_[t] * ht)
            self.D_ /= np.sum(self.D_)
            samp_idxs = np.random.choice(idxs, size=m, replace=True, p=self.D_)
            samp_idxs.sort()
            X_samp = X[samp_idxs,:]
            y_samp = y[samp_idxs]



    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self.partial_loss(X, y, self.iterations_)


    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y_pred = np.zeros(X.shape[0])

        for t in range(T):
            y_pred += self.models_[t].predict(X) * self.weights_[t]

        return np.sign(y_pred)


    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ..metrics import misclassification_error
        y_pred = self.partial_predict(X, T)
        loss = misclassification_error(y, y_pred)
        return loss
