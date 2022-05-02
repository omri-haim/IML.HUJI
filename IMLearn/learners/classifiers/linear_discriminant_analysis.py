from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        for k_idx, k in enumerate(self.classes_):
            pi_k = X[y == k].shape[0] / len(y)
            mu_k = X[y == k].mean(axis=0)

            if k_idx == 0:
                self.pi_ = np.array([pi_k])
                self.mu_ = mu_k
                temp = X[y == k] - mu_k
                cov = (X[y == k] - mu_k).T @ (X[y == k] - mu_k)
            else:
                self.pi_ = np.append(self.pi_, pi_k)
                self.mu_ = np.vstack((self.mu_, mu_k))
                temp = X[y == k] - mu_k
                cov += (X[y == k] - mu_k).T @ (X[y == k] - mu_k)


        self.cov_ = cov/len(y)
        self._cov_inv = np.linalg.inv(self.cov_)


    def _predict(self, X: np.ndarray) -> np.ndarray:
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
        likelihoods = self.likelihood(X)
        responses = np.argmax(likelihoods, axis=1)
        return responses


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        # for k_idx, k in enumerate(self.classes_):
        a = self._cov_inv @ (self.mu_).T
        b = np.log(self.pi_) - 0.5*np.diag(self.mu_@ self._cov_inv @ (self.mu_).T)
        likelihoods = X @ a + b
        return likelihoods


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
        from ...metrics import misclassification_error
        y_pred = self._predict(X)
        loss = misclassification_error(y, y_pred)
        return loss
