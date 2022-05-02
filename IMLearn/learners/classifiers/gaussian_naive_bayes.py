from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
            var_k = np.mean((X[y == k]-mu_k)**2, axis=0)

            if k_idx == 0:
                self.pi_ = np.array([pi_k])
                self.mu_ = mu_k
                self.vars_ = var_k
            else:
                self.pi_ = np.append(self.pi_, pi_k)
                self.mu_ = np.vstack((self.mu_, mu_k))
                self.vars_ = np.vstack((self.vars_, var_k))

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

        X = np.repeat(X[:,:,np.newaxis], len(self.classes_), axis=2)
        mu = self.mu_[:,:,np.newaxis]
        mu = np.swapaxes(mu, 0, 2)
        vars = self.vars_[:,:,np.newaxis]
        vars = np.swapaxes(vars, 0, 2)

        a = - 0.5*(np.log(2*np.pi*vars) + (X - mu)**2/vars)
        likelihoods = np.log(self.pi_) + np.sum(a, axis=1)
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
