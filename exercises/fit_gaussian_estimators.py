from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
import matplotlib.pyplot as plt


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    m = 1000
    s = 1
    X = np.random.normal(m, s, 1000)
    estim = UnivariateGaussian()
    estim = estim.fit(X)
    print('(', estim.mu_, ', ', estim.var_, ')', sep="")

    # Question 2 - Empirically showing sample mean is consistent
    samp_size = np.array(list(range(10, 1010, 10)))
    dist = np.zeros((samp_size.size))
    for samp in range(10, 1010, 10):
        estim = estim.fit(X[0:samp])
        n = int(samp/10-1)
        dist[n] = np.abs(estim.mu_ - m)

    plt.plot(samp_size, dist)
    plt.xlabel('Sample Size')
    plt.ylabel('Absolute Distance')
    plt.title('Absolute distance between the estimated\n and true value of the '
              'expectation')
    plt.show()



    # Question 3 - Plotting Empirical PDF of fitted model
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
