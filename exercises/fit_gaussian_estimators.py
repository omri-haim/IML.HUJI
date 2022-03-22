from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
import matplotlib.pyplot as plt


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    m = 10
    s = 1
    X = np.random.normal(loc=m, scale=s, size=1000)
    estim = UnivariateGaussian()
    estim = estim.fit(X)
    print('(', estim.mu_, ', ', estim.var_, ')', sep="")

    # Question 2 - Empirically showing sample mean is consistent
    samp_size = np.array(list(range(10, 1010, 10)))
    dist = np.zeros((samp_size.size))
    for samp in range(10, 1010, 10):
        estim0 = UnivariateGaussian()
        estim0 = estim0.fit(X[0:samp])
        n = int(samp/10-1)
        dist[n] = np.abs(estim0.mu_ - m)

    plt.plot(samp_size, dist)
    plt.xlabel('Sample Size')
    plt.ylabel('Absolute Distance')
    plt.title('Absolute distance between the estimated\n and true value of the '
              'expectation')
    plt.show()


    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = estim.pdf(X)
    plt.scatter(X, pdfs)
    plt.xlabel('Samples')
    plt.ylabel('Empirical PDF of sample')
    plt.title('Empirical PDF of fitted model')
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    M = np.transpose(np.array([0, 0, 4, 0]))
    COV = np.transpose(np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5,
                                                                                  0, 0, 1]]))
    X = np.random.multivariate_normal(M, COV, 1000)
    estim = MultivariateGaussian()
    estim = estim.fit(X)
    print('Estimated Expectation:', estim.mu_)
    print('Estimated Covariance:\n', estim.cov_)

    # Question 5 - Likelihood evaluation
    F1 = np.linspace(-10, 10, 200)
    F3 = np.linspace(-10, 10, 200)
    N = len(F1)
    LL = np.zeros((N, N))
    c = 0
    for f1 in np.nditer(F1):
        r = 0
        for f3 in np.nditer(F3):
            M1 = np.array([f1, 0, f3, 0])
            LL[r, c] = MultivariateGaussian.log_likelihood(M1, COV, X)
            r += 1
        c += 1

    plt.imshow(np.flip(LL), interpolation='nearest', extent=[-10, 10, -10, 10])
    plt.colorbar()
    plt.xlabel('f1')
    plt.ylabel('f3')
    plt.title('Log-Likelihood Heatmap as a function of f1 and f3')
    plt.show()

    # Question 6 - Maximum likelihood

    f3_max, f1_max = np.unravel_index(np.argmax(LL, axis=None), LL.shape)
    f1_max = np.round(F1[f1_max], 3)
    f3_max = np.round(F3[f3_max], 3)
    # print("f1_max = ", f1_max)
    # print("f3_max = ", f3_max)

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
    a = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])

    m1 = 1
    s1 = 1

    m2 = 10
    s2 = 1

    ll1 = UnivariateGaussian.log_likelihood(m1, s1, a)

    ll2 = UnivariateGaussian.log_likelihood(m2, s2, a)

    print(round(ll1, 2))
    print(round(ll2, 2))
