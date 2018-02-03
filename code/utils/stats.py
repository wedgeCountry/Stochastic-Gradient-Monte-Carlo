import numpy as np


def moving_average(a, n=3) :
    # from http://stackoverflow.com/questions/14313510/ddg#14314054
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def running_mean(x, m):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[m:] - cumsum[:-m]) / m


def compare(hist, density):
    (bins, counts) = hist
    midpoints = bins[:-1] + np.diff(bins)*0.5
    frequency = counts / np.sum(counts)
    return np.sum(np.abs(frequency[i] - density(m)) for i, m in enumerate(midpoints))


def Kolmogorov_Smirnov_statistics(X, cdf):

    n = len(X)
    M = -1e5
    for i, Fxi in enumerate(cdf(sorted(X))):
        M = max(M, max(1.0 * (i+1) / n - Fxi, Fxi - 1.0 * i / n ))
    return M


def empirical_KS(X, Y):
    '''
    Calculates the empirical Kolmogorov Smirnov statistics of two realizations of random variables X and Y.
    Basically, it calculates the maximum difference of the empirical cdfs of X and Y.
    '''

    # histogram height increments
    dX = 1./len(X)
    dY = 1./len(Y)

    # sort lists
    X = sorted(X)
    Y = sorted(Y)

    S = 0.0 # sum of X - Y
    M = 0.0 # maximum of abs(S)

    # loop
    while len(X) > 0 and len(Y) > 0:
        if X[0] < Y[0]:
            S += dX
            X.pop(0)
            M = max(M, np.abs(S))
        elif Y[0] < X[0]:
            S -= dY
            Y.pop(0)
            M = max(M, np.abs(S))
        else:
            S = S + dX - dY
            X.pop(0)
            Y.pop(0)
            M = max(M, np.abs(S))
    # Now X or Y may still contain elements
    # But the the other ecdf = 1 and the difference can get only smaller
    return M