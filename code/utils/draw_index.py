import numpy as np
import scipy.stats


from utils.stats import Kolmogorov_Smirnov_statistics, empirical_KS


def draw_index(ws):
    """
    draw index from a given probability distribution constant with weights ws
    :param ws:
    :return:
    """
    sorted_ws = np.cumsum(sorted(ws))
    sorted_is = sorted(range(len(ws)), key=lambda i: ws[i])
    u = np.random.uniform(0,1)
    for i in range(len(ws)):
        if u <= sorted_ws[i]:
            return sorted_is[i]
    return -1


import matplotlib.pyplot as plt
if __name__ == "__main__":
    
    X = [1,3,5,6,9,10]
    Y = [0,2,5,6,7,8]
    
    X = scipy.stats.norm.rvs(0, 1, 100)
    Y = scipy.stats.norm.rvs(0, 1, 100)
    print empirical_KS(X, Y)
    print Kolmogorov_Smirnov_statistics(X, scipy.stats.norm.cdf)
    plt.plot(sorted(X), np.cumsum([1./len(X)]*len(X)))
    plt.plot(sorted(Y), np.cumsum([1./len(Y)]*len(Y)))

    #plt.show()
    #exit(0)
    
    ws = [0.2, 0.6, 0.1, 0.1]
    print draw_index(ws)
    plt.hist([draw_index(ws) for i in range(1000)], bins=len(ws), normed=True)
    plt.show()
