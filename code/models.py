
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from itertools import izip
import random
from collections import Iterable
from samplers import ULA, LDA, MALA
from tools import draw_index

# 1D Gaussian mixture models
class Gaussians(object):
    
    def __init__(self, means, weights, batch_size = 10, data_size = 5000):
        assert isinstance(means, list) and isinstance(weights, list)
        if sum(weights) < 1:
            weights = [1 - sum(weights)] + weights
        assert sum(weights) == 1
        super(Gaussians, self).__init__()
        self.ms = np.array(means)
        self.ws = np.array(weights)
        self.indices = np.array(range(len(self.ms)))
        self.batch_size = batch_size
        self.data_size = data_size
        # draw with ws[i] probability from normal pdf i.
        self.data = [np.random.normal(self.ms[draw_index(self.ws)], 1)
                            for i in range(self.data_size)]
        
        self.name = "Gaussian Mixtures"
        self.configuration = "# function = %s \n" \
                             "# means = %s \n# weights = %s \n" % \
                             (self.name, ", ".join(map(str, self.ms)), ", ".join(map(str, self.ws)))
    
    def __str__(self):
        return self.configuration
    
    def __call__(self, x):
        return self.f(x)
        
    def p(self, x, mu):
        return stats.norm.pdf(x, mu, 1) 
    
    def pdf(self, x, ms=None, ws=None):
        ms = self.ms if ms is None else ms
        ws = self.ws if ws is None else ws
        return sum(wj * self.p(x, mj) for wj, mj in izip(ws, ms))
        
    def cdf(self, x, ms = None, ws = None):
        if isinstance(x, list):
            return [self.cdf(xi, ms, ws) for xi in x]
        ms = self.ms if ms is None else ms
        ws = self.ws if ws is None else ws
        return sum(w * stats.norm.cdf(x, m, 1) for w, m in izip(ws, ms))
    
    def f(self, x): # U(x) = - log p(x)
        if isinstance(x, list):
            return map(self.f, x)
        return -np.log(self.pdf(x, self.ms, self.ws))
                
    def df(self, x): # -grad log p
        if isinstance(x, list):
            return map(self.df, x)
        grad = np.sum(self.ws*(x - self.ms)*np.asarray([self.p(x,mj) for mj in self.ms]).flatten()) / self.pdf(x, self.ms, self.ws)
        return grad
        
    # random subsampled function: - grad log tilde p 
    # theta: tuple or list of tuples
    def rf(self, theta, batch = None):
        if isinstance(theta, list):
            return map(self.rdf, theta)
        if batch is None:
            batch = self.data
        return np.mean([-np.log(self.pdf(x, theta, self.ws)) for x in batch])
    
    # random subsampled gradient: - grad log tilde p 
    # theta: tuple or list of tuples
    def rdf(self, theta, batch = None):
        if isinstance(theta, list):
            return map(self.rdf, theta)
        if batch is None:
            batch = random.sample(self.data, self.batch_size)
        
        # grad ms
        px = [self.pdf(xi, theta, self.ws) for xi in batch]
        xi = batch[0]
        grad = -sum(self.ws*(xi-theta)*np.asarray([self.p(xi,mj) for mj in theta]).flatten()/pxi for xi, pxi in izip(batch, px)) / float(len(batch))
        
        return grad
    
    

if __name__ == "__main__":



    ms = [-4., 4.]
    ws = [0.7, 0.3]
    f = Gaussians(ms, ws)
    
    ula = ULA(f, f.df)
    
    x0 = lambda: 0
    dt, max_iter = 0.1, 180
    times = ula.get_times(None, dt, max_steps=max_iter)
    N = 210
    thetas = [ula.simulate(x0, times)[-1][0] for i in range(N)]
    print thetas
    x = np.arange(-10, 10, 0.001)
    y = np.exp(-f(x))
    y /= np.trapz(y, x)
    plt.plot(x, y)
    plt.hist(thetas, bins = 30, normed = True)
    plt.show()
    
    exit(0)




    ms = [-4, 4, 19]
    ws = [0.3, 0.2]
    f = Gaussians(ms, ws, batch_size = 40)
    theta = np.array(ms)
    ula = LD(f.rf, f.rdf)
    dt, max_iter = 0.05, 1000
    times = ula.get_times(None, dt, max_steps = max_iter)
    thetas = ula.simulate(lambda: theta, times)
    #print thetas
    th0 = [theta[0] for theta in thetas]
    th1 = [theta[1] for theta in thetas]
    th2 = [theta[2] for theta in thetas]
    print np.mean(th0)
    print np.mean(th1)
    print np.mean(th2)
    
    plt.hist(th0)
    plt.hist(th1)
    plt.hist(th2)
    plt.show()