
import numpy as np
import matplotlib.pyplot as plt
import collections

class DiscretizationScheme(object):

    def __init__(self, b, s):
        super(DiscretizationScheme, self).__init__()
        self.b = b      # drift term
        self.s = s      # diffusion term

class Euler(DiscretizationScheme):

    def __init__(self, b, s):
        super(Euler, self).__init__(b, s)

    def step(self, x, dt):
        if dt <= 0.0:
            return x
        try:
            dim = len(x)
        except:
            dim = 1
        xi = np.random.normal(0,1, dim)
        #print x, self.b(x), self.s(x) * np.sqrt(dt) * xi
        return x + dt * self.b(x) + np.sqrt(dt) * self.s(x) * xi

class Milstein(DiscretizationScheme):

    def __init__(self, b, s, ds):
        super(Milstein, self).__init__(b, s)
        self.ds = ds

    def step(self, x, dt):
        if dt <= 0.0:
            return x
        xi = np.random.normal(0,1)
        return np.asarray(x + self.b(x) * dt + self.s(x) * np.sqrt(dt) * xi \
                            + 0.5 * dt * self.s(x) * self.ds(x) * (xi**2 - 1))


# N - dimensional Euler
class EulerND(DiscretizationScheme):

    def __init__(self, b, s):
        super(EulerND, self).__init__(b, s)

    def step(self, x, dt):
        xi = np.random.normal(0,1,len(x)).reshape([len(x), 1])
        return x + self.b(x) * dt + np.sqrt(dt) * self.s(x) * xi


# TODO: Works in only one dimension
def Euler_step(x, b, s, dt = 0.1):
    if dt <= 0.0:
        return x
    try:
        dim = len(paths[0][0])
    except:
        dim = 1
    xi = np.random.normal(0,1, dim)
    # TODO: What happesn if s is a matrix?
    return np.asarray(x + b(x) * dt + np.sqrt(dt) * s(x) * xi)



if __name__ == "__main__":

    b = lambda x : x * np.sin(x)
    s = lambda x : np.sqrt(np.abs(x))
    ds = lambda x: 0.5*x/(np.abs(x)**(1.5))
    Z = lambda   : np.random.normal(4,1)
    
    X = Euler(b, s)
    Y = Milstein(b,s,ds)
    
    x0 = Z()
    T = 1
    dt = 0.05
    N = 1000

