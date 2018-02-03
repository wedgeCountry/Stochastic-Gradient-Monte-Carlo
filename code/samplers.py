import math
import numpy as np
import matplotlib.pyplot as plt
from discretizer import *
from scipy.optimize import line_search
import scipy.stats

class Sampler(object):

    def __init__(self, U, DU, discretizer = None):
        super(Sampler, self).__init__()
        self.U = U                # objective function
        self.DU = DU              # gradient od objectve
        self.name = ""
        self.discretizer = Euler(lambda x: -self.DU(x), lambda x: np.sqrt(2)) \
                              if discretizer is None else discretizer
        
    def step(self, x, dt):
        raise NotImplementedError()
    
    # increment until T is reached or max_steps   
    def get_times(self, T, dt, max_steps = None):
        if max_steps is not None:
            L = max_steps
        else:
            L = int(T/dt)+1
        return [ i * dt for i in range(L) ]
        
    def simulate(self, x0, times):
        Xt = [x0()]
        for i, time in enumerate(times[1:]):
            dt = time - times[i]
            Xti = self.step(Xt[-1], dt)
            Xt.append(Xti)
            #print Xt[-1]
            
        return Xt
            

class ULA(Sampler):

    def __init__(self, U, DU, discretizer = None):
        super(ULA, self).__init__(U, DU, discretizer)
        self.name = "ULA"
    
    def step(self, x, dt):
        proposal = self.discretizer.step(x, dt)
        return proposal


class MALA(ULA):
    # WARNING: Assumes U(x) = - log pi(x)
    
    def __init__(self, U, DU, discretizer = None):
        super(MALA, self).__init__(U, DU, discretizer)
        self.name = "MALA"
        self.acceptance_rates = []
      
    def acceptance_rate(self, x, y, dt):
        xpdf = scipy.stats.norm.pdf(x, loc=y - dt*self.DU(y), scale=np.sqrt(2*dt))
        ypdf = scipy.stats.norm.pdf(y, loc=x - dt*self.DU(x), scale=np.sqrt(2*dt))
        a = (self.U.pdf(y)*xpdf)/(self.U.pdf(x)*ypdf)
        return a
    
    def accept(self, x, y, dt):
        u = np.random.uniform(0,1)
        a = self.acceptance_rate(x, y, dt)
        return u <= a # TODO: np.log(u) <= a
    
    def step(self, x, dt):
        proposal = self.discretizer.step(x, dt)
        if self.accept(x, proposal, dt):
            return proposal
        else:
            return x


class LDA(Sampler):

    def __init__(self, U, DU, discretizer = None, a = 1.0, b = 5.0, g = 1.0):
        super(LDA, self).__init__(U, DU, discretizer)
        self.name = "LDA"
        assert g > 0.5 and g <= 1
        self.a = a
        self.b = b
        self.g = g
        self.delay = 1
        self.counter = 1

    # get cooling schedule increment
    def cooling_scheme(self, counter, a, b, g):
        if counter % self.delay:
            self.counter += 1
        return a / (b + self.counter)**g

    # increment until T is reached or max_steps       
    def get_times(self, T, dt, max_steps = None):
        if max_steps is not None:
            times = map(lambda c: self.cooling_scheme(c, self.a, self.b, self.g), range(max_steps))
            return list(np.cumsum(times))
        else:
            times = [0.0]
            counter = 1
            while times[-1] < T:
                times.append(times[-1] + self.cooling_scheme(counter))
                counter += 1
            return times
    
    def step(self, x, dt):
        proposal = self.discretizer.step(x, dt)
        return proposal


class DLDA(LDA):

    def __init__(self, U, DU, discretizer = None, a = 1.0, b = 5.0, g = 1.0):
        super(DLDA, self).__init__(U, DU, discretizer = None, a = 1.0, b = 5.0, g = 1.0)
        self.name = "DLDA"
        self.delay = 20
        self.counter = 1


class SGD(ULA):

    def __init__(self, U, DU, ms, ws, draw_batch, discretizer = None):
        super(SGD, self).__init__(U, DU, discretizer = discretizer)
        self.ms = ms
        self.ws = ws
        self.draw_batch = draw_batch
        self.name = "SGD"
        
    def step(self, x, dt):
        batch = self.draw_batch()
        gx = -self.DU(x, batch=batch)
        line_search = [(dt, self.U.rf(x+dt*gx, batch)) for dt in [0.01, 0.1, 0.5, 1., 1.5]]
        dt = min(line_search, key = lambda x: x[1])[0]
        return Euler_step(x, lambda x: gx, lambda x: np.sqrt(2), dt=dt)
        

class GD(ULA):
    
    def __init__(self, U, DU, discretizer = None):
        super(GD, self).__init__(U, DU, discretizer = discretizer)
        self.name = "GD"
        
    def step(self, x, dt):
        gx = -self.DU(x)
        line_search = [(dt, self.U(x+dt*gx)) for dt in [0.01, 0.1, 0.5, 1., 1.5]]
        dt = min(line_search, key = lambda x: x[1])[0]
        return Euler_step(x, lambda x: gx, lambda x: np.sqrt(2), dt=dt)
        


class HMC(LDA):
    # TODO: Not tested yet!

    def __init__(self, U, DU, discretizer = None):
        super(HMC, self).__init__(U, DU, discretizer)
        self.name = "HMC"
        self.m = 1.0    # moment term
        self.c = 1.0    # diffuison scale
        self.D = np.matrix([[0,0], [0,self.c]])  # semi-definite diffusion matrix
        self.Q = np.matrix([[0,-1], [1,0]])      # skew-symmetric perturbation matrix
        self.G = np.array([0,0]).reshape([2,1])  # Gamma term = 0 for constant D, Q
        self.DH = lambda x: np.array([self.DU(x[0]), x[1]/self.m]).reshape([2,1])
        euler = EulerND(lambda x: - (self.D + self.Q) * self.DH(x) + self.G, \
                        lambda x: np.sqrt(2) * self.D)
        self.discretizer = euler

    # increment until T is reached or max_steps   
    def get_times(self, T, dt, max_steps = None):
        if max_steps is not None:
            L = max_steps
        else:
            L = int(T/dt)+1
        return [ i * dt for i in range(L) ]
    
    def simulate(self, x0, times):
        Xt = [ np.array([x0(), x0()]).reshape([2,1])]
        for i, time in enumerate(times[1:]):
            dt = time - times[i]
            Xt.append(self.step(Xt[-1], dt))
        return [float(x[0]) for x in Xt]


def test_samplers():
    
    U = lambda x : x**2
    DU = lambda x : 2*x
    Z = lambda   : np.random.normal(4,1)
    
    T = 100
    dt = 0.05
    
    samplers = [ULA(U, DU), MALA(U, DU), LDA(U, DU), HMC(U, DU)]
    
    for sampler in samplers:
        times = sampler.get_times(T, dt)
        path = sampler.simulate(Z, times)
        plt.plot(times, path, label=sampler.name)
        
    plt.legend()
    plt.ylim([-5,5])
    plt.show()

if __name__ == "__main__":

    test_samplers()
    