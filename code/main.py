
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from experiments.grid_search import grid_search_LD
from sampling.samplers import ULA, MALA
from experiments.Experiment import Experiment
from utils.stats import Kolmogorov_Smirnov_statistics

from sampling.GaussianMixtureModels import GaussianMixtureModels

from utils.Parser import Parser


def spread_mu():
    
    
    Z = lambda : np.random.normal(0,1) 
    x0 = lambda : 0#Z
    T = 500
    N = 12
    m = 3
    steps = 50
    oms = [0.7, 0.3]
    experiment = Experiment(N=N, max_steps=steps, dt = 0.5)
    
    mus = [1, 2, 4, 6, 8]
    for counter, mu in enumerate(mus):
        
        f = GaussianMixtureModels([-mu, mu], oms)
        sampler = MALA(f, f.df)
        sampler.dt = experiment.dt

        
        ks_statistics = []
        for i in range(m):
            experiment.run(sampler)
#            experiment.write_data(sampler)

            data = [p for p in experiment.get_path(sampler, -1)[0] if not math.isnan(p)]
            #print data
            ks = Kolmogorov_Smirnov_statistics(data, f.cdf)
            ks_statistics.append(ks)
        
        print mu, "-->", np.median(ks)
            
    
def run_experiment(experiment, samplers, f):
    
    for sampler in samplers:
        experiment.run(sampler)
        experiment.write_data(sampler)
        print sampler.name
    print "---"  


def plot_experiment(experiment, samplers, f):
    
    x = np.arange(-15, 15, 0.01)
    y = np.exp(np.exp(-samplers[0].U(x)))
    y -= 1
    plt.plot(x, y, label='true posterior')
     
    for sampler in samplers:
        data = [p for p in experiment.get_path(sampler, -1)[0] if not math.isnan(p)]
        plt.hist(data, bins = 100, label = sampler.name, normed = True, alpha = 0.5)
        ks = Kolmogorov_Smirnov_statistics(data, f.cdf)
        print sampler.name, "KS =", ks
            
    plt.xlim(-12, 12)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig("visualization/histogram_comparison.png")
    plt.savefig("visualization/histogram_comparison.eps")
    #plt.show()


if __name__ == "__main__":


    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = 1
    if parser:
        # load config file
        p = Parser(os.path.join(dir_path, "config.txt"))
        p.parse()
        #print map(lambda x: x[1].name, p.experiments)
        p.run()        
                
        #plot_all()
    else:
        #spread_mu()
        #exit(0)
        mu = [-4, 2, 9]
        om = [0.3, 0.2, 0.5]
        
        #mu = [-4, 4]
        #om = [0.3, 0.7]
        ms = [-10, -3, 5]
        ws = [0.5, 0.2, 0.3]
        
        f = GaussianMixtureModels(ms, ws)
        grid_search_LD(f)
    # grid_search_LD(f)
    # grid_search_ULA(f)
        exit(0)
        
        steps = 1500
        dt = 0.6
        N = 200
        
        experiments = [ \
                    # (Experiment(N=N, dt=dt, max_steps=steps), LD(f, f.rdf, a = 0.5, b = 1, g = 0.71)), \
                    (Experiment(N=N, dt=dt, max_steps=steps), ULA(f, f.rdf)), \
                    # (Experiment(N=N, dt=dt, max_steps=steps), MALA(f, f.df)), \
                    ]

        for experiment, sampler in experiments:
            experiment.run(sampler)
            experiment.write_data(sampler)
            print sampler.name
            