import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import stats
import math


from samplers import ULA, MALA, LDA, HMC
from Experiment import Experiment
from tools import Kolmogorov_Smirnov_statistics

from discretizer import Euler, Milstein
from models import Gaussians

from configuration import Parser
from plots import plot_all



def grid_search_LD(f):
    
    Z = lambda : np.random.normal(0,1) 
    x0 = lambda : 0#Z
    T = 50
    dt = 0.05
    N = 120
    m = 1
    steps = 500
    experiment = Experiment(N=N, dt=dt, max_steps=steps)
    
    a = [5.0, 2.0, 1.0, 0.5, 0.1]
    b = [5.0, 1.0, 0.5]
    g = [0.6, 0.7, 0.8, 1.0]
    
    samplers = []
    for ai in a:
        for bi in b:
            for gi in g:
                samplers.append(LDA(f, f.df, a = ai, b = bi, g = gi))
                samplers[-1].name = "LDA|%s|%s|%s" % (ai, bi, gi)
    
    
    with open("optimization.tex", "w+") as optfile:
        table_begin = "\n\\begin{tabular}{|cccc}\n a & b &$\gamma$& KS \\\\ \n \hline \n"
        table_end = "\\end{tabular}\n"
        
        optfile.write("%s LD\n%s steps: %s\n" % ("%", "%", steps))
        optfile.write("\small")
        optfile.write(table_begin)
        
        bestc, bestks = "", 1.0
        
        for counter, sampler in enumerate(samplers):
            if counter % 12 == 0:
                optfile.write(table_begin)
                optfile.write(table_end)
        
            print sampler.name
            config = "%s & %s & %s" % (sampler.a, sampler.b, sampler.g)
            
            ks_statistics = []
            for i in range(m):
                experiment.run(sampler)
    #            experiment.write_data(sampler)
    
                data = [p for p in experiment.get_path(sampler, -1)[0] if not math.isnan(p)]
                ks = Kolmogorov_Smirnov_statistics(data, f.cdf)
                ks_statistics.append(ks)
            
            print np.median(ks)
            optfile.write("%s & %0.3f \\\\" % (config, np.median(ks)))
            if ks < bestks:
                bestc = config
                bestks = ks
        optfile.write(table_end)
        
    print "Optimum:\n", bestc, ";  %0.3f" % bestks
    
    
    
def grid_search_ULA(f):
    
    Z = lambda : np.random.normal(0,1) 
    x0 = lambda : 0#Z
    T = 50
    N = 120
    m = 30
    steps = 500
    
    dts = np.arange(1.0, 5.1, 0.5)
    
    experiment = Experiment(N=N, max_steps=steps)
    sampler = ULA(f, f.df)
    
    with open("optimization.tex", "wa") as optfile:
        table_begin = "\n\\begin{tabular}{|cc}\n dt & KS \\\\ \n \hline \n"
        table_end = "\n\\end{tabular}\n"
        
        optfile.write("LD\nsteps: %s\n" % steps)
        optfile.write("\small")
        optfile.write(table_begin)
        
        bestc, bestks = "", 1.0
        
        for counter, dt in enumerate(dts):
            
            if counter > 0 and counter % 14 == 0:
                optfile.write(table_end)
                optfile.write(table_begin)
        
            experiment.dt = dt
            config = " %s " % (dt)
            
            ks_statistics = []
            for i in range(m):
                experiment.run(sampler)
    #            experiment.write_data(sampler)
    
                data = [p for p in experiment.get_path(sampler, -1)[0] if not math.isnan(p)]
                ks = Kolmogorov_Smirnov_statistics(data, f.cdf)
                ks_statistics.append(ks)
            
            print dt, "-->", np.median(ks)
            optfile.write("%s & %0.3f \\\\" % (config, np.median(ks)))
            if ks < bestks:
                bestc = config
                bestks = ks
        optfile.write(table_end)
        
    print "Optimum:\n", bestc, ";  %0.3f" % bestks
    
    
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
        
        f = Gaussians([-mu, mu], oms)
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

    parser = 1
    if parser:
        p = Parser("visualization/Config_plotting.txt")
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
        
        f = Gaussians(ms, ws)
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
            