import math

import numpy as np

from experiments.Experiment import Experiment
from sampling.samplers import LDA, ULA
from utils.stats import Kolmogorov_Smirnov_statistics


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