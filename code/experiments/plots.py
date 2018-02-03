import os
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt

import utils.stats
from sampling.GaussianMixtureModels import GaussianMixtureModels
from utils.stats import Kolmogorov_Smirnov_statistics, empirical_KS
from utils.file import loadtxt, unzip


def get_config(filename):
    #get configuration
    mu = []
    om = []
    function = None
    with open(filename, "r") as onefile:
        for line in onefile:
            if line.startswith("#"):
                if "function = " in line:
                    function = line.split("=")[1].rstrip().lstrip()
                if "means =" in line:
                    mu = map(float, line.split("=")[1].strip().split(","))
                elif "weights =" in line:
                    om = map(float, line.split("=")[1].strip().split(","))
            else:
                break
    
    if function == "Gaussian Mixtures":
        function = GaussianMixtureModels(mu, om)
    
    return function, mu, om

    
def get_files_and_times(directory):
    if not os.path.exists(directory):
        print "directory '%s' does not exist." % directory
        return None, None
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    ending = ".%s" % files[0].split(".")[-1]
    times = [float(f.rstrip(ending)) for f in files]
    files, times = unzip(sorted(izip(files, times), key = lambda x: x[1]))
    return files, times


##
## Plotting modes
##
def plot_hist(ax, name, color, count = -1, bins = 60, pdfcolor="red", directory = None):
    files, times = get_files_and_times(directory)
    f, mu, om = get_config(os.path.join(directory, files[0]))

    data = loadtxt(os.path.join(directory, files[-1]))
    data = [d for d in data if not np.isnan(d)]
    #ds = loadtxt(os.path.join(directory, files[-1]), flat = False)
    #for d in ds:
    #    print np.mean(d)
    #print data
    #print data
    #print data
    ax.hist(data, bins = bins, label = name, normed = True, alpha = 1.0, color = color)
    #print min(data)
    #print data
    x = np.arange(min(data) - 2, max(data) + 2, 0.01)
    ax.plot(x, f.pdf(x), '--', label='true posterior', linewidth=5, color = pdfcolor)
    
def plot_means(ax, name, color, directory = None):
    files, times = get_files_and_times(directory)
    f, mu, om = get_config(os.path.join(directory, files[0]))
    
    datas = [loadtxt(os.path.join(directory, filename)) for filename in files]    
    means = map(np.mean, datas)
    ax.plot(times, means, label = name, color = color)

def plot_time_KS(ax, name, color, directory = None):
    files, times = get_files_and_times(directory)
    f, mu, om = get_config(os.path.join(directory, files[0]))
    
    datas = [loadtxt(os.path.join(directory, filename)) for filename in files]    
    KS = [Kolmogorov_Smirnov_statistics(data, f.cdf) for data in datas]
    ax.plot(times, KS, label = name, color = color)
    #ax.plot([times[0], times[-1]], [np.min(KS), np.min(KS)], color = color)
    
def plot_count_KS(ax, name, color, directory = None, log = 0):
    files, times = get_files_and_times(directory)
    f, mu, om = get_config(os.path.join(directory, files[0]))
    
    datas = [loadtxt(os.path.join(directory, filename)) for filename in files]    
    KS = [Kolmogorov_Smirnov_statistics(data, f.cdf) for data in datas]
    KS = utils.stats.moving_average(KS, 10)
    #KS2 = tools.running_mean(KS, 10)
    #print KS[:100] - KS2[:100]
    if log:
        KS = np.log(KS)
    ax.plot(range(len(KS)), KS, label = name, color = color, linewidth = 2)
    #ax.plot([0, len(times)], [np.min(KS), np.min(KS)], color = color)

def plot_empirical_KS(ax, name, color, directory = None, log = 0):
    files, times = get_files_and_times(directory)
    f, mu, om = get_config(os.path.join(directory, files[0]))
    
    m = 10
    data = [loadtxt(os.path.join(directory, filename)) for filename in files]    
    eKS = [ np.mean( [empirical_KS(data[i], data[i-j]) for j in range(1,m+1)] ) \
                for i in range(m, len(data))]
    eKS = utils.stats.moving_average(eKS, 10)
    if log:
        eKS = np.log(eKS)
    ax.plot(range(len(eKS)), eKS, label = name, color = color)
    
def plot_fancy(ax, name, color, directory=None):
    
    files, times = get_files_and_times(directory)
    data = [loadtxt(os.path.join(directory, filename)) for filename in files]  
    f, mu, om = get_config(os.path.join(directory, files[0]))
    density = f.pdf
    print mu
    
    paths = []
    for i in range(len(data[0])):
        path = [d[i] for d in data]
        if len(paths) == 0 and path[-1] < 0:
            #print i
            paths.append(path)
        if path[-1] > 0:
            #print i
            paths.append(path)
            break
    
    yrange = max(max(abs(min(p)), abs(max(p))) for p in paths)
    length = len(paths[0])
    x = np.arange(-yrange, yrange, 0.01)
    y = density(x) * length
    
    for path in paths:
        plt.plot(path)
    plt.plot([length, length], [-yrange, yrange], color = 'black')
    plt.plot(y + length, x, color = 'black')
    plt.ylim([-yrange, yrange])


##
## Plotting Configuration
##

def get_plot_config():
    vis = "./visualization"
    modes = ["hist", "count-KS", "fancy"]#, "empirical-KS"]
    colors = ["#111d4a","#0094c6","#02ce6f","#5fad41","#1e1e24"]
    ending = ".txt"
    return vis, modes, colors, ending

##
## Main Plotting Function
##

def plot_data(vis, data_name, modes, colors, ending):
    data_dir = "%s/data" % (vis)
    #print data_dir
    names = sorted([f for f in os.listdir(data_dir) if not os.path.isfile(os.path.join(data_dir, f)) and f.split("_")[0] == data_name and not f.startswith("_")])
    
    #print " ".join(names)
    #print " ".join(map(str, range(len(names))))
    #order = raw_input("Please enter order for %s case: \n" % data_name)
    #order = map(int, order.split(" ")[:len(names)])
    #ordered_names = []
    #for i in order:
        #ordered_names.append(names[i])
    #names = ordered_names
    
    #print names
    
    for mode in modes:
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.xaxis.label.set_fontsize(24)
        ax.yaxis.label.set_fontsize(24)
    
        #print "Mode:", mode
        if mode == "hist":
            method = plot_hist
            plt.ylabel("Probability")
            plt.xlabel("x - Value")
           # plt.title("Sampling histogram")            
        elif mode == "fancy":
            method = plot_fancy
            plt.ylabel("x - Value")
            plt.xlabel("iteration")
           # plt.title("Sampling histogram")            
        elif mode == "time-KS":
            method = plot_time_KS
            plt.xlabel("Time")
            plt.ylabel("KS")
        elif mode == "count-KS":
            method = plot_count_KS
            plt.xlabel("Iteration")
            plt.ylabel("KS")
        elif mode == "empirical-KS":
            method = plot_empirical_KS
            #plt.title("Empirical KS statistics with preceding iterations")
            plt.xlabel("Iteration")
            plt.ylabel("eKS")
        else:
            print "mode %s unknown" % mode
            continue
        
        mode = "%s_%s" % (data_name, mode)
        #print "MODE", mode
        mode = mode.lstrip("data_")
        
        for name, color in zip(names, colors):
            directory = "%s/data/%s" % (vis, name)
            name = name.split("/")[-1].split("_")[-1]
            #print "DIR", directory, "NAME", name
            method(ax, name, color, directory = directory)
            
        handles, labels = ax.get_legend_handles_labels()
        use_labels = list(set(labels))
        for i in reversed(range(1, len(labels))):
            if labels[i] in labels[:i]:
                del handles[i]
                del labels[i]
        
        ax.legend(handles, labels)
       # plt.ylim([0, 0.5])
        plt.savefig("%s/%s.png" % (vis, mode))
        plt.savefig("%s/eps/%s.eps" % (vis, mode))
     

def plot_all():
    
    vis, modes, colors, ending = get_plot_config()    
    if not os.path.exists("%s/eps" % vis):
        os.makedirs("%s/eps" % vis)
    data_names = [f for f in os.listdir("%s/data" % vis) if not os.path.isfile(os.path.join(vis, f))]
    data_names = set('_'.join(d.split("_")[:-1]) for d in data_names)
    #print data_names
    
    for data_name in data_names:
        if data_name.startswith("_"):
            continue
        plot_data(vis, data_name, modes, colors, ending)


##
## Other specific plots
##

def plot_Brownian_motion():
    
    vis, modes, colors, ending = get_plot_config()    
    if not os.path.exists("%s/eps" % vis):
        os.makedirs("%s/eps" % vis)
    
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',      # ticks along the bottom edge are off
        right='off',      # ticks along the bottom edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    
    N = 1000
    np.random.seed(42)
    
    B = [0]
    for i in range(N):
        B.append(B[-1] + np.random.normal(0,1))
    plt.plot(range(N+1), B, color=colors[0], label="Bt(w1)")
    np.random.seed(41)
    B = [0]
    for i in range(N):
        B.append(B[-1] + np.random.normal(0,1))
    plt.plot(range(N+1), B, color=colors[1], label="Bt(w2)")
    np.random.seed(40)
    B = [0]
    for i in range(N):
        B.append(B[-1] + np.random.normal(0,1))
    plt.plot(range(N+1), B, color=colors[2], label="Bt(w3)")
    
    plt.grid(True)
    plt.legend()
    plt.savefig("%s/%s.png" % (vis, "BrownianMotion"))
    plt.savefig("%s/eps/%s.eps" % (vis, "BrownianMotion"))
    #plt.show()

from sampling.samplers import MALA
def plot_acceptance_rates():
    
    vis, modes, colors, ending = get_plot_config()    
    if not os.path.exists("%s/eps" % vis):
        os.makedirs("%s/eps" % vis)
    
    names = ["MALAGaussians3_MALA15", "MALAGaussians3_MALA005"]
    
    for index, name in enumerate(names):
        directory = "./visualization/data/%s" % name
        files, times = get_files_and_times(directory)
        f, mu, om = get_config(os.path.join(directory, files[0]))

        sampler = MALA(f, f.df)
        acceptance_rates = []
        prev_data = None
        prev_time = None
        for counter, (time, df) in enumerate(zip(times, files)):
            if prev_data is None:
                prev_data = loadtxt(os.path.join(directory, df))
                prev_time = time
                continue
            data = loadtxt(os.path.join(directory, df))
            rates = []
            print time
            if counter > 20:
                break
            for d1, d2 in zip(prev_data, data):
                rates.append(sampler.acceptance_rate(d1, d2, time-prev_time))
            acceptance_rates.append(np.median(rates))
        print acceptance_rates
        plt.plot(range(len(acceptance_rates)), acceptance_rates, label = name, color = colors[index])
        plt.show()
    #plt.grid(True)
    plt.legend()
    plt.savefig("%s/%s.png" % (vis, "Acceptance_Rates_%s" % name))
    plt.savefig("%s/eps/%s.eps" % (vis, "Acceptance_Rates_%s" % name))
    plt.show()



if __name__ == "__main__":
    
    #plot_acceptance_rates()
    
    #plot_all()
    #exit(0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vis, modes, colors, ending = get_plot_config()    
    data_names = ["MALAGaussians3", "Gaussians3", "Gaussians", "GaussiansRandom", "GaussiansMono1"]   
    for data_name in data_names:
        plot_data(vis, data_name, ["hist", "count-KS"], colors, ending)
    
    
    
    
    #plt.show()
    