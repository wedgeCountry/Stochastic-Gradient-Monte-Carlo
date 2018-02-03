import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import izip
import os

import multiprocessing as mp

from samplers import ULA, MALA

class Experiment(object):
    
    def __init__(self, x0 = lambda: 0, T = 1, dt = 0.1, N = 30, max_steps = None):
        self.x0 = x0
        self.T = T
        self.dt = dt
        self.N = N
        self.max_steps = int(max_steps)
        
        self.paths = {}
        self.times = {}
        self.histograms = []


    def run(self, sampler, info=None):
        times = sampler.get_times(self.T, self.dt, max_steps = self.max_steps)
        self.times[sampler.name] = times
        
        path = sampler.simulate(self.x0, times)
        self.paths[sampler.name] = [path]
        for i in range(self.N-1):
            if info:
                print i
            path = sampler.simulate(self.x0, times)
            self.paths[sampler.name].append(path)
            #self.paths[sampler.name] = np.vstack([self.paths[sampler.name], path])
         
    def get_directory(self, sampler, destination = "./visualization/data"):
        return "%s/%s" % (destination, sampler.name)
    
    
    def get_path(self, sampler, index = None):
        return self.paths[sampler.name]
    
    
    def prepare_data_directory(self, directory):
        # create if not exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        # clean directory before writing
        for f in os.listdir(directory):
            filepath = os.path.join(directory, f)
            if os.path.isfile(filepath):
                os.remove(filepath)
        
    def write_data(self, sampler, destination = "./visualization/data"):
        
        directory = self.get_directory(sampler, destination)
        self.prepare_data_directory(directory)
        #print directory
        paths = self.paths[sampler.name]
        times = self.times[sampler.name]
        #for path in paths:
        #    print "p ", path
        numeric = False
        try:
            dim = len(paths[0][0])
        except:
            dim = 1
            numeric = True
        for i, time in enumerate(times):
            filename = "%s/%s.txt" %(directory, time)
            #print filename
            with open(filename, "w+") as datafile:
                datafile.write("%s" % sampler.U)
                for j in range(dim):            
                    for path in paths:
                        if numeric:
                            datafile.write("%0.5f " % path[i]) 
                        else:
                            datafile.write("%0.5f " % path[i][j]) 
                    datafile.write("\n")
        return directory
        
        
    
        
        
        
        
        
        
        
        
               
    #def get_histograms(self, index = None, bins = None):
        
        #if index is None:
            #index = range(self.paths.shape[1])
        #elif not isinstance(index, list):
            #index = [int(index)]
        
        #if bins is None:
            #bins = 30
        #else:
            #bins = len(bins)
            
        #remaining = [i for i in index if i not in self.histograms]
        #for i in remaining:
            #self.histograms[i] = np.hist(self.paths[:,i], bins, normed=True)
        
        #return self.histograms
        
        