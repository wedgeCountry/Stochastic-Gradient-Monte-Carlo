import os
import re

from experiments.Experiment import Experiment
from sampling.samplers import *
from sampling.GaussianMixtureModels import GaussianMixtureModels
from experiments.plots import get_plot_config

class Parser(object):
    
    def __init__(self, filename):
        super(Parser, self).__init__()
        self.filename = filename
        self.experiments = []
        self.case_prefix = "data_"
        assert os.path.isfile(filename), filename
     
    def get_sampler_name(self, name, dt):
        return "%s%s" %(name, \
            re.sub("\.", '', ("%0.4f" % dt).strip().strip(".").rstrip("0")))
        
    def parse(self):
        self.experiments = []
        
        case = ""
        ms, ws = [], []
        steps = 500
        N = 30
        dt = 0.5
        force = False
        
        with open(self.filename, "r") as configfile:
            for line in configfile:
                if line.startswith("##") or not line.strip():
                    continue
                elif line.startswith("#"):
                    case = line.lstrip("#").strip()
                    continue
                else:
                    line = re.sub("#.*", '', line)
                
                if "force =" in line:
                    force = line.split("=")[1].strip() in ["True", "true", "1"]
                elif "means" in line and "=" in line:
                    ms = map(float, line.split("=")[1].strip().split(","))
                elif "weights" in line and "=" in line:
                    ws = map(float, line.split("=")[1].strip().split(","))
                elif "N =" in line:
                    N = int(line.split("=")[1].strip())
                elif "steps =" in line:
                    steps = int(line.split("=")[1].strip())
                else:
                    print case, ms
                    if "gaussians" in case.lower() and not "random" in case.lower():
                        F = GaussianMixtureModels(ms, ws)
                        dF = F.df
                        x0 = lambda : np.array(0)
                    elif "gaussians" in case.lower() and "random" in case.lower():
                        F = GaussianMixtureModels(ms, ws, batch_size = 50)
                        dF = F.rdf
                        x0 = lambda : np.array([0] * len(ms))
                    else:
                        continue
                    name = line.split(" ")[0].strip()
                    #print dt
                    if name == "ULA":
                        dt = float(line.split(" ")[1])
                        ex = Experiment(x0 = x0, N=N, dt=dt, max_steps=steps)
                        sampler = ULA(F, dF)
                        sampler.name = self.get_sampler_name(sampler.name, dt)
                    elif name == "MALA":
                        dt = float(line.split(" ")[1])
                        ex = Experiment(x0 = x0, N=N, dt=dt, max_steps=steps)
                        sampler = MALA(F, dF)
                        sampler.name = self.get_sampler_name(sampler.name, dt)
                    elif name == "LDA" or name == "LDA2":
                        line = ''.join(line.split(" ")[1:])
                        a, b, g = map(float, line.strip().split(","))
                        ex = Experiment(x0 = x0, N=N, max_steps=steps)
                        sampler = LDA(F, dF, a=a, b=b, g=g)
                        sampler.name = name
                    elif name == "DLDA":
                        line = ''.join(line.split(" ")[1:])
                        a, b, g = map(float, line.strip().split(","))
                        ex = Experiment(x0 = x0, N=N, max_steps=steps)
                        sampler = DLDA(F, dF, a=a, b=b, g=g)
                        sampler.name = name
                    elif name == "GD":
                        ex = Experiment(x0 = x0, N=N, max_steps=steps)
                        sampler = GD(F, dF)
                    elif name == "SGD":
                        ex = Experiment(x0 = x0, N=N, max_steps=steps)
                        draw_batch = lambda : F.draw_batch(ms, ws)
                        sampler = SGD(F, dF, ms, ws, draw_batch = draw_batch)
                    else:
                        print "WARNING: Unknown sampler %s!\n\n" % name
                        continue
                    
                    sampler.name = "%s_%s" % (case, sampler.name)
                    self.experiments.append((ex, sampler, force))
                   
                   
    def run(self):
        vis, modes, colors, ending = get_plot_config()
        data_names = []
        for experiment, sampler, force in self.experiments:
            
            directory = experiment.get_directory(sampler)
            print "______", directory
            if os.path.exists(directory) and not force:
                continue
            print "_______", sampler.U.ms
            experiment.run(sampler)
            experiment.write_data(sampler)
            data_names.append(directory.split("/")[-1])
        
        data_names = set(d.split("_")[0] for d in data_names)
        for data_name in data_names:
            #plot_data(vis, data_name, modes, colors, ending)
            print data_name


if __name__ == "__main__":
    p = Parser("config.txt")
    p.parse()
    print map(lambda x: x[1].name, p.experiments)
    p.run()        

    #plot_all()