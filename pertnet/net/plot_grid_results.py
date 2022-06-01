from easydict import EasyDict
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from scipy.interpolate import griddata
from sklearn.model_selection import ParameterGrid

ROOT = os.environ['NN_ROOT']
job_topdir = ROOT + 'pertnet/jobs/standard/jobs013_coil28_a/'

dirlist = os.listdir(job_topdir)
loss = []

# READ FROM FILES
xnames = ['batch_size', 'h_dim', 'p_dropout_hidden', 'num_h', 'num_epochs', 'learn_rate']
var2plot = 'num_h'

hpgrid = {}
loss = []
for i in range(len(dirlist)-1):
    try: 
        args_fn = job_topdir + 'job' + str(i) + '/args.json'
        with open(args_fn) as infile:
            hp = EasyDict(json.load(infile))

        loss_fn = job_topdir + 'job' + str(i) + '/loss.txt'
        
        l = np.loadtxt(loss_fn)[1,-1]
        loss.append(l)
    
    except:
        loss.append(np.nan)

    for xn in xnames:
        if i == 0:
            hpgrid[xn] = []
        
        val = hp[xn]
        hpgrid[xn].append(val)

        
loss = np.asarray(loss)        
x = np.asarray(hpgrid[var2plot])


uniq_vals = {}
xnames.remove(var2plot)

for xn in xnames:
    uniq_vals[xn] = np.unique(hpgrid[xn])

pcombs = list(ParameterGrid(uniq_vals))

icombo = 0
for icombo in range(len(pcombs)):
    
    # find index of jobs that match this parameter combination
    for i, xn in enumerate(xnames):
        if i == 0:
            idx = (hpgrid[xn] == pcombs[icombo][xn])
        else:
            idx = idx & (hpgrid[xn] == pcombs[icombo][xn])
    
    idx = np.where(idx)[0]

    # if len(idx) > 0:              
        
    print(idx)
    plt.scatter(x[idx], loss[idx])
    plt.title(str(pcombs[icombo]))
    plt.xlabel(var2plot)
    plt.ylabel('Loss')
    plt.show()


