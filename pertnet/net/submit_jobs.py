import os
import shutil
from easydict import EasyDict
import json
from sklearn.model_selection import ParameterGrid
import numpy as np


# file paths
ROOT = os.environ['NN_ROOT'] 
sbatch_fn = ROOT + 'pertnet/net/job.slurm'
job_fn = ROOT + 'pertnet/net/pertnet_batch.py'
job_topdir = ROOT + 'pertnet/jobs/control/jobs013_nets_a/'


# hyperparameter grid settings
hyperparams = EasyDict()
hyperparams.num_epochs = [300]

hyperparams.batch_size = [50]
hyperparams.num_h = [4]
hyperparams.h_dim = [500]
hyperparams.learn_rate = [3e-5]
hyperparams.p_dropout_in = [0]
hyperparams.p_dropout_hidden = [0.05]
hyperparams.nonlinearity = ['relu']  # 'relu', 'tanh', 'elu', or 'leakyrelu', case-insensitive
hyperparams.lossfun = ['L1']         # 'L1' or 'L2'

hyperparams.ijob = [0,1,2,3,4]

# hyperparams.batch_size = [20,50,100,200,400,800]
# hyperparams.num_h = [1,2,3,4,5,6,7,8,9]
# hyperparams.h_dim = [100,200,300,400,500,600,700,800]
# hyperparams.learn_rate = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
# hyperparams.p_dropout_in = [0]
# hyperparams.p_dropout_hidden = [0, 0.02, .05, 0.1, 0.15]
# hyperparams.nonlinearity = ['relu', 'tanh', 'elu', 'leakyrelu']  
# hyperparams.lossfun = ['L1']         

hpgrid = list(ParameterGrid(hyperparams))


# General job settings
settings = EasyDict()
settings.print_every = 1000
settings.savefigs = True
settings.savemovie = False
settings.plotmovie = False
settings.savemodel = True
settings.traindata_fn = 'pertnet/data/datasets/train_013.dat'
settings.valdata_fn = 'pertnet/data/datasets/val_013.dat'
settings.testdata_fn = 'pertnet/data/datasets/test_013.dat'


#=============================================
# Standard (flux response prediction) settings
# ============================================
# 1. shape control mode 
# settings.shape_control_mode = False

# 2. predictor variables
# settings.xnames =  ['pprime', 'ffprim', 'pres','rmaxis','zmaxis', 'psirz', 'coil_currents', 'vessel_currents', 'pcurrt', 'rcur', 'zcur', 'ip', 'qpsi', 'psimag', 'psibry', 'rbbbs', 'zbbbs', 'psizr_pla']

# 3. target variables
# settings.ynames = ['dpsidix_smooth_coil1']
# settings.ynames = ['dpsidix_smooth_coil2']
# settings.ynames = ['dpsidix_smooth_coil28']
# settings.ynames = ['dpsidbetap']
# settings.ynames = ['dpsidli']


#============================
# Shape control mode settings
# ===========================

# 1. shape control mode 
settings.shape_control_mode = True

# 2. predictor variables
settings.xnames =  ['pprime', 'ffprim', 'pres','rmaxis','zmaxis', 'psirz', 'coil_currents', 'vessel_currents', 'pcurrt', 'rcur', 'zcur', 'ip', 'qpsi', 'psimag', 'psibry', 'rbbbs', 'zbbbs', 'shape_rx_lo_filtered', 'shape_zx_lo_filtered', 'shape_rx_up_filtered', 'shape_zx_up_filtered', 'shape_islimited', 'psizr_pla']


# 3. target variables
settings.ynames = ['gamma', 'shape_drcurdix', 'shape_dzcurdix', 'shape_drxlodix', 'shape_drxupdix', 'shape_dzxlodix', 'shape_dzxupdix']

# settings.ynames = ['drdbetap', 'drdli', 'drdip', 'dzdbetap', 'dzdli', 'dzdip', 
#      'dpsibrydbetap', 'dpsibrydli', 'dpsibrydip', 'drxdbetap', 'drxdli', 
#      'drxdip', 'dzxdbetap', 'dzxdli', 'dzxdip']

# settings.ynames = ['drdis', 'dzdis', 'drxdis', 'dzxdis', 'dpsibrydis']

#settings.ynames = ['drdis', 'drdbetap', 'drdli', 'drdip', 'dzdis', 'dzdbetap', 'dzdli', 
#      'dzdip', 'ddrsepdis', 'ddrsepdbetap', 'ddrsepdli', 'ddrsepdip', 'dpsibrydis', 
#      'dpsibrydbetap', 'dpsibrydli', 'dpsibrydip', 'drxdis', 'drxdbetap', 'drxdli', 
#      'drxdip', 'dzxdis', 'dzxdbetap', 'dzxdli', 'dzxdip']


if os.path.isdir(job_topdir):
    shutil.rmtree(job_topdir)
os.mkdir(job_topdir)

grid_fn = job_topdir + 'hpgrid.txt'
# open(grid_fn, 'w').close()
with open(grid_fn, 'w') as outfile:
    json.dump(hyperparams, outfile, indent=4)
    json.dump(settings, outfile, indent=4)

# Launch jobs
for ijob, hp in enumerate(hpgrid):

    hp = EasyDict(hp)
    
    # one file that contains settings for all jobs
    with open(grid_fn, 'a') as writer:
        writer.write('\n\n' + str(ijob) + '\n')

    with open(grid_fn, 'a') as outfile:
        json.dump(hp, outfile, indent=4)
    
    # create job directory
    jobdir = job_topdir + 'job' + str(ijob) + '/'
    
    if os.path.isdir(jobdir):
        shutil.rmtree(jobdir)
    os.makedirs(jobdir)
                
    # combine input args
    settings.savedir = jobdir
    hp.hidden_dims = [hp.h_dim for i in range(hp.num_h)]
    args = {**hp, **settings}

    # copy files
    args_fn = jobdir + 'args.json'
    shutil.copy(job_fn, jobdir)

    with open(args_fn, 'w') as outfile:
        json.dump(args, outfile, indent=4)

    # launch job
    os.chdir(jobdir)
    os.system('sbatch ' + sbatch_fn)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
