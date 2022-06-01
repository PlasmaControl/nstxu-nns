import os
import shutil
from easydict import EasyDict
import json
from sklearn.model_selection import ParameterGrid
import numpy as np

# file paths
ROOT = os.environ['NN_ROOT']
sbatch_fn = ROOT + 'eqnet/net/job.slurm'
job_fn = ROOT + 'eqnet/net/eqnet_batch.py'
job_topdir = ROOT + 'eqnet/jobs/reconstruction/jobs019a/'


# hyperparameter grid
hyperparams = EasyDict()
hyperparams.num_epochs = [300]

hyperparams.batch_size = [50]
#hyperparams.batch_size = [50, 100, 200, 300, 400, 500]

hyperparams.num_h =  [7]
# hyperparams.num_h =  [3,4,5,6,7,8,9,10]  

hyperparams.h_dim = [800] 
# hyperparams.h_dim = [200, 400, 600, 800, 1000]

hyperparams.learn_rate =  [1e-5] 
#hyperparams.learn_rate =  [1e-6, 2e-6, 5e-6, 1e-5, 5e-5, 1e-4]

hyperparams.p_dropout_in = [0, 0.005, 0.01, 0.02, 0.05]

hyperparams.p_dropout_hidden = [0]
#hyperparams.p_dropout_hidden = [0, 0.02, 0.05, 0.1, 0.15, 0.2]

hyperparams.nonlinearity = ['elu'] 
# hyperparams.nonlinearity = ['tanh'] 
# hyperparams.nonlinearity = ['relu', 'tanh', 'elu', 'leakyrelu'] # case-insensitive

hyperparams.ijob = [0]

hyperparams.lossfun = ['L1']

hpgrid = list(ParameterGrid(hyperparams))




# General job settings
settings = EasyDict()
settings.print_every = 1000
settings.savefigs = True
settings.savemovie = False  
settings.plotmovie = False
settings.savemodel = True
settings.use_pretrained_model = False
settings.root = ROOT
settings.dataset_dir = 'eqnet/data/datasets/'
settings.data_pca_fn = 'data_pca_019.dat'
settings.rawdata_dir = 'data/rawdata/data_by_shot/'
settings.obj_dir = 'data/matlab/run_gspert/obj/'

# predictor variables (uncomment one of these lines)
# settings.xnames = ['coil_currents', 'vessel_currents', 'ip', 'li', 'wmhd']
# settings.xnames = ['coil_currents', 'vessel_currents', 'ip', 'pprime', 'ffprim']
# settings.xnames = ['coil_currents', 'ip', 'bpsignals', 'ivsignals', 'flsignals', 'wmhd']
settings.xnames = ['coil_currents_meas', 'ip', 'bpsignals', 'ivsignals', 'flsignals', 'vloop']

# target variables (uncomment one of these)
# settings.ynames = ['psirz'] 
# settings.ynames = ['psizr_pla']
settings.ynames = ['psizr_pla_iv2']
settings.shape_control_mode = False  # predict flux (not shape parameters)

# settings.ynames = ['shape_' + x for x in ['rx_lo', 'zx_lo', 'rx_up', 'zx_up', 'rcur', 'zcur', 'gap1', 'gap2', 'a', 'b', 'kappa', 'delta', 'R0', 'rmax', 'rx_lo_filtered', 'zx_lo_filtered', 'rx_up_filtered', 'zx_up_filtered', 'islimited']]
#settings.ynames += ['psizr_pla']
# settings.shape_control_mode = True # direct prediction of shape parameters

if os.path.isdir(job_topdir):
    shutil.rmtree(job_topdir)
os.mkdir(job_topdir)


grid_fn = job_topdir + 'hpgrid.txt'
with open(grid_fn, 'w') as outfile:
    json.dump(hyperparams, outfile, indent=4)

# Launch jobs
for ijob, hp in enumerate(hpgrid):


    hp = EasyDict(hp)
    
    # this file is a record for the settings of all the jobs
    with open(grid_fn, 'a') as writer:
        writer.write('\n\njob' + str(ijob) + '\n')

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

    # settings for the individual job
    with open(args_fn, 'w') as outfile:
        json.dump(args, outfile, indent=4)

    # launch job
    os.chdir(jobdir)
    os.system('sbatch ' + sbatch_fn)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
