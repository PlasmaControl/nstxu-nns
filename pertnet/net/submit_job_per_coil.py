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
job_topdir = ROOT + 'pertnet/jobs/standard/jobs013_allcoils_f/'


ynames = ['dpsidix_smooth_coil' + str(icoil) for icoil in range(1,55)]
# ynames = ['dpsidix_smooth_coil1']
ynames += ['dpsidbetap', 'dpsidli']


for ii, yname in enumerate(ynames):
    
    jobdir = job_topdir + 'job' + str(ii) + '/'

    # input args for the job
    hyperparams = EasyDict()
    hyperparams.num_epochs = 100
    hyperparams.batch_size = 50
    hyperparams.num_h = 5
    hyperparams.h_dim = 500
    hyperparams.learn_rate = 3e-5
    hyperparams.p_dropout_in = 0
    hyperparams.p_dropout_hidden = 0.05
    hyperparams.nonlinearity = 'relu'  # 'relu', 'tanh', 'elu', or 'leakyrelu', case-insensitive
    hyperparams.lossfun = 'L1'        # 'L1' or 'L2'
    
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
    settings.shape_control_mode = False

    # 2. predictor variables
    settings.xnames =  ['pprime', 'ffprim', 'pres','rmaxis','zmaxis', 'psirz', 'coil_currents', 'vessel_currents', 'pcurrt', 'rcur', 'zcur', 'ip', 'qpsi', 'psimag', 'psibry', 'rbbbs', 'zbbbs', 'psizr_pla']

    # 3. target variables
    settings.ynames = [yname]

    # =======================
    settings.savedir = jobdir
    hyperparams.hidden_dims = [hyperparams.h_dim for i in range(hyperparams.num_h)]
    args = {**hyperparams, **settings}
 
    # create job directory and copy files
    args_fn = jobdir + 'args.json'

    if os.path.isdir(jobdir):
        shutil.rmtree(jobdir)
    os.makedirs(jobdir)

    shutil.copy(job_fn, jobdir)

    with open(args_fn, 'w') as outfile:
        json.dump(args, outfile, indent=4)


    # submit job
    os.chdir(jobdir)
    os.system('sbatch ' + sbatch_fn)

