
''''
This script defines input args used by the neural net, such as which variables are the 
inputs and outputs, and various hyperparameters. These parameters are then written to the
args.json file which is used by pertnet_batch.py. 

If performing multiple batch job submission, submit_jobs.py in the net folder performs the similar task. 

These settings configure Pertnet for "flux-mode", predict the response of the flux wrt coil currents, li, betap.
'''

import os
import shutil
from easydict import EasyDict
import json
import numpy as np

# file paths
ROOT = os.environ['NN_ROOT']
jobdir = ROOT + 'pertnet/examples/flux-mode'


''' HYPERPARAMETERS FOR THE NN'''  
hp = EasyDict()
hp.num_epochs = 300          #  num of epochs to train for
hp.batch_size = 50           #  batch size during training
hp.learn_rate =  3e-5        #  learning rate used during training (Adam optimizer)
hp.p_dropout_in = 0          #  dropout rate input layer (range 0-1)
hp.p_dropout_hidden = 0.05   #  dropout rate hidden layers (range 0-1)
hp.nonlinearity = 'relu'     #  nonlinearity between layers (implemented: 'relu', 'tanh', 'elu', 'leakyrelu')
hp.lossfun = 'L1'            #  loss function for training (implemented: 'L1', 'L2')
hp.num_h =  4                #  number of hidden layers
hp.h_dim = 500               #  dimension of each hidden layer
hp.hidden_dims = [hp.h_dim for i in range(hp.num_h)]  # dimensions of each hidden layer in list format. This is what actually gets used by NN


''' GENERAL JOB SETTINGS '''
settings = EasyDict()
settings.root = ROOT
settings.print_every = 1000
settings.savefigs = True
settings.savemodel = False
settings.use_pretrained_model = False
settings.dataset_dir = '/pertnet/data/datasets/'
settings.data_pca_fn = '/data_pca_013.dat'
settings.load_results_dir = jobdir + '/results_cached/'
settings.save_results_dir = jobdir + '/results/'
settings.obj_dir = '/data/matlab/run_gspert/obj/'

# which shots and times to plot equilibria for
settings.shots2plot = [204655, 204963, 204944]  
settings.times2plot = [.040, .060, .100, 0.150, .200, .300, .500, 1.00]


''' 
SELECT DATA INPUTS AND OUTPUTS
'''

settings.shape_control_mode = False

# predictor variables
settings.xnames =  ['pprime', 'ffprim', 'pres','rmaxis','zmaxis', 'psirz', 
    'coil_currents', 'vessel_currents', 'pcurrt', 'rcur', 'zcur', 'ip', 
    'qpsi', 'psimag', 'psibry', 'rbbbs', 'zbbbs', 'shape_rx_lo_filtered', 
    'shape_zx_lo_filtered', 'shape_rx_up_filtered', 'shape_zx_up_filtered', 
    'shape_islimited', 'psizr_pla']

# target variables

# Choose yname from: 'dpsidbetap', 'dpsidli', 'dsidix_smooth_coil###' where ### within 1-54
#    when ### within 1-13: these correspond to the PF coils [OH, PF1AU, PF1BU, PF1CU, PF2U, 
#                          PF3U, PF4, PF5, PF3L, PF2L, PF1CL, PF1BL, PF1AL]
#    when ### > 14: this corresponds to vacuum vessel elements

settings.ynames = ['dpsidix_smooth_coil1']  # response to OH coil


# write to file
args = {**hp, **settings}
args_fn = jobdir + '/args.json'
with open(args_fn, 'w') as outfile:
        json.dump(args, outfile, indent=4)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






























    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
