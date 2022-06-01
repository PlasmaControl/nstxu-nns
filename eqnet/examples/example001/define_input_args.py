
''''
This script defines input args used by the neural net, such as which variables are the 
inputs and outputs, and various hyperparameters. These parameters are then written to the
args.json file which is used by eqnet_batch.py. 

Changing the variable 'nn_mode' will change the inputs and outputs as described in the reference paper,
to perform either equilibrium reconstruction or forward Grad-Shavranov solving. 

If performing multiple batch job submission, submit_jobs.py in the net folder performs the similar task. 
'''

import os
import shutil
from easydict import EasyDict
import json
import numpy as np

# file paths
ROOT = os.environ['NN_ROOT']
jobdir = ROOT + 'eqnet/examples/example001'


''' HYPERPARAMETERS FOR THE NN'''
hp = EasyDict()
hp.num_epochs = 300       #  num of epochs to train for
hp.batch_size = 50        #  batch size during training
hp.learn_rate =  1e-5     #  learning rate used during training (Adam optimizer)
hp.p_dropout_in = 0       #  dropout rate input layer (range 0-1)
hp.p_dropout_hidden = 0   #  dropout rate hidden layers (range 0-1)
hp.nonlinearity = 'elu'   #  nonlinearity between layers (implemented: 'relu', 'tanh', 'elu', 'leakyrelu')
hp.lossfun = 'L1'         #  loss function for training (implemented: 'L1', 'L2')
hp.num_h =  7             #  number of hidden layers
hp.h_dim = 800            #  dimension of each hidden layer
hp.hidden_dims = [hp.h_dim for i in range(hp.num_h)]  # dimensions of each hidden layer in list format. This is what actually gets used by NN


''' GENERAL JOB SETTINGS '''
settings = EasyDict()
settings.root = ROOT
settings.print_every = 1000
settings.savefigs = True
settings.savemodel = False
settings.use_pretrained_model = False
settings.dataset_dir = '/eqnet/data/datasets/'
settings.data_pca_fn = '/data_pca_017.dat'
settings.rawdata_dir = '/data/rawdata/data_by_shot'
settings.obj_dir = '/data/matlab/run_gspert/obj/'
settings.load_results_dir = jobdir + '/results_cached/'
settings.save_results_dir = jobdir + '/results/'


# which shots and times to plot equilibria for
# shots2plot should be selected from the testdataset:
#   [204655, 204656, 204658, 204659, 204660, 204661, 204944,
#    204960, 204961, 204962, 204963, 204964, 204965, 204966,
#    204967, 204968, 204969, 204971, 204972, 205005, 205062]
settings.shots2plot = [204655, 204963, 204944]  
settings.times2plot = [.040, .060, .100, 0.150, .200, .300, .500, 1.00]


''' 
SELECT MODE OF OPERATION: RECONSTRUCTION, FORWARD, ETC.
DATA INPUTS AND OUTPUTS DEPEND ON THE MODE
'''

nn_mode = 'reconstruction'   # choose forward, forward-control, reconstruction or reconstruction-control



'''
USER WOULD GENERALLY NOT MODIFY ANYTHING BEYOND THIS POINT
'''
if nn_mode == 'forward':

    settings.shape_control_mode = False  

    # predictor variables
    settings.xnames = ['coil_currents', 'vessel_currents', 'ip', 'pprime', 'ffprim']

    # target variables
    # psizr_pla = psizr minus the applied flux from the coil AND vessel currents
    settings.ynames = ['psizr_pla']   

elif nn_mode == 'forward-control':
    
    settings.shape_control_mode = True
    settings.xnames = ['coil_currents', 'vessel_currents', 'ip', 'pprime', 'ffprim']
    settings.ynames = ['shape_' + x for x in ['rx_lo', 'zx_lo', 'rx_up', 'zx_up', 'rcur', 'zcur', 
        'gap1', 'gap2', 'a', 'b', 'kappa', 'delta', 'R0', 'rmax', 'rx_lo_filtered', 'zx_lo_filtered', 
        'rx_up_filtered', 'zx_up_filtered', 'islimited']]

elif nn_mode=='reconstruction':
    
    settings.shape_control_mode = False  
    settings.xnames = ['coil_currents_meas', 'ip', 'bpsignals', 'ivsignals', 'flsignals', 'vloop']
    settings.ynames = ['psizr_pla_iv']   # psizr_pla = psizr minus the applied flux from coil currents, not including vessel currents

elif nn_mode == 'reconstruction-control':
    
    settings.shape_control_mode = True
    settings.xnames = ['coil_currents_meas', 'ip', 'bpsignals', 'ivsignals', 'flsignals', 'vloop']
    settings.ynames = ['shape_' + x for x in ['rx_lo', 'zx_lo', 'rx_up', 'zx_up', 'rcur', 'zcur', 
        'gap1', 'gap2', 'a', 'b', 'kappa', 'delta', 'R0', 'rmax', 'rx_lo_filtered', 'zx_lo_filtered', 
        'rx_up_filtered', 'zx_up_filtered', 'islimited']]


settings.pretrained_model_fn  = jobdir + '/results_cached/net.pth'




# write to file
args = {**hp, **settings}
args_fn = jobdir + '/args.json'
with open(args_fn, 'w') as outfile:
        json.dump(args, outfile, indent=4)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



















