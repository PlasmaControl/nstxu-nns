import os
ROOT = os.environ['NN_ROOT']
import sys
sys.path.append(ROOT)
import numpy as np
from easydict import EasyDict
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import shutil
import json
from torch.utils.data import TensorDataset, DataLoader
from eqnet.data.data_utils import load_data
import scipy.io as sio
from eqnet.net.eqnet_utils import (plot_response_coeffs, make_shot_response_movie, 
                                   plot_loss_curve, train, mse_loss, Weighted_MSE_Loss, 
                                   LMLS, MLP, DataPreProcess, visualize_response_prediction, 
                                   get_pred_psizr, get_true_psizr, plot_psizr, plot_psizr_for_shot, 
                                   plot_psizr_pla_for_shot, plot_shape_timetraces, 
                                   gen_output_preds, train_val_test_split)


print('Loading parameters...')

# load parameters
fn = os.getcwd() + '/args.json'
with open(fn) as infile:
    hp = EasyDict(json.load(infile))
hp.root = ROOT
if 'use_pretrained_model' not in hp:
    hp.use_pretrained_model = False

# load data
print('Loading data...')
data_pca = load_data(ROOT + hp.dataset_dir + hp.data_pca_fn)

traindata, valdata, testdata = train_val_test_split(data_pca, ftrain=0.8, fval=0.1, mix=True)


# process data
print('Normalizing data...')
preprocess = DataPreProcess(traindata, hp.xnames, hp.ynames, t_thresh=None)
trainX, trainY,_,_ = preprocess.transform(traindata, randomize=True, holdback_fraction=0, by_shot=True)
valX, valY,_,_ = preprocess.transform(valdata, randomize=True, holdback_fraction=0)

# dataloaders
train_dataset = TensorDataset(trainX, trainY)
train_dataloader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
val_dataset = TensorDataset(valX, valY)
val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

# initialize NN
in_dim = trainX.shape[1]
out_dim = trainY.shape[1]
net = MLP(in_dim, out_dim, hp.hidden_dims, nonlinearity=hp.nonlinearity,
          p_dropout_in=hp.p_dropout_in, p_dropout_hidden=hp.p_dropout_hidden)

# loss function and optimizer
if hp.lossfun=='L1':
    loss_fcn = torch.nn.L1Loss()
else: 
    loss_fcn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=hp.learn_rate)


if hp.use_pretrained_model:
    pth = './net.pth'
    net.load_state_dict(torch.load(pth))
    net.eval()
else:
    # train
    print('Training...')
    net, training_loss, validation_loss = train(
        net, loss_fcn, optimizer, train_dataloader, val_dataloader, hp)
    print('Training complete.')

    net.eval()

    # write loss to file
    np.savetxt('loss.txt', (training_loss, validation_loss))

    # save model
    if hp.savemodel:
        print('Saving model...')
        pth = hp.savedir + 'net.pth'
        torch.save(net.state_dict(), pth)

    # plot loss curve
    print('Making figures...')
    plot_loss_curve(training_loss, validation_loss, hp)


# save predictions
out = {}
out['test'] = gen_output_preds(testdata, preprocess, net, hp)
out['val']  = gen_output_preds(valdata, preprocess, net, hp)
sio.savemat('./out.mat', {'out':out})
    

# plot various predictions
print('Making figures...')
shotlist = np.unique(testdata['shot'])
shotlist = shotlist[0:-1:3]
# shotlist = [204650]

if hp.shape_control_mode:

    # plot time traces of shape control parameters    
    plot_shape_timetraces(shotlist, net, testdata, preprocess,hp)

else:

    # plot timetraces of pca coefficients
    plot_response_coeffs(testdata, preprocess, net, hp, ncoeffs=2)

    # plot predictions of flux on grid
    plot_times = [.040, .060, .100, 0.150, .200, .300, .500, 1.00]
    tok_data_struct = sio.loadmat(ROOT + hp.obj_dir + 'nstxu_obj_config2016_6565.mat')['tok_data_struct']
    mpc = sio.loadmat(ROOT + hp.obj_dir + 'mpcx.mat')['mpcx']
    mpv = sio.loadmat(ROOT + hp.obj_dir + 'mpvx.mat')['mpvx']

    if 'psirz' in hp.ynames:
        plot_fun = plot_psizr_for_shot
    elif ('psizr_pla' in hp.ynames) or ('psizr_pla_iv' in hp.ynames):
        plot_fun = plot_psizr_pla_for_shot    

    for shot in shotlist:
        plot_fun(shot, plot_times, net, testdata, preprocess, tok_data_struct, hp, mpc, mpv)



print('Done.')

