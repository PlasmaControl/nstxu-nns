import os
NN_ROOT = os.environ['NN_ROOT']
import sys
sys.path.append(NN_ROOT)
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
from pertnet.data.data_utils import load_data
from pertnet.net.utils import (plot_response_coeffs, make_shot_response_movie, plot_loss_curve,
                               train, mse_loss, weighted_mse_loss, LMLS, MLP, DataPreProcess,
                               visualize_response_prediction, plot_response_timetraces)

print('Loading parameters...')

# load parameters
fn = os.getcwd() + '/args.json'
with open(fn) as infile:
    hp = EasyDict(json.load(infile))
hp.root = NN_ROOT
if 'use_pretrained_model' not in hp:
    hp.use_pretrained_model = False

# load data
print('Loading data...')
traindata = load_data(hp.root + hp.traindata_fn)
valdata = load_data(hp.root + hp.valdata_fn)

# process dataset (normalize, randomize, etc)
print('Normalizing data...')
preprocess = DataPreProcess(traindata, hp.xnames, hp.ynames, t_thresh=None)
trainX, trainY,_,_ = preprocess.transform(traindata, randomize=True, holdback_fraction=0.1)
valX, valY,_,_ = preprocess.transform(valdata, randomize=True, holdback_fraction=0.0)

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
    print('Saving loss...')
    np.savetxt('loss.txt', (training_loss, validation_loss))

    # save model
    if hp.savemodel:
        print('Saving model...')
        pth = './net.pth'
        torch.save(net.state_dict(), pth)

    plot_loss_curve(training_loss, validation_loss, hp)


# plot & visualize results
print('Making figures...')
shotlist = [203172, 203942, 204069, 204155]

if hp.shape_control_mode:
    plot_response_timetraces(shotlist, net, valdata, preprocess,hp)
else:
    plot_response_coeffs(valdata, preprocess, net, hp, ncoeffs=3)
    for shot in shotlist:
        visualize_response_prediction(valdata, preprocess, net, loss_fcn, hp, shot, nsamples=10)


if hp.plotmovie or hp.savemovie:
    _ = make_shot_response_movie(valdata, preprocess, net, loss_fcn, hp, ishot=0)

print('Done.')











