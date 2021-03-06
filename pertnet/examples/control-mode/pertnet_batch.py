import os
ROOT = os.environ['NN_ROOT']
import sys
sys.path.append(ROOT)
import numpy as np
from easydict import EasyDict
import torch
import torch.nn as nn
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import shutil
import json
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as sio
from pertnet.data.data_utils import load_data
from pertnet.net.pertnet_utils import (plot_response_coeffs, gen_output_preds, 
                            plot_loss_curve, train, MLP, DataPreProcess, visualize_response_prediction, 
                            plot_response_timetraces, train_val_test_split)

print('Loading parameters...')

# load parameters
fn = os.getcwd() + '/args.json'
with open(fn) as infile:
    hp = EasyDict(json.load(infile))

# load data
print('Loading data...')
data_pca = load_data(ROOT + hp.dataset_dir + hp.data_pca_fn)
traindata, valdata, testdata = train_val_test_split(data_pca, ftrain=0.8, fval=0.1, mix=True)


# process data (normalize, randomize, etc)
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

if hp.lossfun=='L1':
    loss_fcn = torch.nn.L1Loss()
else: 
    loss_fcn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=hp.learn_rate)

if hp.use_pretrained_model:
    pth = hp.load_results_dir + '/net.pth'
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
    np.savetxt(hp.save_results_dir + 'loss.txt', (training_loss, validation_loss))

    # save model
    if hp.savemodel:
        print('Saving model...')
        pth = hp.save_results_dir + '/net.pth'
        torch.save(net.state_dict(), pth)

    plot_loss_curve(training_loss, validation_loss, hp)

# save predictions
out = {}
out['test'] = gen_output_preds(testdata, preprocess, net, hp)
out['val']  = gen_output_preds(valdata, preprocess, net, hp)
sio.savemat(hp.save_results_dir + '/out.mat', {'out':out})

# plot & visualize results
print('Making figures...')

if hp.shape_control_mode:
    plot_response_timetraces(hp.shots2plot, net, data_pca, preprocess,hp)
else:
    plot_response_coeffs(hp.shots2plot, data_pca, preprocess, net, hp, ncoeffs=3)
    tok_data = sio.loadmat(ROOT + hp.obj_dir + 'tok_data.mat')['tok_data']
    for shot in hp.shots2plot:
        visualize_response_prediction(data_pca, preprocess, net, loss_fcn, hp, shot, tok_data, nsamples=10)

plt.show()
print('Done.')











