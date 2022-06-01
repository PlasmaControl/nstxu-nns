import numpy as np
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch
import scipy.io as sio
import copy
import mat73

# ====================
# Train-Val-Test split
# ====================

def train_val_test_split(data_pca, ftrain=0.8, fval=0.1, ftest=0.1, mix=False):
    
    shots = data_pca['shot']
    times = data_pca['time']

    uniqshots = np.unique(shots)

    ntrain = int(ftrain*len(uniqshots))
    nval = int(fval*len(uniqshots))
    ntest = len(uniqshots) - ntrain - nval

    '''
    train_val_shots = uniqshots[0:ntrain+nval]
    if mix:
        train_val_shots = np.random.permutation(train_val_shots)

    trainshots = train_val_shots[0:ntrain]
    valshots = train_val_shots[ntrain:ntrain+nval]
    testshots = uniqshots[ntrain+nval:]

    def shot_to_sample_idx(uniqshots, shots):
        for i, shot in enumerate(uniqshots):
            k = np.where(shots==shot)[0]
            if i == 0:
                idx = np.copy(k)
            else:
                idx = np.hstack((idx,k))            
        return idx

    itrain = shot_to_sample_idx(trainshots, shots)
    ival   = shot_to_sample_idx(valshots, shots)
    itest  = shot_to_sample_idx(testshots, shots)
    '''



    trainshots = uniqshots[0:ntrain]
    valshots = uniqshots[ntrain:ntrain+nval]
    testshots = uniqshots[ntrain+nval:]

    itrain = np.where(shots <= trainshots[-1])[0]
    ival = np.where( (shots >= valshots[0]) & (shots <= valshots[-1]))[0]
    itest = np.where( (shots >= testshots[0]) & (shots <= testshots[-1]))[0]
   


    traindata = copy.deepcopy(data_pca)
    valdata = copy.deepcopy(data_pca)
    testdata = copy.deepcopy(data_pca)

    keys = list(data_pca.keys())

    for key in keys:

        if key=='time' or key=='shot':
            traindata[key] = data_pca[key][itrain,:]
            valdata[key] = data_pca[key][ival,:]
            testdata[key] = data_pca[key][itest,:]
        else:
            traindata[key].coeff_ = data_pca[key].coeff_[itrain,:]
            valdata[key].coeff_   = data_pca[key].coeff_[ival,:]
            testdata[key].coeff_  = data_pca[key].coeff_[itest,:]

    return traindata, valdata, testdata



# =====================================
# Visualize Response Predictions
# =====================================
def visualize_response_prediction(data, preprocess, net, loss_fcn, hp, ishot=0, nsamples=10):

    def inverse_transform(y):
        y = y.detach().numpy()
        y = preprocess.Y_scaler.inverse_transform(y)
        y = preprocess.Y_pca.inverse_transform(y)
        y = y.reshape(65, 65).T
        return y

    shotlist = np.unique(data['shot'])
    shot = shotlist[ishot]
    X, Y,_,_ = preprocess.transform(data, randomize=False)
    iuse = np.where(data['shot'] == shot)[0]
    iuse = np.linspace(min(iuse), max(iuse), nsamples, dtype=int)
    times = data['time'][iuse]
    X = X[iuse, :]
    Y = Y[iuse, :]
    Ypreds = net(X)

    fig = plt.figure(figsize=(20, 10))
    ax = list(range(nsamples))

    for i in range(nsamples):
        ytrue = Y[i, :]
        ypred = Ypreds[i, :]
        loss = loss_fcn(ytrue, ypred)

        psi_true = inverse_transform(ytrue)
        psi_pred = inverse_transform(ypred)

        ax[i] = fig.add_subplot(2, int(np.ceil(nsamples / 2)), i + 1)

        cs = ax[i].contour(psi_true, 20, linestyles='dashed')
        ax[i].contour(psi_pred, levels=cs.levels, linestyles='solid')
        ax[i].legend(('True', 'Prediction'))
        ax[i].set_title('t=%.3fs, Loss=%.5f' % (times[i], loss))

    fig.suptitle('Shot %d' % shot)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if hp.savefigs:
        fn = hp.save_results_dir + '/response' + str(ishot) + '.png'
        plt.savefig(fn, dpi=50)


# =====================================
# Plot prediction of response pca coeff
# =====================================
def plot_response_coeffs(data, preprocess, net, hp, ncoeffs='all'):

    shotlist = np.unique(data['shot'])
    nshots = len(shotlist)

    X, Y,_,_ = preprocess.transform(data)
    if ncoeffs=='all':
        ncoeffs = Y.shape[1]

    for icoeff in range(ncoeffs):

        fig = plt.figure(figsize=(20, 10))
        ax = list(range(nshots))

        for ishot, shot in enumerate(shotlist):
            i = np.where(data['shot'] == shot)[0]
            t = data['time'][i]
            X, Y,_,_ = preprocess.transform(data, randomize=False)
            X = X[i, :]
            Y = Y[i, :]
            Ypred = net(X).detach().numpy()

            ax[ishot] = fig.add_subplot(4, int(np.ceil(nshots / 4)), ishot + 1)
            ax[ishot].plot(t, Y[:, icoeff], linestyle='dashed')
            ax[ishot].plot(t, Ypred[:, icoeff])
            ax[ishot].set_title(shot)

        fig.suptitle('PCA Coeff %d' % (icoeff + 1))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if hp.savefigs:
            fn = hp.save_results_dir + '/coeff' + str(icoeff) + '.png'
            plt.savefig(fn, dpi=50)


# ===============
# Plot loss curve
# ===============
def plot_loss_curve(training_loss, validation_loss, hp):

    plt.figure()
    plt.semilogy(training_loss)
    plt.semilogy(validation_loss)
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    if hp.savefigs:
        fn = hp.save_results_dir + '/loss_curve.png'
        plt.savefig(fn, dpi=50)


# ===========================
# Make movie of shot response
# ===========================
def make_shot_response_movie(data, preprocess, net, loss_fcn, hp, ishot=0):

    def inverse_transform(y):
        y = y.detach().numpy()
        y = preprocess.Y_scaler.inverse_transform(y)
        y = preprocess.Y_pca.inverse_transform(y)
        y = y.reshape(65, 65).T
        return y

    shotlist = np.unique(data['shot'])
    shot = shotlist[ishot]
    X, Y,_,_ = preprocess.transform(data, randomize=False)
    iuse = np.where(data['shot'] == shot)[0]
    Nplots = min(20, len(iuse))
    iuse = np.linspace(min(iuse), max(iuse), Nplots, dtype=int)
    times = data['time'][iuse]
    X = X[iuse, :]
    Y = Y[iuse, :]
    Ypreds = net(X)

    fig, ax = plt.subplots(1, 2, figsize=(6, 6))
    div = make_axes_locatable(ax[1])
    cax = div.append_axes('right', '5%', '5%')

    def animate(i):

        ytrue = Y[i, :]
        ypred = Ypreds[i, :]
        loss = loss_fcn(ytrue, ypred)

        psi_true = inverse_transform(ytrue)
        psi_pred = inverse_transform(ypred)

        psi_min = min(psi_pred.min(), psi_true.min())
        psi_max = max(psi_pred.max(), psi_true.max())
        levels = np.linspace(psi_min, psi_max, 30)

        ax[0].cla()
        ax[0].set_title('True')
        cs = ax[0].contourf(psi_true, levels=levels)
        ax[0].contour(psi_true, levels=levels, linewidths=0.5, colors='k')

        ax[1].cla()
        ax[1].set_title('Prediction')
        ax[1].contourf(psi_pred, levels=levels)
        ax[1].contour(psi_pred, levels=levels, linewidths=0.5, colors='k')

        cax.cla()
        fig.colorbar(cs, cax=cax)
        fig.suptitle('Shot %d, Time %.3f, Loss %.5f' % (shot, times[i], loss))
        cs.set_clim(vmin=levels.min(), vmax=levels.max())

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    anim = animation.FuncAnimation(fig, animate, frames=range(len(times)), interval=50)

    if hp.savemovie and hp.savedir is not None:
        anim.save(hp.savedir + 'response_movie.mp4', fps=10, dpi=50)

    return anim


# ========
# TRAINING
# ========
def train(net, loss_fcn, optimizer, train_dataloader, val_dataloader, hp):

    training_loss = []
    validation_loss = []
    val_dataiter = iter(val_dataloader)

    for epoch in range(hp.num_epochs):
        for i, data in enumerate(train_dataloader):

            x_batch, y_batch = data
            y_pred = net(x_batch)

            # compute loss
            loss = loss_fcn(y_pred, y_batch)

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()

            if i % hp.print_every == 0:

                # save training loss
                training_loss.append(batch_loss)

                # calculate and save validation loss
                try:
                    x, y = val_dataiter.next()
                except StopIteration:
                    val_dataiter = iter(val_dataloader)
                    x, y = val_dataiter.next()
                
                net.eval()
                with torch.no_grad():
                    ypred = net(x)
                    val_loss = loss_fcn(ypred, y).item()
                    validation_loss.append(val_loss)
                net.train()

                print('Epoch: %d of %d, train_loss: %3e, val_loss: %3e' %
                      (epoch + 1, hp.num_epochs, batch_loss, val_loss))

    return net, training_loss, validation_loss


# ==============
# LOSS FUNCTIONS
# ==============
def mse_loss(input, target):
    return torch.mean((input - target) ** 2)

def weighted_mse_loss(input, target, weight=None):
    return torch.mean(weight * (input - target) ** 2)

def LMLS(input, target):
    return torch.mean(torch.log(1 + 0.5 * (input - target)**2))

class Weighted_MSE_Loss():
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
    def loss(self, input, target):
        return torch.mean(self.weights * (input-target)**2)

class PLoss():
    def __init__(self, p=2):
        super().__init__()
        self.p = p
    def loss(self, input, target):
        eps = 1e-12
        e = torch.abs(input-target) + eps
        loss = torch.mean(e**self.p)
        return loss

# ===========================
# MULTILAYER PERCEPTRON CLASS
# ===========================
class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''

    def __init__(self, in_dim, out_dim, hidden_dims, nonlinearity='RELU', p_dropout_in=0, p_dropout_hidden=0.2):
        super().__init__()
        
        if nonlinearity.upper() == 'RELU':
            nonlinearity = nn.ReLU()
        elif nonlinearity.upper() == 'TANH':
            nonlinearity = nn.Tanh()
        elif nonlinearity.upper() == 'LEAKYRELU':
            nonlinearity = nn.LeakyReLU()
        elif nonlinearity.upper() == 'ELU':
            nonlinearity = nn.ELU()
        
        dims = [in_dim]
        dims.extend(hidden_dims)
        dims.extend([out_dim])

        nlayers = len(dims) - 1
        layers = []

        for i in range(nlayers):
            if i == 0:
                p = p_dropout_in
            else:
                p = p_dropout_hidden
            layers.append(nn.Dropout(p=p))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nonlinearity)

        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==================
# DATA PREPROCESSING
# ==================
class DataPreProcess():
    def __init__(self, datadict, xnames, ynames, t_thresh=None):
        super().__init__()

        X = self.makeX(datadict, xnames)
        X_scaler = StandardScaler()
        X_scaler.fit(X)

        Y = self.makeX(datadict, ynames)
        Y_scaler = StandardScaler()
        Y_scaler.fit(Y)

        # write to class object
        self.X_scaler = X_scaler
        self.Y_scaler = Y_scaler
        self.Y_pca = datadict[ynames[0]]
        self.t_thresh = t_thresh        
        self.xnames = xnames
        self.ynames = ynames

    def makeX(self, datadict, xnames):

        for i, key in enumerate(xnames):

            try:
                x = datadict[key].coeff_
            except:
                x = datadict[key]

            if i == 0:
                Xdata = np.copy(x)
            else:
                Xdata = np.hstack([Xdata, x])

        return Xdata

    def transform(self, datadict, randomize=True, holdback_fraction=0.0, by_shot=False):

        X = self.makeX(datadict, self.xnames)
        Y = self.makeX(datadict, self.ynames)

        X = self.X_scaler.transform(X)
        Y = self.Y_scaler.transform(Y)

        time = datadict['time']
        shot = datadict['shot']

        # use only certain times
        if self.t_thresh is not None:
            t = datadict['time']
            iuse = np.where(t > self.t_thresh)[0]
            X = X[iuse, :]
            Y = Y[iuse, :]
            time = time[iuse]
            shot = shot[iuse]
        
        # remove samples with nans
        i = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
        X = X[i,:]
        Y = Y[i,:]
        time = time[i]
        shot = shot[i]

        if holdback_fraction > 0 and by_shot:
            uniqshots = np.unique(shot)
            sz = int( (1.0-holdback_fraction)*len(uniqshots))
            select_shots = np.random.choice(uniqshots, sz, replace=False)
            
            for ishot, select_shot in enumerate(select_shots):
                
                k = np.where(shot == select_shot)[0]
                if ishot==0:
                    idx = k
                else:
                    idx = np.hstack((idx,k))
            
            X = X[idx,:]
            Y = Y[idx,:]
            time = time[idx]
            shot = shot[idx]

        # convert to torch data types
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)


        # randomize order
        if randomize:
            idx = torch.randperm(X.shape[0])
            X = X[idx, :]
            Y = Y[idx, :]
            time = time[idx]
            shot = shot[idx]


        # hold back some samples
        if holdback_fraction > 0 and not by_shot:
            nsamples = X.shape[0]
            nkeep = int((1.0 - holdback_fraction)*nsamples)
            X = X[:nkeep,:]
            Y = Y[:nkeep,:]
            time = time[:nkeep]
            shot = shot[:nkeep]

        return X, Y, shot, time


# ==================
# Growth rate calcs
# ==================
class GrowthRate():
    
    def __init__(self, geometry_files_dir):
        
        self.M = sio.loadmat(geometry_files_dir + '/M.mat')['M']
        self.MpcMpv = sio.loadmat(geometry_files_dir + '/MpcMpv.mat')['MpcMpv']
        self.Mpp = sio.loadmat(geometry_files_dir + '/Mpp.mat')['Mpp']
        self.Pxx = sio.loadmat(geometry_files_dir + '/Pxx.mat')['Pxx']
        self.Rxx = sio.loadmat(geometry_files_dir + '/Rxx.mat')['Rxx']
        self.Mppi = np.linalg.inv(self.Mpp)
    
    def calc_gamma(self, dpsidix_shot):
        
        nsamples = dpsidix_shot.shape[0]
        gamma = np.zeros(nsamples)
        
        for i in range(nsamples):
            
            dpsidix = dpsidix_shot[i,:,:]
            dcphidix = self.Mppi @ dpsidix
            X = self.Pxx.T @ self.MpcMpv.T @ dcphidix
            amat = -np.linalg.inv(self.M + X) @ self.Rxx
            e, _ = np.linalg.eig(amat)
            max_eig = np.max(np.real(e))
            gamma[i] = max_eig
        
        return gamma           

    

# ======================
# plot shape timetraces
# ======================

def plot_shape_timetraces(shotlist, net, valdata, preprocess,hp):
    
    X,Y,shots,times = preprocess.transform(valdata, randomize=False, holdback_fraction=0)
    Ypred = net(X).detach().numpy()    

    Y = preprocess.Y_scaler.inverse_transform(Y)
    Ypred = preprocess.Y_scaler.inverse_transform(Ypred)
    
    
    for shot in shotlist:
        
        i = np.where(shots==shot)[0]
        fig = plt.figure(figsize=(16, 10))
        ax = list(range(len(hp.ynames)))
        
        for k, yname in enumerate(hp.ynames):

            ax[k] = fig.add_subplot(4, 5, k+1)
            ax[k].plot( times[i], Y[i,k], c='r', linestyle='dashed')
            ax[k].plot( times[i], Ypred[i,k], c='b', linestyle='dashed')    
            ax[k].set_ylabel(yname)
            ax[k].set_xlabel('Time [s]')

        fig.suptitle('Shot %d' % shot)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])   

        if hp.savefigs:
            fn = hp.save_results_dir + '/shapes' + str(int(shot)) + '.png'
            plt.savefig(fn, dpi=100)

# ========================
# Save output predictions
# ========================
def gen_output_preds(data, preprocess, net, hp):
    

    X, Y, shots, times = preprocess.transform(data, randomize=False, holdback_fraction=0)
    Ypred = net(X)    
    Ypred = Ypred.detach().numpy()
    Y = Y.detach().numpy()
    X = X.detach().numpy()

    out = {}
    out['X'] = X
    out['Y'] = Y
    out['Ypred'] = Ypred
    out['Y_coeff'] = preprocess.Y_scaler.inverse_transform(Y)
    out['Ypred_coeff'] = preprocess.Y_scaler.inverse_transform(Ypred)
    out['shots'] = shots
    out['times'] = times

    for tag in hp.xnames + hp.ynames:

        pca = data[tag]
        out[tag + '_pca'] = {}
            
        if pca.coeff_.shape[1] <= 1:
            out[tag] = pca.coeff_
        else:
            try:
                out[tag + '_pca']['coeff_'] = pca.coeff_
                out[tag + '_pca']['components_'] = pca.components_
                out[tag + '_pca']['mean_'] = pca.mean_

                if pca.components_.shape[1] < 100:
                    out[tag] = pca.inverse_transform(pca.coeff_)
            except:
                continue
    return out
    


# =====================
# plot flux predictions
# =====================

def plot_flux_preds(shot, plot_times, net, testdata, preprocess, tok_data, hp):

    mpc = tok_data['mpc'][0][0] /  (-2*np.pi)
    mpv = tok_data['mpv'][0][0] /  (-2*np.pi)
    rg = tok_data['rg'][0][0].reshape(-1)
    zg = tok_data['zg'][0][0].reshape(-1)
    rlim = tok_data['limdata'][0][0][1,:]
    zlim = tok_data['limdata'][0][0][0,:]
    nz = len(zg)
    nr = len(rg)

    def inverse_transform(y):
        y = y.detach().numpy()
        y = preprocess.Y_scaler.inverse_transform(y)
        y = preprocess.Y_pca.inverse_transform(y)
        y = y.reshape(-1, 65, 65)
        # y = np.transpose(y, (0,2,1))
        return y

    X, Y, shots, times = preprocess.transform(testdata, randomize=False)        

    # get correct indices for shot and plot_times
    iuse = np.where(shots == shot)[0]
    times = times[iuse]        
    i = [np.argmin(np.abs(times-t)) for t in plot_times]    
    times = np.squeeze(times[i])
    iuse = iuse[i]
    X = X[iuse,:]
    Y = Y[iuse,:]    

    Ypred = net(X)
    flux_efit_projected = inverse_transform(Y)
    flux_pred = inverse_transform(Ypred)

    ic_pca = testdata['coil_currents']
    iv_pca = testdata['vessel_currents']

    ic = ic_pca.inverse_transform(ic_pca.coeff_[iuse,:]).T
    iv = iv_pca.inverse_transform(iv_pca.coeff_[iuse,:]).T

    psizr_app_ic_projected = (mpc @ ic).reshape(nz, nr, -1)
    psizr_app_iv_projected = (mpv @ iv).reshape(nz, nr, -1)

    psizr_app_ic_projected = np.transpose(psizr_app_ic_projected, [2, 1, 0])
    psizr_app_iv_projected = np.transpose(psizr_app_iv_projected, [2, 1, 0])


    if 'psizr_pla_iv' in hp.ynames[0]:
        psizr_pred = flux_pred + psizr_app_ic_projected
        psizr_efit_projected = flux_efit_projected + psizr_app_ic_projected
    
    elif 'psizr_pla' in hp.ynames[0]:
        psizr_pred = flux_pred + psizr_app_ic_projected + psizr_app_iv_projected
        psizr_efit_projected = flux_efit_projected + psizr_app_ic_projected + psizr_app_iv_projected

    
    psi_pred = psizr_pred
    psi_true = psizr_efit_projected

    
    # Plot flux contours
    nsamples = len(times)    
    fig = plt.figure(figsize=(16, 14))
    ax = list(range(nsamples))
    
    for i, t in enumerate(times):
                     
        ax[i] = fig.add_subplot(2, int(np.ceil(nsamples / 2)), i + 1)
        cs1 = ax[i].contour(rg, zg, psi_true[i], 12, colors='r', linestyles='dashed')
        cs2 = ax[i].contour(rg, zg, psi_pred[i], levels=cs1.levels, colors='b', linestyles='solid')
        ax[i].plot(rlim, zlim, color='gray', linewidth=3)
        ax[i].set_aspect('equal')  
        ax[i].set_title('%s: %dms ' %(shot, t*1000), fontweight='bold') 

        if i == 0:
            h1,_ = cs1.legend_elements()
            h2,_ = cs2.legend_elements()
            ax[i].legend([h1[0], h2[0]], ['EFIT01 PCA Projection', 'EQNET'], fontsize=13)
     

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    if hp.savefigs:
        fn = hp.save_results_dir + '/eq' + str(int(shot)) + '.png'
        plt.savefig(fn, dpi=100)
        
        
    













    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
