from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from mds_utils import *
import copy
import os
from pdb import set_trace
import mat73
from scipy.ndimage import median_filter


def main():
    
    # Settings
    ROOT = os.environ['NN_ROOT']
    smoothit = False  # smoothit=True is not recommended
    window = None     # window size if smoothit = True
    evt = 0.999       # explained variance threshold
    t_rampup = 0.2    
    ncomps_max = 20
    save_suffix = '009.dat'
    save_dir = ROOT + 'pertnet/data/datasets/'

    # load data

    datadir = ROOT + 'data/rawdata/data_by_var/'
    traindir = datadir + 'train/'
    valdir = datadir + 'val/'
    testdir = datadir + 'test/'
   
    xnames = ['xmat', 'gamma', 'pprime', 'ffprim', 'pres','rmaxis','zmaxis',
              'psirz', 'coil_currents', 'vessel_currents', 'pcurrt', 'rcur', 'zcur', 'ip',
              'qpsi', 'psimag', 'psibry', 'rbbbs', 'zbbbs']
    
    # xnames = xnames + ['dpsidix_smooth_coil1']
    xnames = xnames + ['dpsidix_smooth_coil' + str(icoil) for icoil in range(1,55)]
    
    train_pca = {}
    val_pca = {}
    test_pca = {}

    for xn in xnames:

        print('Loading ' + xn + '...')        

        trainX, trainshots, traintimes = loadX(traindir, xn, smoothit=smoothit, window=window)
        valX, valshots, valtimes = loadX(valdir, xn, smoothit=smoothit, window=window)
        testX, testshots, testtimes = loadX(testdir, xn, smoothit=smoothit, window=window)

        print('  fitting ' + xn + '...')
        pca = fit_pca(trainX, xn, traintimes, t_rampup=t_rampup, explained_variance_thresh=evt, ncomps_max=ncomps_max)

        print('  measuring coefficients...')
        train_pca[xn] = eval_pca(trainX, pca)
        val_pca[xn] = eval_pca(valX, pca)
        test_pca[xn] = eval_pca(testX, pca)
            
    train_pca['shot'] = trainshots
    train_pca['time'] = traintimes
    val_pca['shot'] = valshots
    val_pca['time'] = valtimes
    test_pca['shot'] = testshots
    test_pca['time'] = testtimes

    print('Saving model...')

    train_fn = save_dir + 'train_' + save_suffix
    val_fn = save_dir + 'val_' + save_suffix
    test_fn = save_dir + 'test_' + save_suffix

    save_data(train_pca, train_fn)
    save_data(val_pca, val_fn)        
    save_data(test_pca, test_fn)        

    print('Done')


def load(datadir, varname):
    fn = datadir + varname + '.mat'
    X = mat73.loadmat(fn)[varname]
    X = X.reshape(X.shape[0], -1)
    return X

def loadX(datadir, varname, smoothit=False, window=5):
    iuse = load(datadir, 'igood')
    iuse = np.squeeze(iuse).astype(bool)

    shot = load(datadir, 'shot')
    time = load(datadir, 'time')

    X = load(datadir, varname)
    X = X[iuse,:]
    shot = shot[iuse]
    time = time[iuse]

    if smoothit:
        X = median_filter(X, size=(window,1))
    
    return X, shot, time


def eval_pca(X, pca, smooth_coeffs=False, window=5):
    if pca is None:
        return X
    else:
        coeff = pca.transform(X)
        if smooth_coeffs:
            coeff = median_filter(coeff, size=(window,1))
        pca.coeff_ = coeff
        return copy.deepcopy(pca)


def fit_pca(X, xname, time, t_rampup=0.2, explained_variance_thresh=0.999, ncomps_max=20):

    if X.shape[1] > 1:
        
        iramp = np.where(time < t_rampup)[0]
        iflat = np.where(time > t_rampup)[0]
        iflat_sampled = np.random.choice(iflat, len(iramp))
        ifit = np.concatenate([iramp, iflat_sampled])
    
        Xfit = X[ifit]
        
        n_components = min(min(Xfit.shape), 100) - 1

        pca = PCA(n_components=n_components, svd_solver='arpack')
        pca.fit(Xfit)

        n_components = np.where(np.cumsum(pca.explained_variance_ratio_)
                                > explained_variance_thresh)[0] + 1

        if n_components.size == 0 or n_components[0] > ncomps_max:
            n_components = ncomps_max
        else:
            n_components = n_components[0]

        
        expl_var = np.sum(pca.explained_variance_ratio_[:n_components])
        print('  using %d components. Explained variance %f' % (n_components, expl_var))

        pca = PCA(n_components=n_components, svd_solver='arpack')
        pca.fit(Xfit)
    else:
        pca = None
    
    return pca


if __name__ == '__main__':
    main()
