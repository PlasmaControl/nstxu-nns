from easydict import EasyDict
import numpy as np
import os
import matplotlib.pyplot as plt

ROOT = os.environ['NN_ROOT']
job_topdir = ROOT + 'pertnet/jobs/coil2/scan1/'


N = len(os.listdir(job_topdir))
loss = []

for i in range(N):
    try:
        fn = job_topdir + 'job' + str(i) + '/loss.txt'
        with open(fn, 'r') as reader:
            loss.append(float(reader.read()))
    except:
        loss.append(np.nan)

loss = np.asarray(loss)

idx = loss.argsort()[:10]
best_loss = loss[idx]

print('\nBest jobs:')
print(idx)
print('\nLoss of best jobs:')
print(best_loss)

plt.plot(loss)
plt.show()

# open the figures of the best job
fns = ['coeff0.png', 'loss_curve.png', 'response0.png', 'response1.png', 'response9.png']

for fn in fns:
    cmd = 'xdg-open ' + job_topdir + 'job' + str(idx[0]) + '/' + fn
    os.system(cmd)
