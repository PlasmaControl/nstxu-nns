from easydict import EasyDict
import numpy as np
import os
import matplotlib.pyplot as plt

ROOT = os.environ['NN_ROOT']
job_topdir = ROOT + 'eqnet/jobs/forward-profiles/jobs016a/'


N = len(os.listdir(job_topdir))

loss = []

for i in range(N):
    try:
        loss_fn = job_topdir + 'job' + str(i) + '/loss.txt'
        l = np.loadtxt(loss_fn)[1,-1]
        loss.append(l)
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


cmd = 'xdg-open ' + job_topdir + 'job' + str(idx[0]) + '/eq203172.png'
os.system(cmd)

cmd = 'xdg-open ' + job_topdir + 'job' + str(idx[0]) + '/eq204155.png'
os.system(cmd)

cmd = 'xdg-open ' + job_topdir + 'job' + str(idx[0]) + '/eq204069.png'
os.system(cmd)

cmd = 'xdg-open ' + job_topdir + 'job' + str(idx[0]) + '/eq203942.png'
os.system(cmd)

cmd = 'xdg-open ' + job_topdir + 'job' + str(idx[0]) + '/loss_curve.png'
os.system(cmd)
