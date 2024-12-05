import os

import numpy as np;
import scipy.io

def getcurDir():
    curfilePath = os.path.abspath(__file__)

    # this will return current directory in which python file resides.
    curDir = os.path.abspath(os.path.join(curfilePath, os.pardir))
    # parentDirName = os.path.split(parentDir)[-1]
    return curDir


root_path = getcurDir();

keff_path = root_path + '/fem_results/' + 'keff.txt'
phi_path = root_path + '/fem_results/' + 'phi1.txt'

keff = None;

with open(keff_path, 'r') as fs:
    while True:
        line = fs.readline()  # 整行读取数据
        if not line:
            break
        keff = np.array(line, dtype=float);

u = []
with open(phi_path, 'r') as fs:
    while True:
        line = fs.readline()  # 整行读取数据
        if not line:
            break
        data = line.split(' ')
        if data is not None:
            x, y, z = data
            u.append([z])
    u = np.array(u, dtype=float)
    u = np.abs(u)

GRID_SIZE = 170
# 设置定义域
lb = np.array([0, 0])
ub = np.array([170, 170])
X = np.linspace(lb[0], ub[0], GRID_SIZE+1) 
Y = np.linspace(lb[1], ub[1], GRID_SIZE+1)
X = np.asarray(X, dtype=float)
Y = np.asarray(Y, dtype=float)
X_VALID, Y_VALID = np.meshgrid(X, Y)
x = np.hstack((X_VALID.flatten()[:, None], Y_VALID.flatten()[:, None]))

from scipy.interpolate import griddata
u = griddata(x, u.flatten(), (X_VALID, Y_VALID), method='cubic')
u = np.transpose(u)
scipy.io.savemat(root_path + '/fem_results/' + '/data.mat', {'u': u, 'keff': keff})

        
        

        