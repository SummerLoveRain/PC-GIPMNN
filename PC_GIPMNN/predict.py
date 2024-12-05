from cmath import inf
import logging
import sys
import time
import numpy as np
from scipy.interpolate import griddata
import scipy.io
import torch
from plot.surface import plot_surface_3D
from plot.line import plot_line
from plot.heatmap import plot_heatmap, plot_heatmap3
from model_config import PINNConfig

from init_config import TASK_NAME, get_device, getParentDir, path, root_path
from train_config import *

# 打印相关信息
def log(obj):
    print(obj)
    logging.info(obj)

if __name__ == "__main__":
    start_time = time.time()
    device = get_device(sys.argv)

    # 加载各个区域的坐标数据
    TIME_STR = '20230407_184947'
    
    path = '/' + TASK_NAME + '/' + TIME_STR + '/'
        
    # 预测网格点
    GRID_SIZE = 170
    X = np.linspace(lb[0], ub[0], GRID_SIZE+1) 
    Y = np.linspace(lb[1], ub[1], GRID_SIZE+1)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X_VALID, Y_VALID = np.meshgrid(X, Y)
    x = np.hstack((X_VALID.flatten()[:, None], Y_VALID.flatten()[:, None]))
    x_star = x[:, 0:1]
    y_star = x[:, 1:2]


    # 加载模型
    net_path = root_path + '/' + path + '/PINN.pkl'
    model_config = PINNConfig.reload_config(net_path=net_path)

    x_star = model_config.data_loader(x_star)
    y_star = model_config.data_loader(y_star)
    u_pred = model_config.forward(x_star, y_star)
    x_star = model_config.detach(x_star)
    y_star = model_config.detach(y_star)
        
    for i in range((GRID_SIZE+1)*(GRID_SIZE+1)):
        x = x_star[i, 0]
        y = y_star[i, 0]
        if (((y <= 70)and(x <= 170)) or ((y <= 110)and(x <= 150)) or ((y <= 130)and(x <= 130)) or ((y <= 150)and(x <= 110)) or ((y <= 170)and(x <= 70))):
            continue
        else:
            u_pred[i, 0] = 0

    
    # 归一化
    u_pred = torch.sign(torch.mean(u_pred))*u_pred
    # u_pred = torch.sign(torch.mean(u_pred))*u_pred/torch.norm(u_pred)
    u_pred = u_pred * (u_pred.shape[0]/torch.sum(u_pred))
    
    u_pred = model_config.detach(u_pred)
    x = np.hstack((X_VALID.flatten()[:, None], Y_VALID.flatten()[:, None]))
    u_pred = griddata(x, u_pred.flatten(), (X_VALID, Y_VALID), method='cubic')

    scipy.io.savemat(root_path + '/' +
                     TASK_NAME + '/' + TIME_STR + '/data_pred.mat', {'u': u_pred, 'keff': model_config.keff})
    
    
    # 加载FDM数据
    fdm_fem = 'fem'
    data = scipy.io.loadmat(getParentDir() + '/' + fdm_fem +'_results/data.mat')
    u_fdm = data['u']
    u_fdm = np.sign(np.mean(u_fdm)) * u_fdm
    # u_fdm = np.sign(np.mean(u_fdm)) * u_fdm/np.linalg.norm(u_fdm, ord=2)
    u_fdm = u_fdm * (u_fdm.shape[0]*u_fdm.shape[0]/np.sum(u_fdm))
    
    keff = data['keff'][0, 0]
    u_fdm = u_fdm.flatten()
    u_pred = u_pred.flatten()

    L_infinity_u = np.linalg.norm(u_fdm-u_pred, ord=inf)
    L_infinity_rel_u = L_infinity_u/np.linalg.norm(u_fdm, ord=inf)
    L_2_rel_u = np.linalg.norm(u_fdm-u_pred, ord=2)/np.linalg.norm(u_fdm, ord=2)
    
    L_infinity_keff = np.abs(keff-model_config.keff)
    rel_keff = np.abs(keff-model_config.keff)/np.abs(keff)
    
    
    log_str = 'keff ' + str(model_config.keff) +\
        ' L_infinity_keff '+str(L_infinity_keff) +' rel_keff '+str(rel_keff) +\
        ' L_infinity_u ' + str(L_infinity_u) + ' L_infinity_rel_u ' + str(L_infinity_rel_u) + ' L_2_rel_u ' + str(L_2_rel_u)
    log(log_str)


    u_fdm = u_fdm.reshape((X.size, Y.size))
    u_pred = u_pred.reshape((X.size, Y.size))
    file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/heatmap3'
    plot_heatmap3(X_VALID, Y_VALID, u_fdm, u_pred, E=None, xlabel='x',
                  ylabel='y', T_title=str.upper(fdm_fem), P_title='PRED', file_name=file_name)
    
    file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/heatmap_' + fdm_fem
    plot_heatmap(X_VALID, Y_VALID, u_fdm, xlabel='x',
                  ylabel='y', file_name=file_name)
    
    file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/heatmap_pred'
    plot_heatmap(X_VALID, Y_VALID, u_pred, xlabel='x',
                  ylabel='y', file_name=file_name)
    
    file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/heatmap_abs'
    plot_heatmap(X_VALID, Y_VALID, np.abs(u_fdm-u_pred), xlabel='x',
                  ylabel='y', file_name=file_name)
    
    file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/heatmap_rel'
    plot_heatmap(X_VALID, Y_VALID, np.abs(u_fdm-u_pred)/np.max(u_fdm), xlabel='x',
                  ylabel='y', file_name=file_name)
    
    file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/u_fdm'
    plot_surface_3D(X_VALID, Y_VALID, u_fdm, title=str.upper(fdm_fem), xlabel='x', ylabel='y', zlabel='u', file_name=file_name)
    
    file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/u_pred'
    plot_surface_3D(X_VALID, Y_VALID, u_pred, title='PRED', xlabel='x', ylabel='y', zlabel='u', file_name=file_name)


    elapsed = time.time() - start_time
    print('Predicting time: %.4f' % (elapsed))
