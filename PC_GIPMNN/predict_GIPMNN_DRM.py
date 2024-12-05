from cmath import inf
import logging
import sys
import time
import numpy as np
from scipy.interpolate import griddata
import scipy.io
import torch
from plot.line import plot_line
from plot.surface import plot_surface_3D
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

    # TIME_STR_arr = [['20221018_150534', '20221018_101731', '20221019_171046', '20221102_031153'],
    #                 ['20221018_154447', '20221018_101736', '20221019_171054', '20221102_031159'],
    #                 ['20221018_154453', '20221018_101742', '20221019_171100', '20221102_031205'],
    #                 ['20221018_154458', '20221018_101748', '20221019_171113', '20221102_031211'],
    #                 ['20221018_154503', '20221018_101754', '20221019_171119', '20221102_031217'],
    #                 ['20221018_150534', '20221018_101731', '20221019_171046', '20221102_031153'],
    #                 ['20221018_154447', '20221018_101736', '20221019_171054', '20221102_031159'],
    #                 ['20221019_073702', '20221018_114853', '20221019_174136', '20221102_031224'],
    #                 ['20221019_073708', '20221018_114900', '20221019_174141', '20221102_031230'],
    #                 ['20221019_073714', '20221018_114906', '20221019_174146', '20221102_031237'],
    #                 ['20221019_073720', '20221018_114912', '20221019_174153', '20221102_031243'],
    #                 ['20221019_073726', '20221018_114917', '20221019_174159', '20221102_031250']]
    
    # TIME_STR_arr = [['20221102_031153'],
    #                 ['20221102_031159'],
    #                 ['20221102_031205'],
    #                 ['20221102_031211'],
    #                 ['20221102_031217'],
    #                 ['20221102_031153'],
    #                 ['20221102_031159'],
    #                 ['20221102_031224'],
    #                 ['20221102_031230'],
    #                 # ['20221102_031237'], 
    #                 ['20221103_115355'],
    #                 ['20221102_031243'],
    #                 ['20221102_031250']]

    TIME_STR_arr = [['20230407_184947']]
        
    # 预测网格点
    GRID_SIZE = 170
    # GRID_SIZE = 200
    # GRID_SIZE = 400
    # GRID_SIZE = 900
    X = np.linspace(lb[0], ub[0], GRID_SIZE+1) 
    Y = np.linspace(lb[1], ub[1], GRID_SIZE+1)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X_VALID, Y_VALID = np.meshgrid(X, Y)
    x = np.hstack((X_VALID.flatten()[:, None], Y_VALID.flatten()[:, None]))
    x_star = x[:, 0:1]
    y_star = x[:, 1:2]

    
    # 加载FDM数据
    fdm_fem = 'fem'    
    
    data_labels = ['$\phi_{'+str.upper(fdm_fem) +'}$', '$\phi_{GIPMNN}$', '$\phi_{DRM}$', '$\phi_{PC-GIPMNN}$']
    TASK_NAMEs = ['task_DIS_IPMNN']
    names = ['pc_gipmnn']
    vmax_list = [0.090 ]
    
    for TIME_STRs, vmax in zip(TIME_STR_arr, vmax_list):

        # 加载FDM数据
        data = scipy.io.loadmat(getParentDir() + '/' + fdm_fem +'_results/data.mat')
        u_fdm = data['u']
        # u_fdm = np.sign(np.mean(u_fdm)) * u_fdm
        u_fdm = np.abs(u_fdm)
        # u_fdm = np.sign(np.mean(u_fdm)) * u_fdm/np.linalg.norm(u_fdm, ord=2)
        u_fdm = u_fdm.flatten()
        u_fdm = np.expand_dims(u_fdm, axis=1)
        u_fdm = u_fdm * (u_fdm.shape[0]*u_fdm.shape[1]/np.sum(u_fdm))

        u_fdm2 = u_fdm.copy()
        
        keff = data['keff'][0, 0]
            
        for TIME_STR, TASK_NAME, name in zip(TIME_STRs, TASK_NAMEs, names):
            path = '/' + TASK_NAME + '/' + TIME_STR + '/'

            # 加载模型
            net_path = root_path + '/' + path + '/PINN.pkl'
            model_config = PINNConfig.reload_config(net_path=net_path)

            x_star = model_config.data_loader(x_star)
            y_star = model_config.data_loader(y_star)
            u_pred = model_config.forward(x_star, y_star)
            x_star = model_config.detach(x_star)
            y_star = model_config.detach(y_star)
            
            # 归一化
            u_pred = torch.sign(torch.mean(u_pred))*u_pred
            # u_pred = torch.sign(torch.mean(u_pred))*u_pred/torch.norm(u_pred)
            u_pred = u_pred * (u_pred.shape[0]/torch.sum(u_pred))

            for i in range((GRID_SIZE+1)*(GRID_SIZE+1)):
                x = x_star[i, 0]
                y = y_star[i, 0]

                # if (x==150 and y==70) or (x==130 and y==110) or (x==110 and y==130) or (x==70 and y==150):
                #     # u_pred[i, 0] = u_pred[i, 0]
                #     u_pred[i, 0] = 255
                #     print(str(x) + ' ' + str(y))

                if (((y <= 70)and(x <= 170)) or ((y <= 110)and(x <= 150)) or ((y <= 130)and(x <= 130)) or ((y <= 150)and(x <= 110)) or ((y <= 170)and(x <= 70))):
                    continue
                else:
                    u_pred[i, 0] = 255
                    u_fdm[i, 0] = 255

            
            u_pred = model_config.detach(u_pred)
            x = np.hstack((X_VALID.flatten()[:, None], Y_VALID.flatten()[:, None]))
            u_pred = griddata(x, u_pred.flatten(), (X_VALID, Y_VALID), method='cubic')
            u_fdm = griddata(x, u_fdm.flatten(), (X_VALID, Y_VALID), method='cubic')

            scipy.io.savemat(root_path + '/output' + '/data_'+ str(GRID_SIZE) + "_" + name +'.mat', {'u': u_pred, 'keff': model_config.keff})
            scipy.io.savemat(root_path + '/output' + '/' + fdm_fem + '_'+ str(GRID_SIZE) + "_" + name +'.mat', {'u': u_fdm, 'keff': model_config.keff})
            
            # u_fdm = u_fdm.flatten()
            # u_pred = u_pred.flatten()

            L_infinity_u = np.linalg.norm(u_fdm-u_pred, ord=inf)
            L_infinity_rel_u = L_infinity_u/np.linalg.norm(u_fdm, ord=inf)
            L_2_rel_u = np.linalg.norm(u_fdm-u_pred, ord=2)/np.linalg.norm(u_fdm, ord=2)
            
            L_infinity_keff = np.abs(keff-model_config.keff)
            rel_keff = np.abs(keff-model_config.keff)/np.abs(keff)
            
            
            # log_str = 'keff_true ' + str(keff) + ' keff ' + str(model_config.keff) +\
            #     ' L_infinity_keff '+str(L_infinity_keff) +' rel_keff '+str(rel_keff) +\
            #     ' L_infinity_u ' + str(L_infinity_u) + ' L_infinity_rel_u ' + str(L_infinity_rel_u) + ' L_2_rel_u ' + str(L_2_rel_u)
            
            log_str = 'keff_true ' + str(keff) + ' keff ' + str(model_config.keff) + ' rel_keff '+str(rel_keff) +\
                ' L_infinity_rel_u ' + str(L_infinity_rel_u)
            log(log_str)
            
            u_fdm = u_fdm.reshape((X.size, Y.size))
            u_fdm2 = u_fdm2.reshape((X.size, Y.size))
            u_pred = u_pred.reshape((X.size, Y.size))
            # file_name = root_path + '/output' + '/heatmap3_' + title + '_' + name
            # plot_heatmap3(X_VALID, Y_VALID, u_fdm, u_pred, E=None, xlabel='x',
            #             ylabel='y', T_title=str.upper(fdm_fem), P_title='PRED', file_name=file_name)
            
            file_name = root_path + '/output' + '/heatmap_' + fdm_fem + '_' + name
            plot_heatmap(X_VALID, Y_VALID, u_fdm, xlabel='x',
                        ylabel='y', file_name=file_name, vmax=7)
            
            # file_name = root_path + '/output' + '/heatmap_pred_' + title + '_' + name
            # plot_heatmap(X_VALID, Y_VALID, u_pred, xlabel='x',
            #             ylabel='y', file_name=file_name)
            
            # file_name = root_path + '/output' + '/heatmap_abs_' + title + '_' + name
            # plot_heatmap(X_VALID, Y_VALID, np.abs(u_fdm-u_pred), xlabel='x',
            #             ylabel='y', file_name=file_name)
            
            file_name = root_path + '/output' + '/heatmap_rel_' + name
            plot_heatmap(X_VALID, Y_VALID, np.abs(u_fdm2-u_pred)/np.max(u_fdm2), xlabel='x',
                        ylabel='y', file_name=file_name, vmax=vmax)
            
            # file_name = root_path + '/output' + '/u_fem_' + title + '_' + name
            # plot_surface_3D(X_VALID, Y_VALID, u_fdm, title=str.upper(fdm_fem), xlabel='x', ylabel='y', zlabel='u', file_name=file_name)
            
            # file_name = root_path + '/output' + '/u_pred_' + title + '_' + name
            # plot_surface_3D(X_VALID, Y_VALID, u_pred, title='PRED', xlabel='x', ylabel='y', zlabel='u', file_name=file_name)
                    


    elapsed = time.time() - start_time
    print('Predicting time: %.4f' % (elapsed))
