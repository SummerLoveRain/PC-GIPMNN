import logging
import sys
from pyDOE import lhs
from init_config import get_device, getParentDir, path, root_path, init_log, train_Adam, train_Adam_LBFGS, train_Adam_ResNet
from train_config import *

# 打印相关信息
def log(obj):
    print(obj)
    logging.info(obj)
    
if __name__ == "__main__":
    # 设置需要写日志
    init_log()
    # cuda 调用
    device = get_device(sys.argv)

    param_dict = {
        'lb': lb,
        'ub': ub,
        'device': device,
        'path': path,
        'root_path': root_path,
    }

    keff_path = getParentDir() + '/fem_results/' + 'keff.txt'
    phi_path = getParentDir() + '/fem_results/' + 'phi1.txt'

    keff = None;

    with open(keff_path, 'r') as fs:
        while True:
            line = fs.readline()  # 整行读取数据
            if not line:
                break
            keff = np.array(line, dtype=float);

    x_valid = []
    y_valid = []
    u_valid = []
    with open(phi_path, 'r') as fs:
        while True:
            line = fs.readline()  # 整行读取数据
            if not line:
                break
            data = line.split(' ')
            if data is not None:
                x, y, z = data
                x_valid.append([x])
                y_valid.append([y])
                u_valid.append([z])
        x_valid = np.array(x_valid, dtype=float)
        y_valid = np.array(y_valid, dtype=float)
        u_valid = np.array(u_valid, dtype=float)

    u_valid = np.sign(np.mean(u_valid)) * u_valid
    u_valid = u_valid * (u_valid.shape[0]/np.sum(u_valid))
    u_valid = u_valid * (u_valid.shape[0]/np.sum(u_valid))
    ### 生成训练点 ####

    GRID_SIZE = 170
    
    import scipy.io
    fdm_fem = 'fem'
    data = scipy.io.loadmat(getParentDir() + '/'+fdm_fem+'_results/data.mat')
    keff = data['keff'][0, 0]
    
    X = np.linspace(lb[0], ub[0], GRID_SIZE+1)
    Y = np.linspace(lb[1], ub[1], GRID_SIZE+1)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X_VALID, Y_VALID = np.meshgrid(X, Y)
    x = np.hstack((X_VALID.flatten()[:, None], Y_VALID.flatten()[:, None]))
    x_star = x[:, 0:1]
    y_star = x[:, 1:2]

    x_train = []
    y_train = []

    x_rb = []
    y_rb = []
    n_rb = []
    n_nb = []
    x_nb = []
    y_nb = []

    x_15 = []
    y_15 = []
    n_15 = []
    x_25 = []
    y_25 = []
    n_25 = []
    x_35 = []
    y_35 = []
    n_35 = []
    x_45 = []
    y_45 = []
    n_45 = []
    x_46 = []
    y_46 = []
    n_46 = []
    x_56 = []
    y_56 = []
    n_56 = []
    x_67 = []
    y_67 = []
    n_67 = []
    x_456 = []
    y_456 = []
    n_456 = []

    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_4 = 0
    
    sigma_a = []
    D = []
    v_sigma_f = []
    for i in range((GRID_SIZE+1)*(GRID_SIZE+1)):
        x = x_star[i, 0]
        y = y_star[i, 0]
        
        # if ((y==170 or y==150 or y==130 or y==110 or y==70) and (x>=0 and x<=170))\
        if (y==170 and x>=0 and x<=70)\
        or (y==150 and x>=70 and x<=110)\
        or (y==130 and x>=110 and x<=130)\
        or (y==110 and x>=130 and x<=150)\
        or (y==70 and x>=150 and x<=170)\
        or (x==70 and y>=150 and y<=170)\
        or (x==110 and y>=130 and y<=150)\
        or (x==130 and y>=110 and y<=130)\
        or (x==150 and y>=70 and y<=110)\
        or (x==170 and y>=0 and y<=70):
            if (y==170 and x>0 and x<70):
                n_rb.append([0, 1])
            elif (y==150 and x>70 and x<110):
                n_rb.append([0, 1])
            elif (y==130 and x>110 and x<130):
                n_rb.append([0, 1])
            elif (y==110 and x>130 and x<150):
                n_rb.append([0, 1])
            elif (y==70 and x>150 and x<170):
                n_rb.append([0, 1])
            elif (x==70 and y>150 and y<170):
                n_rb.append([1, 0])
            elif (x==110 and y>130 and y<150):
                n_rb.append([1, 0])
            elif (x==130 and y>110 and y<130):
                n_rb.append([1, 0])
            elif (x==150 and y>70 and y<110):
                n_rb.append([1, 0])
            elif (x==170 and y>0 and y<70):
                n_rb.append([1, 0])
            elif (y==170 and x==0):
                n_rb.append([-1, 1])
            elif (y==170 and x==70):
                n_rb.append([1, 1])
            elif (y==150 and x==70):
                n_rb.append([1, 1])
            elif (y==150 and x==110):
                n_rb.append([1, 1])
            elif (y==130 and x==110):
                n_rb.append([1, 1])
            elif (y==130 and x==130):
                n_rb.append([1, 1])
            elif (y==110 and x==130):
                n_rb.append([1, 1])
            elif (y==110 and x==150):
                n_rb.append([1, 1])
            elif (y==70 and x==150):
                n_rb.append([1, 1])
            elif (y==70 and x==170):
                n_rb.append([1, 1])
            elif (y==0 and x==170):
                n_rb.append([1, -1])
            x_rb.append([x])
            y_rb.append([y])
        elif (x==0 and y>=0 and y<=170) or (y==0 and x>=0 and x<=170):
            if (x==0 and y==0):
                n_nb.append([-1, -1])
            elif (x==0 and y>0 and y<=170):
                n_nb.append([-1, 0])
            elif (y==0 and x>=0 and x<=170):
                n_nb.append([0, -1])
            x_nb.append([x])
            y_nb.append([y])
        elif (((y <= 70)and(x <= 170)) or ((y <= 110)and(x <= 150)) or ((y <= 130)and(x <= 130)) or ((y <= 150)and(x <= 110)) or ((y <= 170)and(x <= 70))):
            x_train.append([x])
            y_train.append([y])

            if (y==10 and x>0 and x<10):
                x_15.append([x])
                y_15.append([y])
                n_15.append([0, 1])
            elif (x==10 and y>0 and y<10):
                x_15.append([x])
                y_15.append([y])
                n_15.append([1, 0])
            elif (y==10 and x==0):
                x_15.append([x])
                y_15.append([y])
                n_15.append([-1, 1])
            elif (y==10 and x==10):
                x_15.append([x])
                y_15.append([y])
                n_15.append([1, 1])
            elif (y==0 and x==10):
                x_15.append([x])
                y_15.append([y])
                n_15.append([1, -1])
                
            elif (x==70 and y>0 and y<10):
                x_25.append([x])
                y_25.append([y])
                n_25.append([-1, 0])
            elif (x==90 and y>0 and y<10):
                x_25.append([x])
                y_25.append([y])
                n_25.append([1, 0])
            elif (y==10 and x>70 and x<90):
                x_25.append([x])
                y_25.append([y])
                n_25.append([0, 1])
            elif (y==0 and x==70):
                x_25.append([x])
                y_25.append([y])
                n_25.append([-1, -1])
            elif (y==0 and x==90):
                x_25.append([x])
                y_25.append([y])
                n_25.append([1, -1])
            elif (y==10 and x==70):
                x_25.append([x])
                y_25.append([y])
                n_25.append([-1, 1])
            elif (y==10 and x==90):
                x_25.append([x])
                y_25.append([y])
                n_25.append([1, 1])
                
                
            elif (y==70 and x>0 and y<10):
                x_35.append([x])
                y_35.append([y])
                n_35.append([0, -1])
            elif (y==90 and x>0 and y<10):
                x_35.append([x])
                y_35.append([y])
                n_35.append([0, 1])
            elif (x==10 and y>70 and y<90):
                x_35.append([x])
                y_35.append([y])
                n_35.append([1, 0])
            elif (x==0 and y==70):
                x_35.append([x])
                y_35.append([y])
                n_35.append([-1, -1])
            elif (x==10 and y==70):
                x_35.append([x])
                y_35.append([y])
                n_35.append([1, -1])
            elif (x==0 and y==90):
                x_35.append([x])
                y_35.append([y])
                n_35.append([-1, 1])
            elif (x==10 and y==90):
                x_35.append([x])
                y_35.append([y])
                n_35.append([1, 1])

            elif (x==70 and y>70 and y<90):
                x_45.append([x])
                y_45.append([y])
                n_45.append([-1, 0])
            elif (y==70 and x>70 and x<90):
                x_45.append([x])
                y_45.append([y])
                n_45.append([0, -1])
            elif (y==70 and x==70):
                x_45.append([x])
                y_45.append([y])
                n_45.append([0, -1])
                
            elif (y==90 and x==70):
                x_456.append([x])
                y_456.append([y])
                n_456.append([-1, 1])
            elif (y==70 and x==90):
                x_456.append([x])
                y_456.append([y])
                n_456.append([1, -1])
                
            elif (y==90 and x>70 and x<90):
                x_46.append([x])
                y_46.append([y])
                n_46.append([0, 1])
            elif (x==90 and y>70 and y<90):
                x_46.append([x])
                y_46.append([y])
                n_46.append([1, 0])
            elif (x==90 and y==90):
                x_46.append([x])
                y_46.append([y])
                n_46.append([1, 1])
                
            elif (y==130 and x>0 and x<30):
                x_56.append([x])
                y_56.append([y])
                n_56.append([0, 1])
            elif (x==30 and y>110 and y<130):
                x_56.append([x])
                y_56.append([y])
                n_56.append([1, 0])
            elif (y==110 and x>30 and x<70):
                x_56.append([x])
                y_56.append([y])
                n_56.append([0, 1])
            elif (x==70 and y>90 and y<110):
                x_56.append([x])
                y_56.append([y])
                n_56.append([1, 0])
            elif (y==70 and x>90 and x<110):
                x_56.append([x])
                y_56.append([y])
                n_56.append([0, 1])
            elif (x==110 and y>30 and y<70):
                x_56.append([x])
                y_56.append([y])
                n_56.append([1, 0])
            elif (y==30 and x>110 and x<130):
                x_56.append([x])
                y_56.append([y])
                n_56.append([0, 1])
            elif (x==130 and y>0 and y<30):
                x_56.append([x])
                y_56.append([y])
                n_56.append([1, 0])
            elif (x==0 and y==130):
                x_56.append([x])
                y_56.append([y])
                n_56.append([-1, 1])
            elif (x==30 and y==130):
                x_56.append([x])
                y_56.append([y])
                n_56.append([1, 1])
            elif (x==30 and y==110):
                x_56.append([x])
                y_56.append([y])
                n_56.append([1, 1])
            elif (x==70 and y==110):
                x_56.append([x])
                y_56.append([y])
                n_56.append([1, 1])
            elif (x==110 and y==70):
                x_56.append([x])
                y_56.append([y])
                n_56.append([1, 1])
            elif (x==110 and y==30):
                x_56.append([x])
                y_56.append([y])
                n_56.append([1, 1])
            elif (x==130 and y==30):
                x_56.append([x])
                y_56.append([y])
                n_56.append([1, 1])
            elif (x==130 and y==0):
                x_56.append([x])
                y_56.append([y])
                n_56.append([1, -1])
                
            elif (y==150 and x>0 and x<50):
                x_67.append([x])
                y_67.append([y])
                n_67.append([0, 1])
            elif (x==50 and y>130 and y<150):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 0])
            elif (y==130 and x>50 and x<90):
                x_67.append([x])
                y_67.append([y])
                n_67.append([0, 1])
            elif (x==90 and y>110 and y<130):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 0])
            elif (y==110 and x>90 and x<110):
                x_67.append([x])
                y_67.append([y])
                n_67.append([0, 1])
            elif (x==110 and y>90 and y<110):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 0])
            elif (y==90 and x>110 and x<130):
                x_67.append([x])
                y_67.append([y])
                n_67.append([0, 1])
            elif (x==130 and y>50 and y<90):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 0])
            elif (y==50 and x>130 and x<150):
                x_67.append([x])
                y_67.append([y])
                n_67.append([0, 1])
            elif (x==150 and y>0 and y<50):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 0])
            elif (x==0 and y==150):
                x_67.append([x])
                y_67.append([y])
                n_67.append([-1, 1])
            elif (x==50 and y==150):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 1])
            elif (x==50 and y==130):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 1])
            elif (x==90 and y==150):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 1])
            elif (x==50 and y==110):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 1])
            elif (x==110 and y==110):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 1])
            elif (x==110 and y==90):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 1])
            elif (x==130 and y==90):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 1])
            elif (x==130 and y==50):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 1])
            elif (x==150 and y==50):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, 1])
            elif (x==150 and y==0):
                x_67.append([x])
                y_67.append([y])
                n_67.append([1, -1])

            if ((x>=70)and(x<90)and(y>=70)and(y<90))\
            or ((x>=0)and(x<10)and(y>=0)and(y<10))\
            or ((x>=70)and(x<90)and(y>=0)and(y<10))\
            or ((x>=0)and(x<10)and(y>=70)and(y<90)):
                sigma_a.append(sigmaA3)
                D.append(D3)
                v_sigma_f.append(vSigmaF3)
                num_3 = num_3 + 1
            elif ((x>=0)and(x<50)and(y>=130)and(y<150))\
            or ((x>=30)and(x<90)and(y>=110)and(y<130))\
            or ((x>=70)and(x<110)and(y>=90)and(y<110))\
            or ((x>=90)and(x<130)and(y>=70)and(y<90))\
            or ((x>=110)and(x<130)and(y>=30)and(y<70))\
            or ((x>=130)and(x<150)and(y>=0)and(y<50)):
                sigma_a.append(sigmaA1)
                D.append(D1)
                v_sigma_f.append(vSigmaF1)
                num_1 = num_1 + 1
            elif ((x>=0 and x<=70)and(y>=150 and y<=170))\
            or((x>=50 and x<=110)and(y>=130 and y<=150))\
            or ((x>=90 and x<=130)and(y>=110 and y<=130))\
            or ((x>=110 and x<=150)and(y>=90 and y<=110))\
            or ((x>=130 and x<=150)and(y>=50 and y<=90))\
            or ((x>=150 and x<=170) and (y>=0 and y<=70)):
                sigma_a.append(sigmaA4)
                D.append(D4)
                v_sigma_f.append(vSigmaF4)
                num_4 = num_4 + 1
            else:
                sigma_a.append(sigmaA2)
                D.append(D2)
                v_sigma_f.append(vSigmaF2)
                num_2 = num_2 + 1

    sigma_a = np.expand_dims(sigma_a, axis=1)
    D = np.expand_dims(D, axis=1)
    v_sigma_f = np.expand_dims(v_sigma_f, axis=1)
    
    x_nb = np.array(x_nb, dtype=float)
    y_nb = np.array(y_nb, dtype=float)
    n_nb = np.array(n_nb, dtype=float)
    x_rb = np.array(x_rb, dtype=float)
    y_rb = np.array(y_rb, dtype=float)
    n_rb = np.array(n_rb, dtype=float)
    x_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=float)

    x_15 = np.array(x_15, dtype=float)
    y_15 = np.array(y_15, dtype=float)
    n_15 = np.array(n_15, dtype=float)
    x_25 = np.array(x_25, dtype=float)
    y_25 = np.array(y_25, dtype=float)
    n_25 = np.array(n_25, dtype=float)
    x_35 = np.array(x_35, dtype=float)
    y_35 = np.array(y_35, dtype=float)
    n_35 = np.array(n_35, dtype=float)
    x_45 = np.array(x_45, dtype=float)
    y_45 = np.array(y_45, dtype=float)
    n_45 = np.array(n_45, dtype=float)
    x_46 = np.array(x_46, dtype=float)
    y_46 = np.array(y_46, dtype=float)
    n_46 = np.array(n_46, dtype=float)
    x_56 = np.array(x_56, dtype=float)
    y_56 = np.array(y_56, dtype=float)
    n_56 = np.array(n_56, dtype=float)
    x_67 = np.array(x_67, dtype=float)
    y_67 = np.array(y_67, dtype=float)
    n_67 = np.array(n_67, dtype=float)
    x_456 = np.array(x_456, dtype=float)
    y_456 = np.array(y_456, dtype=float)
    n_456 = np.array(n_456, dtype=float)

    # 训练参数
    train_dict = {
        'x_nb': x_nb,
        'y_nb': y_nb,
        'n_nb': n_nb,
        'x_rb': x_rb,
        'y_rb': y_rb,
        'n_rb': n_rb,
        'x_15':x_15,
        'y_15':y_15,
        'n_15':n_15,
        'x_25':x_25,
        'y_25':y_25,
        'n_25':n_25,
        'x_35':x_35,
        'y_35':y_35,
        'n_35':n_35,
        'x_45':x_45,
        'y_45':y_45,
        'n_45':n_45,
        'x_46':x_46,
        'y_46':y_46,
        'n_46':n_46,
        'x_56':x_56,
        'y_56':y_56,
        'n_56':n_56,
        'x_67':x_67,
        'y_67':y_67,
        'n_67':n_67,
        'x_456':x_456,
        'y_456':y_456,
        'n_456':n_456,
        'x_train': x_train,
        'y_train': y_train,
        'x_valid': x_valid,
        'y_valid': y_valid,
        'u_valid': u_valid,
        # 其余参数
        'keff': keff,
        'sigma_a': sigma_a,
        'D': D,
        'v_sigma_f': v_sigma_f,
        'D1': D1,
        'D2': D2,
        'D3': D3,
        'D4': D4,
    }

    # layers = [2, 20, 20, 20, 20, 1]
    # layers = [2, 40, 40, 40, 40, 1]
    # layers = [2, 40, 40, 40, 40, 40, 40, 40, 40, 1]
    # log(layers)

    # 训练
    # train_Adam_LBFGS(layers, device, param_dict, train_dict, Adam_steps=20000, LBFGS_steps=10000)
    # train_Adam(layers, device, param_dict, train_dict, Adam_steps=50000)
    # train_Adam(layers, device, param_dict, train_dict, Adam_steps=100000)
    # train_Adam(layers, device, param_dict, train_dict, Adam_steps=200000)
    # train_Adam_LBFGS(layers, device, param_dict, train_dict, Adam_steps=100000, LBFGS_steps=10000)
    
    in_num = 2
    out_num = 7
    # block_layers = [40, 40]
    block_layers = [20, 20]
    block_num = 2
    log_str = 'in_num ' + str(in_num) + ' out_num ' + str(out_num) + ' block_layers ' + str(block_layers) + ' block_num ' + str(block_num)
    log(log_str)
    # train_Adam_ResNet(in_num, out_num, block_layers, block_num, device, param_dict, train_dict, Adam_steps=50000, Adam_init_lr=1e-3)
    train_Adam_ResNet(in_num, out_num, block_layers, block_num, device, param_dict, train_dict, Adam_steps=500000, Adam_init_lr=1e-3)
