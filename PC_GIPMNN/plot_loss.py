import numpy as np
from plot.line import plot_line

from init_config import TASK_NAME, path, root_path
from train_config import *

# step值越过多少个点再记录
def read_PINN_log(PINN_log_path, step=1):
    keys = ['Loss']
    values = []
    with open(PINN_log_path, 'r') as fs:
        line_num = 0
        while True:
            line = fs.readline()  # 整行读取数据
            if not line:
                break
            # 读取loss
            if 'Iter ' in line:
                line_num = line_num + 1
                if line_num % step != 0:
                    continue
                # 总误差
                datas = line.split(' Loss ')[-1].split(' ')
                if datas is not None:
                    if len(keys) == 1:
                        keys[1:] = datas[1::2]
                    values.append(datas[0::2])
    values = np.array(values, dtype=float)
    keys = np.array(keys)
    return keys, values



if __name__ == "__main__":
    datas = []
    TIME_STR = '20220810_133109'
    path = '/' + TASK_NAME + '/' + TIME_STR + '/'

    PINN_log_path = root_path + '/' + path + '/log.txt'

    keys, values = read_PINN_log(
        PINN_log_path=PINN_log_path, step=1)

    Num_Epoch = len(values)
    epochs = [i for i in range(1, Num_Epoch + 1)]
    epochs = np.array(epochs)

# INFO:root:Adam Iter 2880 Loss 1.6945746 lambda 0.8325684 keff 1.2011025058382696 loss_fuel 0.0021193232 loss_rod 0.059793875 loss_b 0.010333053 loss_norm 0.92915833 loss_lambda 0.69317013 LR 0.001 min_loss 1.4852759 lambda 0.69672567 keff 1.4352851460037572 abs_keff 0.26054651444690125 rel_keff 0.2217910498964391
    # idx = [3, 4, 5, 6, 7, 9, 10]
    idx = [7, 10]
    t_keys = keys[idx]
    # t_values = values[:, idx]

    for i in idx:
        value = values[:, i]
        data = np.stack((epochs, value), 1)
        datas.append(data)

    data_labels = t_keys[:]
    # data_labels = None

    xy_labels = ['Epoch/10', 'Loss']
    plot_line(datas=datas,
              data_labels=data_labels,
              xy_labels=xy_labels,
              title=None,
              file_name=root_path + '/' + path + '/loss',
              ylog=True)
    print('done')
