import logging
import time
import numpy as np
import torch
from base_config import BaseConfig
from cmath import inf


# 打印相关信息
def log(obj):
    print(obj)
    logging.info(obj)


class PINNConfig(BaseConfig):
    def __init__(self, param_dict, train_dict, model):
        super().__init__()
        self.init(loss_name='sum')
        self.model = model
        # 设置使用设备:cpu, cuda
        lb, ub, self.device, self.path, self.root_path = self.unzip_param_dict(
            param_dict=param_dict)
        # 上下界
        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)

        # 加载训练参数
        x_nb, y_nb, n_nb, x_rb, y_rb, n_rb, \
        x_15, y_15, n_15, x_25, y_25, n_25, x_35, y_35, n_35, x_45, y_45, n_45,\
        x_46, y_46, n_46, x_56, y_56, n_56, x_67, y_67, n_67, x_456, y_456, n_456,\
        x_train, y_train, x_valid, y_valid, u_valid, self.keff_true, sigma_a, D, v_sigma_f, \
        self.D1, self.D2, self.D3, self.D4 = self.unzip_train_dict(train_dict=train_dict)

        # 加载边界条件数据
        self.x_nb = self.data_loader(x_nb)
        self.y_nb = self.data_loader(y_nb)
        self.n_nb_2 = self.data_loader(n_nb[:, 1:2])
        self.n_nb_1 = self.data_loader(n_nb[:, 0:1])
        self.x_rb = self.data_loader(x_rb)
        self.y_rb = self.data_loader(y_rb)
        self.n_rb_1 = self.data_loader(n_rb[:, 0:1])
        self.n_rb_2 = self.data_loader(n_rb[:, 1:2])

        self.x_15 = self.data_loader(x_15)
        self.y_15 = self.data_loader(y_15)
        self.n_15_1 = self.data_loader(n_15[:, 1:2])
        self.n_15_2 = self.data_loader(n_15[:, 0:1])
        self.x_25 = self.data_loader(x_25)
        self.y_25 = self.data_loader(y_25)
        self.n_25_1 = self.data_loader(n_25[:, 1:2])
        self.n_25_2 = self.data_loader(n_25[:, 0:1])
        self.x_35 = self.data_loader(x_35)
        self.y_35 = self.data_loader(y_35)
        self.n_35_1 = self.data_loader(n_35[:, 1:2])
        self.n_35_2 = self.data_loader(n_35[:, 0:1])
        self.x_45 = self.data_loader(x_45)
        self.y_45 = self.data_loader(y_45)
        self.n_45_1 = self.data_loader(n_45[:, 1:2])
        self.n_45_2 = self.data_loader(n_45[:, 0:1])
        self.x_46 = self.data_loader(x_46)
        self.y_46 = self.data_loader(y_46)
        self.n_46_1 = self.data_loader(n_46[:, 1:2])
        self.n_46_2 = self.data_loader(n_46[:, 0:1])
        self.x_56 = self.data_loader(x_56)
        self.y_56 = self.data_loader(y_56)
        self.n_56_1 = self.data_loader(n_56[:, 1:2])
        self.n_56_2 = self.data_loader(n_56[:, 0:1])
        self.x_67 = self.data_loader(x_67)
        self.y_67 = self.data_loader(y_67)
        self.n_67_1 = self.data_loader(n_67[:, 1:2])
        self.n_67_2 = self.data_loader(n_67[:, 0:1])
        self.x_456 = self.data_loader(x_456)
        self.y_456 = self.data_loader(y_456)
        self.n_456_1 = self.data_loader(n_456[:, 1:2])
        self.n_456_2 = self.data_loader(n_456[:, 0:1])


        self.x_train = self.data_loader(x_train)
        self.y_train = self.data_loader(y_train)
        self.x_valid = self.data_loader(x_valid)
        self.y_valid = self.data_loader(y_valid)
        # self.u_valid = self.data_loader(u_valid, requires_grad=False)
        self.u_valid = u_valid
        
        # 加载参数列表
        self.sigma_a = self.data_loader(sigma_a, requires_grad=False)
        self.D = self.data_loader(D, requires_grad=False)
        self.v_sigma_f = self.data_loader(v_sigma_f, requires_grad=False)
        
        self.keff = None
        self.lambda_ = None        
        
        self.N = self.x_train.shape[0]
        # u = np.zeros(shape=[self.N_R, 1])
        # u[0, 0] = 1
        u = np.random.rand(self.N, 1)
        
        # u = u/np.linalg.norm(u)
        norm_u = np.sign(np.mean(u))*u
        norm_u = norm_u * (norm_u.shape[0]/np.sum(norm_u))
        
        self.u = self.data_loader(norm_u, requires_grad=False)
        self.lambda_last = 1

    # 训练参数初始化
    def init(self, loss_name='mean', model_name='PINN'):
        self.start_time = None
        # 小于这个数是开始保存模型
        self.min_loss = 1e8
        # 记录运行步数
        self.nIter = 0
        # 损失计算方式
        if loss_name == 'sum':
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        else:
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        # 保存模型的名字
        self.model_name = model_name

    # 参数读取

    def unzip_param_dict(self, param_dict):
        param_data = (param_dict['lb'], param_dict['ub'],
                      param_dict['device'], param_dict['path'],
                      param_dict['root_path'])
        return param_data

    def unzip_train_dict(self, train_dict):
        train_data = (
            train_dict['x_nb'],
            train_dict['y_nb'],
            train_dict['n_nb'],
            train_dict['x_rb'],
            train_dict['y_rb'],
            train_dict['n_rb'],
            

            train_dict['x_15'],
            train_dict['y_15'],
            train_dict['n_15'],
            train_dict['x_25'],
            train_dict['y_25'],
            train_dict['n_25'],
            train_dict['x_35'],
            train_dict['y_35'],
            train_dict['n_35'],
            train_dict['x_45'],
            train_dict['y_45'],
            train_dict['n_45'],
            train_dict['x_46'],
            train_dict['y_46'],
            train_dict['n_46'],
            train_dict['x_56'],
            train_dict['y_56'],
            train_dict['n_56'],
            train_dict['x_67'],
            train_dict['y_67'],
            train_dict['n_67'],
            train_dict['x_456'],
            train_dict['y_456'],
            train_dict['n_456'],

            train_dict['x_train'],
            train_dict['y_train'],
            train_dict['x_valid'],
            train_dict['y_valid'],
            train_dict['u_valid'],
            train_dict['keff'],
            train_dict['sigma_a'],
            train_dict['D'],
            train_dict['v_sigma_f'],
            train_dict['D1'],
            train_dict['D2'],
            train_dict['D3'],
            train_dict['D4'],
        )
        return train_data

    def net_model(self, x, y):
        X = torch.cat((x, y), 1)
        X = self.coor_shift(X, self.lb, self.ub)
        result = self.model.forward(X)
        result = result ** 2
        u = result[:, 0:1]
        v = result[:, 1:2]
        w = result[:, 2:3]
        r = result[:, 3:4]
        p = result[:, 4:5]
        q = result[:, 5:6]
        h = result[:, 6:7]
        return u, v, w, r, p, q, h

    def forward(self, x, y):
        u, v, w, r, p, q, h = self.net_model(x, y)
        # bu = (x>=0) & (x<=10) & (y>=0) & (y<=10)
        bu = (x>=0) & (x<10) & (y>=0) & (y<10)
        result_u = torch.where(bu, 1, 0)
        # bv = (x>=70) & (x<=90) & (y>=0) & (y<=10)
        bv = (x>=70) & (x<90) & (y>=0) & (y<10)
        result_v = torch.where(bv, 1, 0)
        # bw = (x>=0) & (x<=10) & (y>=70) & (y<=90)
        bw = (x>=0) & (x<10) & (y>=70) & (y<90)
        result_w = torch.where(bw, 1, 0)
        # br = (x>=70) & (x<=90) & (y>=70) & (y<=90)
        br = (x>=70) & (x<90) & (y>=70) & (y<90)
        result_r = torch.where(br, 1, 0)
        # bp = ((x>=0) & (x<=30) & (y>=0) & (y<=130))|\
        #     ((x>=30) & (x<=70) & (y>=0) & (y<=110))|\
        #     ((x>=70) & (x<=90) & (y>=0) & (y<=90))|\
        #     ((x>=90) & (x<=110) & (y>=0) & (y<=70))|\
        #     ((x>=110) & (x<=130) & (y>=0) & (y<=30))
        bp = ((x>=0) & (x<30) & (y>=0) & (y<130))|\
            ((x>=30) & (x<70) & (y>=0) & (y<110))|\
            ((x>=70) & (x<90) & (y>=0) & (y<90))|\
            ((x>=90) & (x<110) & (y>=0) & (y<70))|\
            ((x>=110) & (x<130) & (y>=0) & (y<30))
        result_p = torch.where(bp, 1, 0) - result_u - result_v - result_w - result_r
        # bq = ((x>=0) & (x<=50) & (y>=0) & (y<=150))|\
        #      ((x>=50) & (x<=90) & (y>=0) & (y<=130))|\
        #      ((x>=90) & (x<=110) & (y>=0) & (y<=110))|\
        #      ((x>=110) & (x<=130) & (y>=0) & (y<=90))|\
        #      ((x>=130) & (x<=150) & (y>=0) & (y<=50))
        bq = ((x>=0) & (x<50) & (y>=0) & (y<150))|\
             ((x>=50) & (x<90) & (y>=0) & (y<130))|\
             ((x>=90) & (x<110) & (y>=0) & (y<110))|\
             ((x>=110) & (x<130) & (y>=0) & (y<90))|\
             ((x>=130) & (x<150) & (y>=0) & (y<50))
        result_q = torch.where(bq, 1, 0) - result_p - result_u - result_v - result_w - result_r
        bh = (x>=0) & (x<=170) & (y>=0) & (y<=170)
        result_h = torch.where(bh, 1, 0) - result_q - result_p - result_u - result_v - result_w - result_r
        result = result_u*u + result_v*v+result_w*w+result_r*r+result_p*p+result_q*q+result_h*h
        
        # t1 = torch.sum(result_u)
        # t2 = torch.sum(result_v)
        # t3 = torch.sum(result_w)
        # t4 = torch.sum(result_r)
        # t5 = torch.sum(result_p)
        # t6 = torch.sum(result_q)
        # t7 = torch.sum(result_h)
        return result

    # 训练一次
    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()

        # 初始化loss为0
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        self.loss.requires_grad_()

        x = self.x_train
        y = self.y_train
        u = self.forward(x, y)
        u_x = self.compute_grad(u, x)
        u_xx = self.compute_grad(u_x, x)
        u_y = self.compute_grad(u, y)
        u_yy = self.compute_grad(u_y, y)

        # Rayleigh-Quotient 计算最小特征值 lambda_
        # <Lu, u>/<u, u>
        Lu = -self.D * (u_xx+u_yy) + self.sigma_a * u
        
        loss_IPM = self.loss_func(Lu - self.lambda_last * self.v_sigma_f * self.u)

        x_nb = self.x_nb
        y_nb = self.y_nb
        u_nb = self.forward(x_nb, y_nb)
        u_nb_x = self.compute_grad(u_nb, x_nb)
        u_nb_y = self.compute_grad(u_nb, y_nb)
        equation_nb = u_nb_x*self.n_nb_1 + u_nb_y*self.n_nb_2
        loss_Nb = self.loss_func(equation_nb)

        x_rb = self.x_rb
        y_rb = self.y_rb
        u_rb = self.forward(x_rb, y_rb)
        u_rb_x = self.compute_grad(u_rb, x_rb)
        u_rb_y = self.compute_grad(u_rb, y_rb)
        equation_rb = self.D4*(u_rb_x*self.n_rb_1 + u_rb_y*self.n_rb_2)+1/2*u_rb
        loss_Rb = self.loss_func(equation_rb)


        # 处理界面条件
        # 临界点有连续性
        loss_continuity = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss_continuity.requires_grad_()
        
        # 增加间断点满足方程
        loss_dis = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss_dis.requires_grad_()    
        
        u_15, _, _, _, p_15, _, _ = self.net_model(self.x_15, self.y_15)
        u_15_x = self.compute_grad(u_15, self.x_15)
        u_15_y = self.compute_grad(u_15, self.y_15)
        p_15_x = self.compute_grad(p_15, self.x_15)
        p_15_y = self.compute_grad(p_15, self.y_15)
        
        _, v_25, _, _, p_25, _, _ = self.net_model(self.x_25, self.y_25)
        v_25_x = self.compute_grad(v_25, self.x_25)
        v_25_y = self.compute_grad(v_25, self.y_25)
        p_25_x = self.compute_grad(p_25, self.x_25)
        p_25_y = self.compute_grad(p_25, self.y_25)
        
        _, _, w_35, _, p_35, _, _ = self.net_model(self.x_35, self.y_35)
        w_35_x = self.compute_grad(w_35, self.x_35)
        w_35_y = self.compute_grad(w_35, self.y_35)
        p_35_x = self.compute_grad(p_35, self.x_35)
        p_35_y = self.compute_grad(p_35, self.y_35)
        
        _, _, _, r_45, p_45, _, _ = self.net_model(self.x_45, self.y_45)
        r_45_x = self.compute_grad(r_45, self.x_45)
        r_45_y = self.compute_grad(r_45, self.y_45)
        p_45_x = self.compute_grad(p_45, self.x_45)
        p_45_y = self.compute_grad(p_45, self.y_45)
        
        _, _, _, r_46, _, q_46, _ = self.net_model(self.x_46, self.y_46)
        r_46_x = self.compute_grad(r_46, self.x_46)
        r_46_y = self.compute_grad(r_46, self.y_46)
        q_46_x = self.compute_grad(q_46, self.x_46)
        q_46_y = self.compute_grad(q_46, self.y_46)

        _, _, _, _, p_56, q_56, _ = self.net_model(self.x_56, self.y_56)
        p_56_x = self.compute_grad(p_56, self.x_56)
        p_56_y = self.compute_grad(p_56, self.y_56)
        q_56_x = self.compute_grad(q_56, self.x_56)
        q_56_y = self.compute_grad(q_56, self.y_56)

        _, _, _, _, _, q_67, h_67 = self.net_model(self.x_67, self.y_67)
        q_67_x = self.compute_grad(q_67, self.x_67)
        q_67_y = self.compute_grad(q_67, self.y_67)
        h_67_x = self.compute_grad(h_67, self.x_67)
        h_67_y = self.compute_grad(h_67, self.y_67)

        _, _, _, r_456, p_456, q_456, _ = self.net_model(self.x_456, self.y_456)
        # r_456_x = self.compute_grad(r_456, self.x_456)
        # r_456_y = self.compute_grad(r_456, self.y_456)
        # p_456_x = self.compute_grad(p_456, self.x_456)
        # p_456_y = self.compute_grad(p_456, self.y_456)
        # q_456_x = self.compute_grad(p_456, self.x_456)
        # q_456_y = self.compute_grad(p_456, self.y_456)
        
        loss_continuity = self.loss_func(u_15, p_15) +\
                        self.loss_func(v_25, p_25)+\
                        self.loss_func(w_35, p_35)+\
                        self.loss_func(r_45, p_45)+\
                        self.loss_func(r_46, q_46)+\
                        self.loss_func(p_56, q_56)+\
                        self.loss_func(q_67, h_67)+\
                        self.loss_func(r_456, p_456)+\
                        self.loss_func(r_456, q_456)+\
                        self.loss_func(p_456, q_456)
        
        loss_dis =  self.loss_func(-self.D3 * (u_15_x * self.n_15_1 + u_15_y * self.n_15_2) + self.D2 * (p_15_x * self.n_15_1 + p_15_y * self.n_15_2))+\
                    self.loss_func(-self.D3 * (v_25_x * self.n_25_1 + v_25_y * self.n_25_2) + self.D2 * (p_25_x * self.n_25_1 + p_25_y * self.n_25_2))+\
                    self.loss_func(-self.D3 * (w_35_x * self.n_35_1 + w_35_y * self.n_35_2) + self.D2 * (p_35_x * self.n_35_1 + p_35_y * self.n_35_2))+\
                    self.loss_func(-self.D3 * (r_45_x * self.n_45_1 + r_45_y * self.n_45_2) + self.D2 * (p_45_x * self.n_45_1 + p_45_y * self.n_45_2))+\
                    self.loss_func(-self.D3 * (r_46_x * self.n_46_1 + r_46_y * self.n_46_2) + self.D1 * (q_46_x * self.n_46_1 + q_46_y * self.n_46_2))+\
                    self.loss_func(-self.D2 * (p_56_x * self.n_56_1 + p_56_y * self.n_56_2) + self.D1 * (q_56_x * self.n_56_1 + q_56_y * self.n_56_2))+\
                    self.loss_func(-self.D1 * (q_67_x * self.n_67_1 + q_67_y * self.n_67_2) + self.D4 * (h_67_x * self.n_67_1 + h_67_y * self.n_67_2))
               

        # 权重
        alpha_IPM = 1
        alpha_Rb = 1
        alpha_Nb = 1
        alpha_continuity = 1
        alpha_dis = 1
        self.loss = loss_IPM * alpha_IPM + loss_Nb*alpha_Nb + loss_Rb*alpha_Rb + loss_continuity * alpha_continuity + loss_dis * alpha_dis

        # 反向传播
        self.loss.backward()
        # 运算次数加1
        self.nIter = self.nIter + 1
        # # 区域计算Rayleigh-Quotient
        # Luu = torch.sum(Lu*u)
        # uu = torch.sum(self.v_sigma_f*u**2)
        # lambda_ = self.detach(Luu/uu)
        lambda_ = self.detach(torch.sum(Lu)/torch.sum(self.v_sigma_f*u))
        lambda_ = lambda_.max()
        self.lambda_last = lambda_
        # loss = self.loss_func(Lu - lambda_*u)

        # 更新u^k = u^(k+1)        
        # 归一化
        # norm_u = u/torch.sum(self.v_sigma_f*u)
        # norm_u = u/torch.norm(u, p=2)
        
        norm_u = torch.sign(torch.mean(u))*u
        norm_u = norm_u * (norm_u.shape[0]/torch.sum(norm_u))
        # norm_u = norm_u/torch.norm(norm_u) * np.sqrt(norm_u.shape[0])
        
        # norm_u = u/torch.max(u)
        # norm_u = u/torch.norm(self.v_sigma_f*u, p=2)
        self.u = self.data_loader(self.detach(norm_u), requires_grad=False)
        # 保存模型
        loss = self.detach(self.loss)
        if loss < self.min_loss:
            self.lambda_ = lambda_
            self.keff = 1/self.lambda_
            self.min_loss = loss
            PINNConfig.save(net=self,
                            path=self.root_path + '/' + self.path,
                            name=self.model_name)
        # 打印日志
        loss_remainder = 1
        if np.remainder(self.nIter, loss_remainder) == 0:
            # 打印常规loss
            loss_IPM = self.detach(loss_IPM)
            loss_Nb = self.detach(loss_Nb)
            loss_Rb = self.detach(loss_Rb)
            loss_continuity = self.detach(loss_continuity)
            loss_dis = self.detach(loss_dis)

            abs_keff = np.abs(self.keff_true-self.keff)
            rel_keff = abs_keff/self.keff_true

            log_str = str(self.optimizer_name) + ' Iter ' + str(self.nIter) + ' Loss ' + str(loss) +\
                ' lambda_ ' + str(lambda_) + ' loss_IPM ' + str(loss_IPM) +\
                ' loss_Nb ' + str(loss_Nb) +' loss_Rb ' + str(loss_Rb) +\
                ' loss_continuity ' + str(loss_continuity) + ' loss_dis ' + str(loss_dis) +\
                ' LR ' + str(self.optimizer.state_dict()[
                    'param_groups'][0]['lr']) + ' min_loss ' + str(self.min_loss) + ' keff_true ' + str(self.keff_true) +\
                ' lambda ' + str(self.lambda_) + ' keff ' + str(self.keff) + \
                ' abs_keff '+str(abs_keff)+' rel_keff '+str(rel_keff)

            log(log_str)
            
            # 验证
            with torch.no_grad():
                u_valid = self.u_valid

                u_pred = self.forward(self.x_valid, self.y_valid)
                u_pred = torch.sign(torch.mean(u_pred))*u_pred
                # u_pred = torch.sign(torch.mean(u_pred))*u_pred/torch.norm(u_pred)
                u_pred = u_pred * (u_pred.shape[0]/torch.sum(u_pred))
                # u_pred = u_pred / torch.norm(u_pred) * np.sqrt(u_pred.shape[0])
                u_pred = self.detach(u_pred)
                
                keff = 1/lambda_
                
                L_infinity_keff = np.abs(self.keff_true-keff)
                L_infinity_rel_keff = np.abs(self.keff_true-keff)/np.abs(self.keff_true)
            
                L_infinity_u = np.linalg.norm(u_valid-u_pred, ord=inf)
                L_infinity_rel_u = L_infinity_u/np.linalg.norm(u_valid, ord=inf)
                L_2_rel_u = np.linalg.norm(u_valid-u_pred, ord=2)/np.linalg.norm(u_valid, ord=2)
                
                log_str = 'L_infinity_keff ' + str(L_infinity_keff) + ' L_infinity_rel_keff ' + str(L_infinity_rel_keff)+\
                    ' L_infinity_u ' + str(L_infinity_u) + ' L_infinity_rel_u ' + str(L_infinity_rel_u) + ' L_2_rel_u ' + str(L_2_rel_u)
                log(log_str)

            # 打印耗时
            elapsed = time.time() - self.start_time
            print('Time: %.4fs Per %d Iterators' % (elapsed, loss_remainder))
            logging.info('Time: %.4f s Per %d Iterators' %
                         (elapsed, loss_remainder))
            self.start_time = time.time()
        return self.loss
