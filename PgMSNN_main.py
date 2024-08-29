"""
@author: Wenxuan Yuan
Email: wenxuan.yuan@qq.com
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio
import os

import PgMSNN

from domain_config import D1, D2, D3, D4, D5, D6, D7, D8

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_default_dtype(torch.float32)


def gen_C(x_sample_num,  t_sample_num):
    ascending_C = np.tile(np.linspace(1, 2, x_sample_num), (t_sample_num, 1))
    C = nn.Parameter(torch.tensor(ascending_C, dtype=torch.float32), requires_grad=True)
    return C


def gen_evolution_cover(parameters, range_x, x_sample_num,  t_sample_num):
    d, k_1, k_2, u_negative, u_positive, end_t = parameters
    v_negative = d - u_negative * (k_2 * u_negative + k_1) + 2 * u_positive * (k_2 * u_positive + k_1)
    v_positive = d + 2/3 * u_negative * (k_2 * u_negative + k_1) + 1/3 * u_positive * (k_2 * u_positive + k_1)

    X = np.linspace(range_x[0], range_x[1], x_sample_num)
    T = np.linspace(0, end_t, t_sample_num)

    ones_C_ = []
    for inx_t in range(t_sample_num):
        step_container = []
        for inx_x in range(x_sample_num):
            if v_negative * T[inx_t] <= X[inx_x] <= v_positive * T[inx_t]:
                step_container.append(torch.tensor(1, dtype=torch.float32, requires_grad=True))
            else:
                step_container.append(torch.tensor(0, dtype=torch.float32, requires_grad=True))

        ones_C_i = torch.tensor(step_container).reshape(1, -1)
        ones_C_.append(ones_C_i)
    EC = torch.cat(ones_C_, dim=0)

    return EC


def gen_Jump_initial_value_cover(parameters, range_x, x_sample_num,  t_sample_num):
    d, k_1, k_2, u_negative, u_positive, end_t = parameters
    v_negative = d - u_negative * (k_2 * u_negative + k_1) + 2 * u_positive * (k_2 * u_positive + k_1)
    v_positive = d + 2/3 * u_negative * (k_2 * u_negative + k_1) + 1/3 * u_positive * (k_2 * u_positive + k_1)

    X = np.linspace(range_x[0], range_x[1], x_sample_num)
    T = np.linspace(0, end_t, t_sample_num)

    JIVC_ = []
    for inx_t in range(t_sample_num):
        step_container = []
        for inx_x in range(x_sample_num):
            if X[inx_x] < v_negative * T[inx_t]:
                step_container.append(torch.tensor(u_negative, dtype=torch.float32, requires_grad=False))
            elif v_positive * T[inx_t] < X[inx_x]:
                step_container.append(torch.tensor(u_positive, dtype=torch.float32, requires_grad=False))
            else:
                step_container.append(torch.tensor(0, dtype=torch.float32, requires_grad=False))

        JIVC_i = torch.tensor(step_container).reshape(1, -1)
        JIVC_.append(JIVC_i)
    JIVC = torch.cat(JIVC_, dim=0)

    return JIVC


def add_noise(exact, pec=0.05):
    from torch.distributions import normal
    torch.manual_seed(66)

    n_distr = normal.Normal(0.0, 1.0)
    R = n_distr.sample(exact.shape).cuda()
    std_R = torch.std(R)  # std of samples
    std_T = torch.std(exact)
    noise = R * std_T / std_R * pec
    noised_exact = exact + noise

    return noised_exact


if __name__ == '__main__':
    torch.manual_seed(99)

    Domain_list = [D1, D2, D3, D4, D5, D6, D7, D8]
    domain_order = 5

    # load training data
    train_data = sio.loadmat('data/domain_{}.mat'.format(str(domain_order)))
    train_data = train_data['train_data'].astype(float)
    train_data = torch.tensor(train_data, dtype=torch.float32)

    # load exact data
    domain_data = sio.loadmat('data/domain_{}_exact.mat'.format(str(domain_order)))
    exact = domain_data['exact'].astype(float)
    Y_ = exact[-1, :]
    exact = torch.tensor(exact, dtype=torch.float32).cuda()

    # define the model hyperparameters
    MLP_layer_structure = [2, 50, 50, 50, 50, 50, 50, 50, 1]

    x_sample_num = 1000  # Number of samples in the spatial domain
    t_sample_num = 1000  # Number of samples in the time domain

    # Multi-stage training strategy configuration
    # (stage1 loss, stage1 maxit, stage2 loss, stage2 maxit, stage3 loss, stage3 maxit)
    train_config = (0.006, 30000, 0.00085, 880, 0.005, 30000)

    D = Domain_list[domain_order - 1]()
    parameter = (D.d, D.k_1, D.k_2, D.u_negative, D.u_positive, D.t)
    C = gen_C(x_sample_num, t_sample_num)
    EC = gen_evolution_cover(parameter, D.x_boundary, x_sample_num, t_sample_num)  # Deprecated scheme, but not removed for code structural integrity
    JIVC = gen_Jump_initial_value_cover(parameter, D.x_boundary, x_sample_num, t_sample_num)  # Deprecated scheme, but not removed for code structural integrity
    Cs = (C.cuda(), EC.cuda(), JIVC.cuda())

    time_steps = 900  # The number of copies of the exact data used for training
    num_features = 31  # Order of the deep Runge-Kutta method
    dt = D.t / t_sample_num

    DRKT_parameter = (parameter, num_features, dt)  # Parameters used to initialize DRKT

    train_x = train_data[:, :, 0:1].requires_grad_(True).cuda()
    train_t = train_data[:, :, -1:].requires_grad_(True).cuda()

    drhtpinn = PgMSNN.DRKTPINN(train_config, exact[:time_steps, :], time_steps, Cs, MLP_layer_structure, DRKT_parameter, OO0000OOO0000O000=False)

    with torch.autograd.set_detect_anomaly(True):
        loss_list, Iter_list, loss2num, Iter2, predict = drhtpinn(train_x, train_t)

    lossfn = nn.MSELoss()
    lossfinal = torch.sqrt(lossfn(predict, exact))
    print('Average Loss is : {}'.format(lossfinal.item()))
    print('Phase 2 iterates {} times'.format(Iter2))

    Figure_save_path = 'figures/'

    # PgMSNN Prediction Heatmap
    u = predict.cpu().detach().numpy()
    plt.figure(num=2, figsize=(10, 10), dpi=160)
    plt.imshow(u, interpolation='nearest', cmap='YlGnBu', origin='lower', aspect='auto')
    plt.colorbar(shrink=0.70)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel("x Axis")
    plt.ylabel("t Axis")
    plt.title("PgMSNN Prediction")
    plt.savefig(Figure_save_path + "PgMSNN Prediction heatmap.png")
    plt.show()
