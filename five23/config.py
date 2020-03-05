# -*- coding: utf-8 -*-
board_size = 11  # 棋盘大小
simulation_per_step = 542  # 对于初始棋盘格点数121，每个至少访问2次，花销242,。还剩300次自由访问机会
goal = 5  # 五子棋，故为5
batch_size = 512  # AlphaGo Zero这里取值为2048
# lr_ = [(3000, 1e-3), (8000, 2e-4), (8000, 4e-5), (100000000, 2e-6)]
lr_ = [(6000, 1e-3), (14000, 2e-4), (28000, 4e-5), (100000000, 2e-6)]
discount = None  # 折扣系数，None就是没有折扣
ckpt_path = "ckpt"
total_step = 14000
tau_decay_rate = 0.94  # 温度衰减项，越往后温度应该越低  0.94^8*1.6=1.0，即初始温度为1.6时，在8步以后温度衰减为1.0
# 噪声项
noise_eps = 0.2
final_eps = 0.1
c_puct = 5.0  # 这个数调大一点，使得探索得更好，每个动作的访问次数分布得越均匀，初始温度就可以调的越低
dirichlet_alpha = 0.3  # 这个值越大，产生的分布越均匀，棋盘越大，这个值也相应稍微调大点为好
gamma = 0.92  # 权重衰减因子，越是后面的局面，出现的频率就越低，分配的训练权重越大
init_temp = 1.6  # 初始温度  # 温度越大分布越平缓，这里适当调大，主要是考虑到初始分布太尖锐了
max_processes = 5   # 并行产生数据的进程数


def get_lr(step):
    for item in lr_:
        if step < item[0]:
            return item[1]
    return lr_[-1][-1]
