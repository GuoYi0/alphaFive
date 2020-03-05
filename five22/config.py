# -*- coding: utf-8 -*-
# 乘起来为0.05,在乘以该系数得到0.25,q值可能在0.5浮动
board_size = 9  # 棋盘大小
simulation_per_step = 500  # AlphaGo Zero这里取值为1600
goal = 5  # 五子棋，故为5
batch_size = 512  # AlphaGo Zero这里取值为2048
# lr_ = [(3000, 1e-3), (8000, 2e-4), (8000, 4e-5), (100000000, 2e-6)]
lr_ = [(3000, 1e-3), (8000, 2e-4), (18000, 4e-5), (100000000, 2e-6)]
discount = None  # 折扣系数，None就是没有折扣
ckpt_path = "ckpt"
total_step = 14000
tau_decay_rate = 0.94  # 温度衰减项，越往后温度应该越低  0.94^8=1
# 噪声项
noise_eps = 0.2
final_eps = 0.1
c_puct = 5.0  # 这个数调大一点，使得探索得更好，每个动作的访问次数分布得越均匀，初始温度就可以调的越低
dirichlet_alpha = 0.5  # 这个值越大，产生的分布越均匀
gamma = 0.9  # 权重衰减因子，只乘在value上面，不乘在policy上面
init_temp = 1.6  # 初始温度  # 温度越大分布越平缓，这里适当调大，主要是考虑到初始分布太尖锐了
max_processes = 5   # 并行产生数据的进程数


def get_lr(step):
    for item in lr_:
        if step < item[0]:
            return item[1]
    return lr_[-1][-1]
