# -*- coding: utf-8 -*-
# 乘起来为0.05,在乘以该系数得到0.25,q值可能在0.5浮动
board_size = 9  # 棋盘大小
simulation_per_step = 500  # AlphaGo Zero这里取值为1600
goal = 5  # 五子棋，故为5
batch_size = 512  # AlphaGo Zero这里取值为2048
lr_ = [(900, 1e-3), (3000, 2e-4), (8000, 4e-5), (100000000, 2e-6)]
discount = None  # 折扣系数，None就是没有折扣
ckpt_path = "ckpt"
total_step = 2400
tau_decay_rate = 0.9  # 温度衰减项，越往后温度应该越低
# 噪声项
noise_eps = 0.3
final_eps = 0.1
c_puct = 5.0  # 这个数调大一点，使得探索得更好，每个动作的访问次数分布得越均匀，初始温度就可以调的越低
dirichlet_alpha = 0.5  # 这个值越大，产生的分布越均匀
gamma = 0.9  # 权重衰减因子，只乘在value上面，不乘在policy上面
init_temp = 1.2  # 初始温度  # 温度越大分布越平缓
search_thread = 4  # 等效搜索线程
virtual_loss = 3


def get_lr(step):
    for item in lr_:
        if step < item[0]:
            return item[1]
    return lr_[-1][-1]
