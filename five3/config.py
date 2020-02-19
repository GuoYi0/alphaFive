# -*- coding: utf-8 -*-
Cucb = 5  # simulation_per_step取400， 棋盘大小大致为40，平均每个结点访问10次, sqrt(400)/10=2，先验概率平均为1/40
# 乘起来为0.05,在乘以该系数得到0.25,q值可能在0.5浮动
board_size = 7  # 棋盘大小
simulation_per_step = 400  # AlphaGo Zero这里取值为1600
goal = 4  # 先学三子棋
batch_size = 256  # AlphaGo Zero这里取值为2048
lr_ = [(100, 5e-5), (200, 5e-4), (1000, 5e-3), (2000, 5e-4), (3000, 5e-5), (100000000, 5e-6)]
discount = None  # 折扣系数，None就是没有折扣
ckpt_path = "ckpt"
temperature = 0.5
total_step = 8000  # AlphaGo Zero这里取值为490万
tau_decay_rate = 0.9  # 温度衰减项
# 噪声项
noise_eps = 0.25
c_puct = 1.5
dirichlet_alpha = 2  # 这个值越大，产生的分布越均匀
gamma = 0.95  # 权重衰减因子
def get_lr(step):
    for item in lr_:
        if step < item[0]:
            return item[1]
    return lr_[-1][-1]
