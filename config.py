# -*- coding: utf-8 -*-
Cucb = 5  # UCB第二项的系数,棋盘越大，这个值最好就越大
board_size = 13  # 棋盘大小
simulation_per_step = 400
goal = 5  # 先学三子棋
batch_size = 256
lr_ = [(200, 5e-4), (1000, 5e-3), (2000, 5e-4), (3000, 5e-5), (100000000, 5e-6)]
discount = None  # 折扣系数，None就是没有折扣
ckpt_path = "ckpt"
temperature = 0.5
total_step = 8000


def get_lr(step):
    for item in lr_:
        if step < item[0]:
            return item[1]
    return lr_[-1][-1]
