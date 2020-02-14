# -*- coding: utf-8 -*-
Cucb = 5  # UCB第二项的系数,棋盘越大，这个值最好就越大
board_size = 5  # 棋盘大小
simulation_per_step = 100
goal = 3  # 先学三子棋
batch_size = 128
lr = [(20, 1e-4), (100, 1e-3), (200, 1e-4), (300, 1e-5), (100000000, 1e-6)]
discount = None  # 折扣系数，None就是没有折扣
ckpt_path = "ckpt"

def get_lr(step):
    for item in lr:
        if step < item[0]:
            return item[1]
    return lr[-1][-1]
