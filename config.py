# -*- coding: utf-8 -*-
Cucb = 0.1  # UCB第二项的系数
board_size = 6  # 棋盘大小
goal = 3
batch_size = 64
lr = [(20, 1e-4), (100, 1e-3), (200, 1e-4), (300, 1e-5), (100000000, 1e-6)]


def get_lr(step):
    for item in lr:
        if step < item[0]:
            return item[1]
    return lr[-1][-1]
