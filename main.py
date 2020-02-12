# -*- coding: utf-8 -*-
from MCTS import MCTS
import time
import utils
from network import ResNet as model
import os
import matplotlib.pyplot as plt
import config
BOARD_SIZE = 8  # 棋盘大小


def main():
    net = model(config.board_size)
    tree = MCTS(config.board_size, net, simulation_per_step=400)
    while True:
        game_record, eval, steps = tree.game()


if __name__ == '__main__':
    main()
