# -*- coding: utf-8 -*-
from genData.network import ResNet as Model
import config
import pygame
import os
import tensorflow as tf
import numpy as np
from genData.player import Player
import utils
from random import shuffle
import random


GRID_WIDTH = 36
WIDTH = (config.board_size + 2) * GRID_WIDTH
HEIGHT = (config.board_size + 2) * GRID_WIDTH
FPS = 300
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HUMAN = 0
AI = 2


def main():
    config.simulation_per_step = 500
    # 先只搜索6060以上的
    all_ckpts = [os.path.join("ckpt", "alphaFive-"+str(num)) for num in range(60, 8800, 60)][100:-1]
    net0 = Model(config.board_size, tf.Graph())
    net0.restore(all_ckpts[0])
    net1 = Model(config.board_size, tf.Graph())
    net1.restore(all_ckpts[-1])
    player0 = Player(config, training=False, pv_fn=net0.eval)
    player1 = Player(config, training=False, pv_fn=net1.eval)
    players = [{'p': player0, "win": 0, "ckpt": all_ckpts[0]},
               {'p': player1, "win": 0, "ckpt": all_ckpts[-1]}]
    result = open("result.txt", "a")
    low, high = 0, len(all_ckpts)-1
    while low < high:  # 尽量让实力悬殊的ckpt进行对弈
        print("")
        print("==================================================================")
        print(players[0]["ckpt"] + " vs " + players[1]["ckpt"] + '...')
        for i in range(100):  # 最多对弈100局
            players[0]['p'].reset()  # 每一局开始前都要重置
            players[1]['p'].reset()
            game_over = False
            action = None
            state = player1.get_init_state()
            current_ids = i % 2
            value = 0.0
            count = 0
            while not game_over:
                _, action = players[current_ids]['p'].get_action(state, last_action=action, random_a=True)
                board = utils.step(utils.state_to_board(state, config.board_size), action)
                state = utils.board_to_state(board)
                # players[current_ids].pruning_tree(board, state)  # 走完一步以后，对其他分支进行剪枝，以节约内存, 不剪枝，节约时间
                game_over, value = utils.is_game_over(board, config.goal)
                current_ids = (current_ids + 1) % 2  # 下一个选手
                count += 1
            if value == 0.0:  # 和棋了
                print(f"game: {i}, tied! all {count} turns.")
                continue
            else:
                print(f"game: {i} {players[(current_ids+1) % 2]['ckpt']}  won! all {count} turns.")
                players[(current_ids+1) % 2]["win"] += 1
            if i >= 30:
                # 超过24局以后，输赢悬殊太大的话，直接break
                w0 = players[0]["win"]
                w1 = players[1]["win"]
                if w0 == 0 or w1 == 0:
                    break
                elif w0 / w1 > 2.0 or w0 / w1 < 0.5:
                    break
        print_str = players[0]["ckpt"] + ": " + players[1]["ckpt"] + f' = {players[0]["win"]}: {players[1]["win"]}'
        print(print_str)
        print(print_str, file=result, flush=True)
        if players[0]["win"] < players[1]["win"]:
            low += 1
            net0.restore(all_ckpts[low])
            players[0]["ckpt"] = all_ckpts[low]
        else:
            high -= 1
            net1.restore(all_ckpts[high])
            players[1]["ckpt"] = all_ckpts[high]

        players[0]["win"] = players[1]["win"] = 0
    result.close()
    net1.close()
    net0.close()


if __name__ == "__main__":
    main()

