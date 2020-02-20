# -*- coding: utf-8 -*-
import time
import utils
from network import ResNet as model
import tensorflow as tf
import os
import config
from player import Player
import numpy as np
import sys

cur_dir = os.path.dirname(__file__)
os.chdir(cur_dir)

PURE_MCST = 1
AI = 2


def main(game_file_saved_dict="game_record", restore=False):
    if not os.path.exists(game_file_saved_dict):
        os.mkdir(game_file_saved_dict)
    net = model(config.board_size)
    stack = utils.RandomStack(board_size=config.board_size, length=10000)
    player = Player(config, training=True, pv_fn=net.eval)
    # tree = MCTS(config.board_size, net, simulation_per_step=config.simulation_per_step, goal=config.goal)
    step = 720
    total_loss, cross_entropy, value_loss, entropy = net.total_loss, net.cross_entropy_loss, net.value_loss, net.entropy
    lr = tf.get_variable("learning_rate", dtype=tf.float32, initializer=1e-3)
    opt = tf.train.AdamOptimizer(lr).minimize(total_loss)
    net.sess.run(tf.global_variables_initializer())
    tf.summary.scalar("x_entropy_loss", cross_entropy)
    tf.summary.scalar("value_loss", value_loss)
    tf.summary.scalar("total_loss", total_loss)
    tf.summary.scalar("entropy", entropy)
    log_dir = os.path.join("summary", "log_" + time.strftime("%Y%m%d_%H_%M_%S", time.localtime()))
    journalist = tf.summary.FileWriter(log_dir, flush_secs=10)
    summury_op = tf.summary.merge_all()
    sum_res = None
    if restore:
        net.restore(config.ckpt_path)
    while step < config.total_step:
        net.sess.run(tf.assign(lr, config.get_lr(step)))
        print("")
        time1 = int(time.time())
        game_record = player.run()  # 主游戏收集1条episode
        time2 = int(time.time())
        game_time = time2 - time1
        game_length = len(game_record)
        value = game_record[-1][2]
        if value == 0.0:
            print("game tied, length:{}, time cost: {}".format(game_length, game_time))
        elif game_length % 2 == 1:
            print("game {}, black win, length:{}, time cost: {}".format(step, game_length, game_time))
        else:
            print("game {}, white win, length:{}, time cost: {}".format(step, game_length, game_time))
        stack.push(game_record)
        for _ in range(5):
            boards, weights, values, policies = stack.get_data(batch_size=config.batch_size)
            xcro_loss, mse_, entropy_, _, sum_res = net.sess.run([cross_entropy, value_loss, entropy, opt, summury_op],
                                                                 feed_dict={net.inputs: boards, net.distrib: policies,
                                                                            net.winner: values, net.weights: weights,
                                                                            net.training: True})
        journalist.add_summary(sum_res, step)
        print("xcross_loss: %0.3f, mse: %0.3f, entropy: %0.3f" % (xcro_loss, mse_, entropy_))
        if step % 60 == 0:
            net.saver.save(net.sess, save_path=os.path.join(config.ckpt_path, "alphaFive"), global_step=step)
            # evaluate(player)
            # player.training = True
        step += 1
        print("total time: %ds per step" % (int(time.time() - time1)))


def next_unused_name(name):
    save_name = name
    iteration = 0
    while os.path.exists(save_name):
        save_name = name + '-' + str(iteration)
        iteration += 1
    return save_name


def evaluate(player1, player2=None, ngames=10):
    player1.training = False
    if player2 is None:
        player2 = Player(config, training=False, use_net=False)
    wins = np.zeros((2,), dtype=np.int32)  # 前面是AI，后面是纯MCST
    players = [player1, player2]
    for i in range(ngames):  # 玩这么多局游戏
        k = i % 2
        player1.reset()
        player2.reset()
        game_over = False
        state = player1.get_init_state()
        value = 0.0
        while not game_over:
            _, action = players[k].get_action(state)
            board = utils.step(utils.state_to_board(state, config.board_size), action)
            state = utils.board_to_state(board)
            players[k].pruning_tree(board, state)  # 走完一步以后，对其他分支进行剪枝，以节约内存
            game_over, value = utils.is_game_over(board, config.goal)
            k = (k + 1) % 2
        if value == 0.0:
            continue
        else:
            wins[(k + 1) % 2] += 1
    print("{}:{}".format(wins[0], wins[1]))


if __name__ == '__main__':
    main(restore=True)
