# -*- coding: utf-8 -*-
from MCTS import MCTS
import time
import utils
from network import ResNet as model
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import config

BOARD_SIZE = 8  # 棋盘大小


def main(game_file_saved_dict="game_record"):
    if not os.path.exists(game_file_saved_dict):
        os.mkdir(game_file_saved_dict)
    net = model(config.board_size)
    stack = utils.RandomStack()
    tree = MCTS(config.board_size, net, simulation_per_step=4)
    game_time = 1
    cross_entropy = tf.reduce_mean(
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=net.distrib, logits=net.policy), -1))
    mse = tf.squared_difference(net.value, net.winner)
    L2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    total_loss = cross_entropy + mse + L2_loss * 1e-6
    lr = tf.get_variable("learning_rate", dtype=tf.float32, initializer=1e-3)
    opt = tf.train.AdamOptimizer(lr).minimize(total_loss)
    tf.summary.scalar("x_entropy", cross_entropy)
    tf.summary.scalar("mse", mse)
    tf.summary.scalar("L2_loss", L2_loss)
    tf.summary.scalar("total_loss", total_loss)
    log_dir = os.path.join("summary", "log_" + time.strftime("%Y%m%d_%H_%M_%S", time.localtime()))
    journalist = tf.summary.FileWriter(log_dir, flush_secs=10)
    summury_op = tf.summary.merge_all()
    sum_res = None
    while True:
        game_record, expand, steps = tree.run(train=True)
        if game_record[-1]:  # 对了最后一个装的是是否平局
            print("game is a draw, this game length is {}".format(len(game_record) - 1))
        elif len(game_record) % 2 == 0:
            print("game {} completed, black win, this game length is {}".format(game_time, len(game_record) - 1))
        else:
            print("game {} completed, white win, this game length is {}".format(game_time, len(game_record) - 1))
        print("The average eval:{}, the average steps:{}".format(expand, steps))
        utils.write_file(
            game_record,
            game_file_saved_dict + "/" + time.strftime("%Y%m%d_%H_%M_%S",
                                                       time.localtime()) + '_game_time:{}.pkl'.format(game_time))
        train_data = utils.generate_training_data(game_record=game_record, board_size=config.board_size)
        for i in range(len(train_data)):
            stack.push(train_data[i])
        for _ in range(5):
            data, distrib, winner = stack.get_data(batch_size=config.batch_size)
            _, sum_res = net.sess.run([opt, summury_op],
                                      feed_dict={net.inputs: data, net.distrib: distrib, net.winner: winner})
        journalist.add_summary(sum_res)
        if game_time % 200 == 0:
            net.saver.save(net.sess, save_path=os.path.join("ckpt", "alphaFive"), global_step=game_time)
            test_game_record, _, _ = tree.run(train=False)


def next_unused_name(name):
    save_name = name
    iteration = 0
    while os.path.exists(save_name):
        save_name = name + '-' + str(iteration)
        iteration += 1
    return save_name


if __name__ == '__main__':
    main()
