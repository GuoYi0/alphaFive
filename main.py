# -*- coding: utf-8 -*-
from MCTS import MCTS
import time
import utils
from network import ResNet as model
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import config


def main(game_file_saved_dict="game_record"):
    if not os.path.exists(game_file_saved_dict):
        os.mkdir(game_file_saved_dict)
    net = model(config.board_size)
    stack = utils.RandomStack(board_size=config.board_size, length=10000)
    tree = MCTS(config.board_size, net, simulation_per_step=config.simulation_per_step, goal=config.goal)
    step = 1
    cross_entropy = tf.reduce_mean(
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=net.distrib, logits=net.policy), -1))
    mse = tf.reduce_mean(tf.squared_difference(net.value, net.winner))
    L2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    total_loss = cross_entropy + mse + L2_loss * 1e-6
    lr = tf.get_variable("learning_rate", dtype=tf.float32, initializer=1e-3)
    opt = tf.train.AdamOptimizer(lr).minimize(total_loss)
    net.sess.run(tf.global_variables_initializer())
    tf.summary.scalar("x_entropy", cross_entropy)
    tf.summary.scalar("mse", mse)
    tf.summary.scalar("L2_loss", L2_loss)
    tf.summary.scalar("total_loss", total_loss)
    log_dir = os.path.join("summary", "log_" + time.strftime("%Y%m%d_%H_%M_%S", time.localtime()))
    journalist = tf.summary.FileWriter(log_dir, flush_secs=10)
    summury_op = tf.summary.merge_all()
    sum_res = None
    # while True:
    while step < 800:
        print("")
        time1 = int(time.time())
        game_record, expand_count, steps_count = tree.run(train=True)  # 主游戏收集1条episode
        time2 = int(time.time())
        game_time = time2 - time1
        game_length = len(game_record) - 1
        if game_record[-1]:  # 对了最后一个装的是是否平局
            print("game tied, length:{}, time cost: {}".format(game_length, game_time))
        elif game_length % 2 == 1:
            print("game {}, black win, length:{}, time cost: {}".format(step, game_length, game_time))
        else:
            print("game {}, white win, length:{}, time cost: {}".format(step, game_length, game_time))
        print("game per step, expand: %d, simu_steps:%d" % (int(expand_count), int(steps_count)))
        utils.write_file(
            game_record,
            game_file_saved_dict + "/" + time.strftime("%Y%m%d_%H_%M_%S",
                                                       time.localtime()) + '_game_step:{}.pkl'.format(step))
        train_data = utils.generate_training_data(game_record=game_record, board_size=config.board_size,
                                                  discount=config.discount)
        stack.push(train_data)
        for _ in range(5):
            data, distrib, winner = stack.get_data(batch_size=config.batch_size)
            _, sum_res = net.sess.run([opt, summury_op],
                                      feed_dict={net.inputs: data, net.distrib: distrib, net.winner: winner})
        journalist.add_summary(sum_res, step)

        if step % 60 == 0:
            net.saver.save(net.sess, save_path=os.path.join(config.ckpt_path, "alphaFive"), global_step=step)
            # test_game_record, _, _ = tree.run(train=False)
        step += 1
        print("total time: %ds per step" % (int(time.time()-time1)))


def next_unused_name(name):
    save_name = name
    iteration = 0
    while os.path.exists(save_name):
        save_name = name + '-' + str(iteration)
        iteration += 1
    return save_name


if __name__ == '__main__':
    main()
