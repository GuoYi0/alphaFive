# -*- coding: utf-8 -*-
import time
import utils
from genData.network import ResNet as model
import tensorflow as tf
import os
import config
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from threading import Lock, Thread
from genData.player import Player
from multiprocessing import Manager, Process, Queue
# from queue import Queue  #  这个只能在多线程之间使用，即同一个进程内部使用，不能跨进程通信
from multiprocessing.managers import BaseManager
from utils import RandomStack
import gc

# global只能在父子进程中共享只读变量

cur_dir = os.path.dirname(__file__)
os.chdir(cur_dir)
PURE_MCST = 1
AI = 2


# job_lock = Lock()


def main(restore=False):
    stack = RandomStack(board_size=config.board_size, length=config.buffer_size)  # 命名为stack了，事实上是一个队列
    net = model(config.board_size)
    if restore:
        net.restore(config.ckpt_path)
        stack.load(6960)  # 这里需要根据实际情况更改，看从哪一步接着训练，就写为哪一步
    with net.graph.as_default():
        episode_length = tf.placeholder(tf.float32, (), "episode_length")
        total_loss, cross_entropy, value_loss, entropy = net.total_loss, net.cross_entropy_loss, net.value_loss, net.entropy
        lr = tf.get_variable("learning_rate", dtype=tf.float32, initializer=1e-3)
        opt = tf.train.AdamOptimizer(lr).minimize(total_loss)
        net.sess.run(tf.global_variables_initializer())
        tf.summary.scalar("x_entropy_loss", cross_entropy)
        tf.summary.scalar("value_loss", value_loss)
        tf.summary.scalar("total_loss", total_loss)
        tf.summary.scalar("entropy", entropy)
        tf.summary.scalar('episode_len', episode_length)
        log_dir = os.path.join("summary", "log_" + time.strftime("%Y%m%d_%H_%M_%S", time.localtime()))
        journalist = tf.summary.FileWriter(log_dir, flush_secs=10)
        summury_op = tf.summary.merge_all()
    step = 1  # 如果接着训练，这里就改为接着的那一步的下一步。手动改了算了，懒得写成自动识别的了
    cur_pipes = [net.get_pipes(config) for _ in range(config.max_processes)]  # 手动创建进程不需要Manager()
    q = Queue(50)  # 用Process手动创建的进程可以使用这个Queue，否则需要Manager()来管理
    for i in range(config.max_processes):
        proc = Process(target=gen_data, args=(cur_pipes[i], q))
        proc.daemon = True  # 父进程结束以后，子进程就自动结束
        proc.start()

    while step < config.total_step:
        # 每生成一条数据，才训练一次
        net.sess.run(tf.assign(lr, config.get_lr(step)))
        data_record, result = q.get(block=True)  # 获取一个item，没有则阻塞
        r = stack.push(data_record, result)
        if r and stack.is_full():  # 满了再训练会比较慢，但是消除了biase
            for _ in range(4):
                boards, weights, values, policies = stack.get_data(batch_size=config.batch_size)
                xcro_loss, mse_, entropy_, _, sum_res = net.sess.run(
                    [cross_entropy, value_loss, entropy, opt, summury_op],
                    feed_dict={net.inputs: boards, net.distrib: policies,
                               net.winner: values, net.weights: weights, episode_length: len(data_record)})
            step += 1
            journalist.add_summary(sum_res, step)
            print(" ")
            print("step: %d, xcross_loss: %0.3f, mse: %0.3f, entropy: %0.3f" % (step, xcro_loss, mse_, entropy_))
            if step % 60 == 0:
                net.saver.save(net.sess, save_path=os.path.join(config.ckpt_path, "alphaFive"), global_step=step)
                stack.save(step)
                print("save ckpt and data successfully")
    net.saver.save(net.sess, save_path=os.path.join(config.ckpt_path, "alphaFive"), global_step=step)
    stack.save()
    net.close()


def gen_data(pipe, q):
    player = Player(config, training=True, pipe=pipe)
    while True:
        game_record = player.run()
        value = game_record[-1][-2]
        game_length = len(game_record)
        if value == 0.0:
            result = utils.DRAW
        elif game_length % 2 == 1:
            result = utils.BLACK_WIN
        else:
            result = utils.WHITE_WIN
        q.put((game_record, result), block=True)  # block=True满了则阻塞


def next_unused_name(name):
    save_name = name
    iteration = 0
    while os.path.exists(save_name):
        save_name = name + '-' + str(iteration)
        iteration += 1
    return save_name


if __name__ == '__main__':
    main(restore=False)
