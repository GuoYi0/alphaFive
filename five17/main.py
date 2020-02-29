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
from multiprocessing import Manager
import gc

cur_dir = os.path.dirname(__file__)
os.chdir(cur_dir)
PURE_MCST = 1
AI = 2
futures = []

data_lock = Lock()
job_done = Lock()
stack = utils.RandomStack(board_size=config.board_size, length=8000)
e = config.noise_eps


def gen_data(pipe):
    global e
    global stack

    player = Player(config, training=True, pipe=pipe)
    while True:
        game_record = player.run(e)
        value = game_record[-1][2]
        game_length = len(game_record)
        if value == 0.0:
            result = utils.DRAW
        elif game_length % 2 == 1:
            result = utils.BLACK_WIN
        else:
            result = utils.WHITE_WIN
        data_lock.acquire(True)
        r = stack.push(game_record, result)
        data_lock.release()
        if r and job_done.locked():
            # 有可能训练代码太慢了，导致很多制造数据的子进程都结束了，会释放已经释放的进程
            job_done.release()


def main(restore=False):
    global stack
    global data_lock
    global job_done
    global e

    net = model(config.board_size)
    if restore:
        net.restore(config.ckpt_path)
        stack.load()
    with net.graph.as_default():
        total_loss, cross_entropy, value_loss, entropy = net.total_loss, net.cross_entropy_loss, net.value_loss, net.entropy
        lr = tf.get_variable("learning_rate", dtype=tf.float32, initializer=1e-3)
        opt = tf.train.AdamOptimizer(lr).minimize(total_loss)
        net.sess.run(tf.global_variables_initializer())
        tf.summary.scalar("x_entropy_loss", cross_entropy)
        tf.summary.scalar("value_loss", value_loss)
        tf.summary.scalar("total_loss", total_loss)
        tf.summary.scalar("entropy", entropy)
        # log_dir = os.path.join("summary", "log_" + time.strftime("%Y%m%d_%H_%M_%S", time.localtime()))
        log_dir = "E:\\alphaFive\\five17\\summary\\log_20200229_11_47_12"
        journalist = tf.summary.FileWriter(log_dir, flush_secs=10)
        summury_op = tf.summary.merge_all()
    step = 842
    k = (config.final_eps - config.noise_eps) / config.total_step
    executor = ProcessPoolExecutor(max_workers=config.max_processes)  # 定义一个进程池，max_workers是最大进程个数
    # 定义一个通信列表，每个进程给一个管道
    cur_pipes = Manager().list([net.get_pipes(config) for _ in range(config.max_processes)])
    job_done.acquire(True)
    for i in range(config.max_processes):
        executor.submit(gen_data, cur_pipes[i])

    while step < config.total_step:
        e = k * step + config.noise_eps
        net.sess.run(tf.assign(lr, config.get_lr(step)))
        job_done.acquire(True)  # 制造数据的进程每释放一次锁，这里才能继续执行。保证了每产生一条数据就训练一次
        # if stack.is_full():  # 满了再训练太慢了，但是消除了biase， push成功才训练
        for _ in range(4):
            data_lock.acquire(True)
            boards, weights, values, policies = stack.get_data(batch_size=config.batch_size)
            data_lock.release()
            xcro_loss, mse_, entropy_, _, sum_res = net.sess.run(
                [cross_entropy, value_loss, entropy, opt, summury_op],
                feed_dict={net.inputs: boards, net.distrib: policies,
                           net.winner: values, net.weights: weights})
        step += 1
        journalist.add_summary(sum_res, step)
        print(" ")
        print("step: %d, xcross_loss: %0.3f, mse: %0.3f, entropy: %0.3f" % (step, xcro_loss, mse_, entropy_))
        if step % 60 == 0:
            net.saver.save(net.sess, save_path=os.path.join(config.ckpt_path, "alphaFive"), global_step=step)
            data_lock.acquire(True)
            stack.save()
            data_lock.release()
    net.saver.save(net.sess, save_path=os.path.join(config.ckpt_path, "alphaFive"), global_step=step)
    data_lock.acquire(True)
    stack.save()
    data_lock.release()
    executor.shutdown(False)
    net.close()


def next_unused_name(name):
    save_name = name
    iteration = 0
    while os.path.exists(save_name):
        save_name = name + '-' + str(iteration)
        iteration += 1
    return save_name


if __name__ == '__main__':
    main(restore=True)
