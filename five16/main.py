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
future_lock = Lock()
stack = utils.RandomStack(board_size=config.board_size, length=5000)


def recall_fn(future):
    global stack
    global job_done
    global future_lock

    game_record = future.result()
    future_lock.acquire(True)
    futures.remove(future)
    future_lock.release()
    if job_done.locked():
        # 有可能训练代码太慢了，导致很多制造数据的子进程都结束了，会释放已经释放的进程
        job_done.release()  # 移走一个future以后，job_done就可以释放出来了，以让训练进程训练，并再次添加一个制造数据的进程
    value = game_record[-1][2]
    game_length = len(game_record)
    if value == 0.0:
        result = utils.DRAW
    elif game_length % 2 == 1:
        result = utils.BLACK_WIN
    else:
        result = utils.WHITE_WIN
    data_lock.acquire(True)
    stack.push(game_record, result)
    data_lock.release()


def gen_data(pipes, e):
    pipe = pipes.pop()
    player = Player(config, training=True, pipe=pipe)
    data = player.run(e)
    player.close()
    del player
    gc.collect()
    pipes.append(pipe)  # pipes不需要锁，因为使用之前已经pop()出来了，列表里面不存在这个pipe了，不会再次分配给别人了
    # 就是说不存在同一个pipe分配给多个player的现象
    return data


def main(restore=False):
    global futures
    global stack
    global data_lock
    global job_done

    net = model(config.board_size)
    if restore:
        net.restore(config.ckpt_path)
        # stack.load()
    with net.graph.as_default():
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
    step = 1
    k = (config.final_eps - config.noise_eps) / config.total_step
    executor = ProcessPoolExecutor(max_workers=config.max_processes)  # 定义一个进程池，max_workers是最大进程个数
    job_done.acquire(True)  # 这句话很重要，保证了进程池里面最多只有max_processes个进程以及pipe的正确分配
    # 定义一个通信列表，每个进程给一个管道
    cur_pipes = Manager().list([net.get_pipes(config) for _ in range(config.max_processes)])
    while step < config.total_step:
        e = k * step + config.noise_eps
        net.sess.run(tf.assign(lr, config.get_lr(step)))
        # if stack.is_full():  # 满了再训练太慢了，但是消除了biase， push成功才训练
        if stack.is_full():  # 满了再训练太慢了，但是消除了biase， push成功才训练
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
        while len(futures) < config.max_processes:
            # 这个while循环一般只会进来一次，只有当训练得太慢了，制作数据很快的时候，才会多次进来
            print("entering the while loop...")
            ff = executor.submit(gen_data, cur_pipes, e)
            ff.add_done_callback(recall_fn)
            future_lock.acquire(True)
            futures.append(ff)
            future_lock.release()
        job_done.acquire(True)  # 有进程结束以后，才会解锁job_done，从而通过这一步继续往下走，NB了
        ff = executor.submit(gen_data, cur_pipes, e)
        ff.add_done_callback(recall_fn)
        future_lock.acquire(True)
        futures.append(ff)
        future_lock.release()

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
    main(restore=False)
