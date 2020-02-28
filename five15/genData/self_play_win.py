# -*- coding: utf-8 -*-
import config
from genData.player import Player
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, ThreadPoolExecutor
import gc


class SelfPlayWorker(object):
    def __init__(self, cfg, net, training=True):
        self.config = config
        self.model = net
        # self.player = Player
        self.m = Manager()  # 进程之间的通信管理器
        self.training = training
        # 定义几个通信管道
        self.cur_pipes = self.m.list([self.model.get_pipes(self.config) for _ in range(config.max_processes)])
        self.executor = ProcessPoolExecutor(max_workers=self.config.max_processes)

    def start(self, e, training=True):
        # executor = ProcessPoolExecutor(max_workers=self.config.max_processes)
        procs = []
        for i in range(self.config.max_processes):
            pipe = self.cur_pipes[i]
            ff = self.executor.submit(gen_data, pipe, e)   # 只能序列化顶层函数，故gen_data单独写成一个函数
            procs.append(ff)
        wait(procs, return_when=ALL_COMPLETED)  # 等待所有进程结束
        results = []
        for p in procs:
            results.append(p.result())
        # executor.shutdown()
        return results

    def close(self):
        for pipe in self.cur_pipes:
            pipe.close()
        self.executor.shutdown(False)


def gen_data(pipe, e):
    player = Player(config, training=True, pipe=pipe)
    data = player.run(e)
    player.close()
    del player
    gc.collect()
    return data
