# -*- coding: utf-8 -*-
from threading import Thread
from multiprocessing import connection, Pipe
from logging import getLogger
import numpy as np

logger = getLogger(__name__)


class NetworkAPI(object):
    def __init__(self, cfg=None, agent_model=None):
        self.agent_model = agent_model
        self.config = cfg
        self.pipes = []  # 用于进程/线程之间通信
        self.reload = True
        self.prediction_worker = None
        self.done = False

    def start(self, reload):
        """
        开启一个线程，来预测数据
        :param reload:
        :return:
        """
        self.reload = reload
        # 这里貌似只能给线程，不能给进程，
        self.prediction_worker = Thread(target=self.predict_batch_worker, name="prediction_worker")
        # prediction_worker = Process(target=self.predict_batch_worker, name="prediction_worker")
        self.prediction_worker.daemon = True  # 守护线程，不必等其结束即可结束主线程
        self.prediction_worker.start()  # 开启

    def get_pipe(self, reload=True):
        """
        定义一个管道，自己得一端，返回另一端
        :param reload:
        :return:
        """
        me, you = Pipe()  # 通信管道的两端
        self.pipes.append(me)
        self.reload = reload
        return you

    def predict_batch_worker(self):
        """
        把各个有用管道收集来的数据集中eval，再在原来的管道发送出去
        :return:
        """
        while not self.done:
            ready = connection.wait(self.pipes, timeout=0.001)  # 等待有通信管道可用，返回可用的管道
            if not ready:
                continue
            data, result_pipes, data_len = [], [], []
            for pipe in ready:
                while pipe.poll():  # 不停地返回false，直到连接到有用数据，就返回true
                    try:
                        tmp = pipe.recv()  # 如果没有消息可接收，recv方法会一直阻塞。如果连接的另外一端已经关闭，那么recv方法会抛出EOFError。
                    except EOFError as e:
                        logger.error(f"EOF error: {e}")
                        pipe.close()  # 另一端关闭，这端就关闭了
                    else:
                        data.extend(tmp)
                        data_len.append(len(tmp))
                        result_pipes.append(pipe)
            if not data:
                continue
            data = np.asarray(data, dtype=np.float32)
            with self.agent_model.graph.as_default():
                policy, value = self.agent_model.eval(data)
            buf = []
            k, i = 0, 0
            for p, v in zip(policy, value):
                buf.append((p, float(v)))
                k += 1
                if k >= data_len[i]:
                    result_pipes[i].send(buf)
                    buf = []
                    k = 0
                    i += 1

    def close(self):
        self.done = True
        for pipe in self.pipes:
            pipe.close()
