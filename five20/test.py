# -*- coding: utf-8 -*-
import numpy as np
import sys

# #
# # def get_init_state():
# #     """
# #     用一个字符串表示棋盘，从上至下从左至由编码
# #     黑子用3白字用1表示，空格部分用小写字母表示，a表示一个连续空格，b表示两个连续空格，以此类推
# #     :return:
# #     """
# #     fen = ""
# #     for i in range(4):
# #         fen += chr(ord("a") + i) + '/'
# #     return fen
# #
# # print('a'.isalpha())
# # print(ord('c')-ord('a'))
#
# a = np.array([[1,0], [0,0]])
# print(np.equal(a, 0).astype(np.float32))
# # print(np.where(a==0))
# # def get_legal_moves(board):
# #     zeros = np.where(board == 0)
# #     return [(int(i), int(j)) for i, j in zip(*zeros)]
# #
# # print(get_legal_moves(a))
#
# def board_to_state(board: np.ndarray) -> str:
#     fen = ""
#     h, w = board.shape
#     for i in range(h):
#         c = 0
#         for j in range(w):
#             if board[i, j] == 0:
#                 c += 1
#             else:
#                 fen += chr(ord('a')+c) if c > 0 else ''
#                 fen += str(board[i, j] + 2)
#                 c = 0
#         fen += chr(ord('a') + c) if c > 0 else ''
#         fen += '/'
#     return fen
#
# def state_to_board(state:str, board_size:int):
#     """
#     根据字符串表示的state转换为棋盘。字符串中，黑子用1表示，红子用3表示
#     :param state:
#     :param board_size:
#     :return:
#     """
#     board = np.zeros((board_size, board_size), np.int8)
#     i = j = 0
#     for ch in state:
#         if ch == '/':
#             i += 1
#             j = 0
#         elif ch.isalpha():
#             j += ord(ch) - ord('a')
#         else:
#             board[i][j] = int(ch) - 2
#             j += 1
#     return board
#
#
# b = np.array([[1,0], [-1, 1]])
# # print(board_to_state(b))
# #
# # print(state_to_board(board_to_state(b), 2))
#
# index = [(0,0), (1,1)]
# g = list(zip(*index))
# b[g[0], g[1]] = -1
# print(b)
# b = [1,2,3,4,5,6,]
# print(b[-3:])
# print(np.random.dirichlet(2 * np.ones(50)))
# a = np.array([0,2,3,0,2])
# a = 0.28437745
# print("a {}, %.3f, ".format((1,2))%a)
# a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
#
#
# def fenmu():
#     print("haha")
#     return 2.0
#
#
# a /= fenmu()
#
# # print(a)
# a = [1,2,3,1,2,3]
# a.remove(1)
# # b = a.index(1)
# print(a)
# from logging import getLogger
# logger = getLogger(__name__)
# logger.info("haha")
# g = np.random.dirichlet(0.5 * np.ones(20))
# print(g)
# import time
#
# try:
#     for i in range(10):
#         print(i)
#         time.sleep(5)
# except KeyboardInterrupt:
#     print("over")
from concurrent.futures import ThreadPoolExecutor
import time
# from multiprocessing import Process,Pipe
# # 导入进程，管道模块
#
# def f(conn):
#     conn.send([1,'test',None])
#     conn.send([2,'test',None])
#     print(conn.recv())
#     conn.close()
#
# if __name__ == "__main__":
#     parent_conn,child_conn = Pipe()   # 产生两个返回对象，一个是管道这一头，一个是另一头
#     p = Process(target=f,args=(child_conn,))
#     p.start()
#     parent_conn.send('father test')
#     print(parent_conn.recv())
#     print(parent_conn.recv())
#     p.join()

# a = np.array([1,2,3,4,5])
# b = np.zeros((5,3))
# c = 0.1
#
# data = [(a,b,c), (a,b,c), (a,b,c)]
# print(data)
# print("")
# print("======================================================================")
# print("")
# import pickle
#
#
# f = open("d.pkl", "wb")
# pickle.dump(data, f)
# f.close()
#
# g = open("d.pkl", "rb")
# d = pickle.load(g)
# g.close()
# print(d)
import multiprocessing
#
# # 声明一个全局变量
# share_var = ["start flag"]
#
# def sub_process(process_name):
#     # 企图像单个进程那样通过global声明使用全局变量
#     global share_var
#     share_var.append(process_name)
#     # 但是很可惜，在多进程中这样引用只能读，修改其他进程不会同步改变
#     for item in share_var:
#         print(f"{process_name}-{item}")
#     pass
#
# def main_process():
#     process_list = []
#     # 创建进程1
#     process_name = "process 1"
#     tmp_process = multiprocessing.Process(target=sub_process,args=(process_name,))
#     process_list.append(tmp_process)
#     # 创建进程2
#     process_name = "process 2"
#     tmp_process = multiprocessing.Process(target=sub_process, args=(process_name,))
#     process_list.append(tmp_process)
#     # 启动所有进程
#     for process in process_list:
#         process.start()
#     for process in process_list:
#         process.join()
#
# if __name__ == "__main__":
#     main_process()

from multiprocessing import Queue, Process
from concurrent.futures import ProcessPoolExecutor
import time

from multiprocessing import Process, Lock
import json, time, os


def search():
    time.sleep(1)  # 模拟网络io
    with open('db.txt', mode='rt', encoding='utf-8') as f:
        res = json.load(f)
        print(f'还剩{res["count"]}')


def get():
    with open('db.txt', mode='rt', encoding='utf-8') as f:
        res = json.load(f)
        # print(f'还剩{res["count"]}')
    time.sleep(1)  # 模拟网络io
    if res['count'] > 0:
        res['count'] -= 1
        with open('db.txt', mode='wt', encoding='utf-8') as f:
            json.dump(res, f)
            print(f'进程{os.getpid()} 抢票成功')
        time.sleep(1.5)  # 模拟网络io
    else:
        print('票已经售空啦!!!!!!!!!!!')


def task(lock):
    search()

    # 锁住
    lock.acquire()
    get()
    lock.release()
    # 释放锁头


if __name__ == '__main__':
    lock = Lock()  # 写在主进程是为了让子进程拿到同一把锁.
    for i in range(15):
        p = Process(target=task, args=(lock,))
        p.start()
