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
b = np.array([[1,0], [-1, 1]])
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
a = 0.28437745
print("a {}, %.3f, ".format((1,2))%a)
