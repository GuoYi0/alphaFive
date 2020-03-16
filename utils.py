# -*- coding: utf-8 -*-
import numpy as np
import random
from logging import getLogger
from time import time
import pickle

logger = getLogger(__name__)
BLACK_WIN = 1
WHITE_WIN = -1
DRAW = 0


class RandomStack(object):
    def __init__(self, board_size, length=2000):
        self.data = []  # 列表每个元素是 (状态，policy，value， weight)的tuple
        self.board_size = board_size
        self.length = length
        self.white_win = 0
        self.black_win = 0
        self.data_len = []  # 装载每条数据的长度
        self.result = []  # 装载结果
        self.total_length = 0
        self.num = 0
        self.time = time()
        self.self_play_black_win = 0
        self.self_play_white_win = 0

    def save(self, s=""):
        f1 = open(f"data_buffer/data{s}.pkl", "wb")
        pickle.dump(self.data, f1)
        f1.close()

        f1 = open(f"data_buffer/data_len{s}.pkl", "wb")
        pickle.dump(self.data_len, f1)
        f1.close()

        f1 = open(f"data_buffer/result{s}.pkl", "wb")
        pickle.dump(self.result, f1)
        f1.close()

    def load(self, s=""):
        try:
            with open(f"data_buffer/data{s}.pkl", "rb") as f:
                self.data = pickle.load(f)
            with open(f"data_buffer/data_len{s}.pkl", "rb") as f:
                self.data_len = pickle.load(f)
            with open(f"data_buffer/result{s}.pkl", "rb") as f:
                self.result = pickle.load(f)
            self.white_win = self.result.count(WHITE_WIN)
            self.black_win = self.result.count(BLACK_WIN)
            print("load data successfully, with length %d" % len(self.data))
            print("black: white = %d: %d in the memory" % (self.black_win, self.white_win))
        except:
            from IPython import embed;
            embed()
            pass

    def isEmpty(self):
        return len(self.data) == 0

    def is_full(self):
        return len(self.data) >= self.length

    def push(self, data: list, result: int):
        data_len = len(data)  # 数据的长度
        self.total_length += data_len
        self.num += 1
        if result == BLACK_WIN:
            self.self_play_black_win += 1
        elif result == WHITE_WIN:
            self.self_play_white_win += 1
        if self.total_length >= 100:
            t = time()
            print("black: white = %d: %d in the memory, avg_length: %0.1f avg: %0.3fs per piece" % (
                self.black_win, self.white_win, self.total_length / self.num, (t - self.time) / self.total_length))
            print("self-play black: %d, white: %d" % (self.self_play_black_win, self.self_play_white_win))
            self.total_length = self.num = 0
            self.time = t
        # 太短小的数据就舍弃，长度为9的以0.75概率舍弃；长度为20的以0.0概率舍弃，中间线性过渡
        if random.random() <= -0.0682 * data_len + 1.364:
            return False
        self.data.extend(data)  # 数据添加进去
        self.data_len.append(data_len)  # 长度添加进去
        self.result.append(result)  # 结果添加进去
        if result == BLACK_WIN:
            self.black_win += 1
            if random.random() < (self.white_win - self.black_win) / (self.black_win * 1.3):
                self.data.extend(data)  # 数据添加进去
                self.data_len.append(data_len)  # 长度添加进去
                self.result.append(result)  # 结果添加进去
                self.black_win += 1

        elif result == WHITE_WIN:
            self.white_win += 1
            if random.random() < (self.black_win - self.white_win) / (self.white_win * 1.02):
                self.data.extend(data)  # 数据添加进去
                self.data_len.append(data_len)  # 长度添加进去
                self.result.append(result)  # 结果添加进去
                self.white_win += 1
        beyond = len(self.data) - self.length
        if beyond > 0:
            self.data = self.data[beyond:]
            while True:
                if beyond >= self.data_len[0]:  # 需要跳出去的数据长度大于第一条数据的长度
                    beyond -= self.data_len[0]
                    self.data_len.pop(0)
                    result = self.result.pop(0)
                    if result == BLACK_WIN:
                        self.black_win -= 1
                    elif result == WHITE_WIN:
                        self.white_win -= 1
                else:
                    self.data_len[0] -= beyond
                    break
        return True

    def get_data(self, batch_size=1):
        num = min(batch_size, len(self.data))
        idx = np.random.choice(len(self.data), size=num, replace=False)
        boards = np.empty((num, 3, self.board_size, self.board_size), dtype=np.float32)
        weights = np.empty((num,), dtype=np.float32)
        values = np.empty((num,), dtype=np.float32)
        policies = np.empty((num, self.board_size, self.board_size), dtype=np.float32)
        for i, ix in enumerate(idx):
            # 有序棋盘具有对称性，所以有旋转和翻转，共8种对称方式来进行数据增强
            state, p, la, v, w = self.data[ix]
            board = state_to_board(state, self.board_size)
            k = np.random.choice([0, 1, 2, 3])
            board = np.rot90(board, k=k, axes=(0, 1))
            p = np.rot90(p, k=k, axes=(0, 1))
            if la is not None:
                la = [la, (self.board_size - 1 - la[1], la[0]),
                      (self.board_size - 1 - la[0], self.board_size - 1 - la[1]),
                      (la[1], self.board_size - 1 - la[0])][k]
            if random.choice([1, 2]) == 1:
                board = np.flip(board, axis=0)
                p = np.flip(p, axis=0)
                if la is not None:
                    la = (self.board_size - 1 - la[0], la[1])
            boards[i] = board_to_inputs(board, last_action=la)
            weights[i] = w
            values[i] = v
            policies[i] = p
        policies = policies.reshape((num, self.board_size * self.board_size))
        return boards, weights, values, policies


def softmax(x):
    max_value = np.max(x)
    probs = np.exp(x - max_value)
    probs /= np.sum(probs)
    return probs


def board_to_state(board: np.ndarray) -> str:
    """
    由数组表示棋盘转换为字符串表示的棋盘
    :param board: 一个棋盘
    :return:
    """
    fen = ""
    h, w = board.shape
    for i in range(h):
        c = 0
        for j in range(w):
            if board[i, j] == 0:
                c += 1
            else:
                fen += chr(ord('a') + c) if c > 0 else ''
                fen += str(board[i, j] + 2)
                c = 0
        fen += chr(ord('a') + c) if c > 0 else ''
        fen += '/'
    return fen


def state_to_board(state: str, board_size: int):
    """
    根据字符串表示的state转换为棋盘。字符串中
    :param state:
    :param board_size:
    :return:
    """
    board = np.zeros((board_size, board_size), np.int8)
    i = j = 0
    for ch in state:
        if ch == '/':
            i += 1
            j = 0
        elif ch.isalpha():
            j += ord(ch) - ord('a')
        else:
            board[i][j] = int(ch) - 2
            j += 1
    return board


def is_game_over(board: np.ndarray, goal: int) -> tuple:
    """
    基于当前玩家落子前，判断当前局面是否结束，一般来说若结束且非和棋都会返回-1.0，
    因为现在轮到当前玩家落子了，但是游戏却已经结束了，结束前的最后一步一定是对手落子的，对手赢了，则返回-1
    :param board:
    :param goal:五子棋，goal就等于五
    :return:
    """
    h, w = board.shape
    for i in range(h):
        for j in range(w):
            hang = sum(board[i: min(i + goal, w), j])
            if hang == goal:
                return True, 1.0
            elif hang == -goal:
                return True, -1.0
            lie = sum(board[i, j: min(j + goal, h)])
            if lie == goal:
                return True, 1.0
            elif lie == -goal:
                return True, -1.0
            # 斜线有点麻烦
            if i <= h - goal and j <= w - goal:
                xie = sum([board[i + k, j + k] for k in range(goal)])
                if xie == goal:
                    return True, 1.0
                elif xie == -goal:
                    return True, -1.0
            if i >= goal - 1 and j <= w - goal:
                xie = sum([board[i - k, j + k] for k in range(goal)])
                if xie == goal:
                    return True, 1.0
                elif xie == -goal:
                    return True, -1.0
    if np.where(board == 0)[0].shape[0] == 0:  # 棋盘满了，和棋
        return True, 0.0
    return False, 0.0


def get_legal_actions(board: np.ndarray):
    """
    根据棋局返回所有的合法落子位置
    :param board:
    :return:
    """
    zeros = np.where(board == 0)
    return [(int(i), int(j)) for i, j in zip(*zeros)]


def board_to_inputs2(board: np.ndarray, type_=np.float32):
    # return board.astype(np.float32)
    tmp1 = np.equal(board, 1).astype(type_)
    tmp2 = np.equal(board, -1).astype(type_)
    out = np.stack([tmp1, tmp2])
    return out


def board_to_inputs(board: np.ndarray, type_=np.float32, last_action=None):
    """
    根据当前棋局和上一次落子地方，生成network的输入。
    第三个last action的channel估计可以去掉，影响很小。
    :param board:
    :param type_:
    :param last_action:
    :return:
    """
    f1 = np.where(board == 1, 1.0, 0.0)
    f2 = np.where(board == -1, 1.0, 0.0)
    # return np.stack([f1, f2], axis=0).astype(type_)
    f3 = np.zeros(shape=board.shape, dtype=np.float32)
    if last_action is not None:
        f3[last_action[0], last_action[1]] = 1.0
    inputs = np.stack([f1, f2, f3], axis=0).astype(type_)
    return inputs


def step(board: np.ndarray, action: tuple):
    """
    执行动作并翻转棋盘，保持当前选手为1代表的棋局
    :param board:
    :param action:
    :return:
    """
    board[action[0], action[1]] = 1
    return -board


def construct_weights(length: int, gamma=0.95):
    """
    :param length:
    :param gamma:
    :return:
    """
    w = np.empty((int(length),), np.float32)
    w[length - 1] = 1.0  # 最靠后的权重最大
    for i in range(length - 2, -1, -1):
        w[i] = w[i + 1] * gamma
    return length * w / np.sum(w)  # 所有元素之和为length
