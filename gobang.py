# -*- coding: utf-8 -*-
import numpy as np

DRAW = 0  # 和棋为0
CONTINUE = 1  # 继续下为1
WON_LOST = -1  # 输赢已定为-1


class GoBang(object):
    def __init__(self, board_size=15, goal=5):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 当前棋手。先手是1，后手是-1。黑子先下，为1；白字为-1
        self.passed = 0  # 一共下了多少步了
        self.goal = goal  # goal = 5 就是五子棋，

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 当前棋手。先手是1，后手是-1
        self.passed = 0

    def current_board(self):
        return self.board

    def simulate_reset(self, board_state: np.ndarray):
        self.board = board_state.copy()
        black_count = np.where(board_state == 1)[0].shape[0]  # 黑子为1
        white_count = np.where(board_state == -1)[0].shape[0]  # 白子为-1
        if black_count == white_count:
            self.current_player = 1
        elif black_count == white_count + 1:
            self.current_player = -1
        else:
            raise ValueError("Invalid board state")
        self.passed = black_count + white_count

    def step(self, action):
        row, col = action
        assert self.board[row, col] == 0, "Here already has a stone"
        self.board[row, col] = self.current_player
        self.current_player = -self.current_player
        self.passed += 1
        return self.check_over(action), self.board

    def check_over(self, action):
        """
        核查全部的棋局是否结束其计算量太大，就只核查action这个位置附近的就好
        和棋返回-1；定下输赢返回1，可以接着下棋返回0
        :param action:
        :return:
        """
        if self.passed == self.board_size * self.board_size:
            return DRAW  # 和棋返回0
        row, col = action
        for i in range(self.goal):
            if abs(sum(self.board[max(row - i, 0): min(row + self.goal - i, self.board_size), col])) == self.goal:
                return WON_LOST
            if abs(sum(self.board[row, max(col - i, 0): min(col + self.goal - i, self.board_size)])) == self.goal:
                return WON_LOST

            # 斜线有点麻烦
            coords = [(row - i + k, col - i + k) for k in range(self.goal) if
                      0 <= row - i + k < self.board_size and 0 <= col - i + k < self.board_size]
            if len(coords) == self.goal and abs(sum([self.board[r, c] for r, c in coords])) == self.goal:
                return WON_LOST
            coords = [(row - i + k, col + i - k) for k in range(self.goal) if
                      0 <= row - i + k < self.board_size and 0 <= col + i - k < self.board_size]
            if len(coords) == self.goal and abs(sum([self.board[r, c] for r, c in coords])) == self.goal:
                return WON_LOST
        return CONTINUE  # 可以接着下棋返回-1
