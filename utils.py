# -*- coding: utf-8 -*-
import numpy as np

num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
            13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}

char2num = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
            "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20}

temperature = 1


class DistributionCalculator(object):
    def __init__(self, size):
        self.map = {}
        self.order = []
        for i in range(size):
            for j in range(size):
                name = num2char[i] + num2char[j]
                self.order.append(name)
                self.map[name] = 0

    def push(self, key, value):
        self.map[key] = value

    def get(self, train=True):
        result = []
        choice_pool = []
        choice_prob = []
        for key in self.order:
            if self.map[key] != 0:
                choice_pool.append(key)
                tmp = np.float_power(self.map[key], 1 / temperature)
                choice_prob.append(tmp)
                result.append(tmp)
                self.map[key] = 0
            else:
                result.append(0)

        he = sum(result)
        for i in range(len(result)):
            if result[i]:
                result[i] = result[i] / he
        choice_prob = [choice / he for choice in choice_prob]
        if train:
            move = np.random.choice(choice_pool, p=0.8 * np.array(choice_prob) + 0.2 * np.random.dirichlet(
                0.3 * np.ones(len(choice_prob))))
        else:
            move = choice_pool[np.argmax(choice_prob)]
        return move, result


def trans_to_input(state: np.ndarray):
    # out = state.astype(np.float32)
    tmp1 = np.equal(state, 1).astype(np.float32)
    tmp2 = np.equal(state, -1).astype(np.float32)
    out = np.stack([tmp1, tmp2])
    return out


def valid_move(state: np.ndarray):
    return list(tuple(index) for index in np.argwhere(state == 0))
