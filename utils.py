# -*- coding: utf-8 -*-
import numpy as np
import pickle

num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
            13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}

char2num = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
            "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20}

temperature = 1


class DistributionCalculator(object):
    def __init__(self, size):
        self.map = {}  # key 是形如(<class 'int'>, <class 'int'>)的tuple， value是访问次数counter
        self.order = []
        for i in range(size):
            for j in range(size):
                key = (int(i), int(j))
                self.order.append(key)
                self.map[key] = 0

    def push(self, key, value):
        self.map[key] = value

    def get(self, train=True):
        result = []  # 存放归一化后的结果
        choice_pool = []  # 存放访问过的对应的keys，即action
        choice_prob = []  # 存放经温度项修正以后的访问次数
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
            # move = np.random.choice(choice_pool, p=0.8 * np.array(choice_prob) + 0.2 * np.random.dirichlet(
            #     0.3 * np.ones(len(choice_prob))))
            p = 0.8 * np.array(choice_prob) + 0.2 * np.random.dirichlet(0.3 * np.ones(len(choice_prob)))
            from IPython import embed;
            embed()
            move = np.random.choice(choice_pool, p=0.8 * np.array(choice_prob) + 0.2 * np.random.dirichlet(
                0.3 * np.ones(len(choice_prob))))

        else:
            move = choice_pool[int(np.argmax(choice_prob))]
        move = (int(move[0]), int(move[1]))
        return move, result


def trans_to_input(state: np.ndarray):
    # out = state.astype(np.float32)
    tmp1 = np.equal(state, 1).astype(np.float32)
    tmp2 = np.equal(state, -1).astype(np.float32)
    out = np.stack([tmp1, tmp2])
    return out


def valid_move(state: np.ndarray):
    return [(int(index[0]), int(index[1])) for index in np.argwhere(state == 0)]


def write_file(objects, file_name):
    filewriter = open(file_name, 'wb')
    pickle.dump(objects, filewriter)
    filewriter.close()


def generate_training_data(game_record, board_size):
    """
    :param game_record: game_record.append({"distribution": distribution, "action": action})
    :param board_size:
    :return:
    """
    board = np.zeros([board_size, board_size])
    data = []
    player = 1
    if game_record[-1]:
        winner = 0
    elif len(game_record) % 2 == 0:  # 先手（黑手）赢了
        winner = 1
    else:
        winner = -1  # 后手（白手）赢了
    for i in range(len(game_record) - 1):
        action = game_record[i]['action']
        state = trans_to_input(board * player)
        data.append({"state": state, "distribution": game_record[i]['distribution'], "value": winner})
        board[action[0], action[1]] = player
        player, winner = -player, -winner
    return data


class RandomStack(object):
    def __init__(self, length=1000):
        self.state = []
        self.distrib = []
        self.winner = []
        self.length = length

    def isEmpty(self):
        return len(self.state) == 0

    def push(self, item):
        self.state.append(item["state"])
        self.distrib.append(item["distribution"])
        self.winner.append(item["value"])
        if len(self.state) >= self.length:
            self.state = self.state[1:]
            self.distrib = self.distrib[1:]
            self.winner = self.winner[1:]

    def get_data(self, batch_size=None):
        if batch_size is not None and batch_size < len(self.state):
            idx = np.random.choice(len(self.state), size=batch_size, replace=False)
            state = [self.state[i] for i in idx]
            distrib = [self.distrib[i] for i in idx]
            winner = [self.winner[i] for i in idx]
            return np.stack(state), np.stack(distrib), np.stack(winner)

        return np.stack(self.state), np.stack(self.distrib), np.stack(self.winner)
