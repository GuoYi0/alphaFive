# -*- coding: utf-8 -*-
from gobang import GoBang as Game
from gobang import CONTINUE, WON_LOST, DRAW
import utils
import time
import config
import math
from main import BOARD_SIZE


distrib_calculator = utils.DistributionCalculator(config.board_size)


class NODE(object):
    """
    结点不存储游戏画面的信息，只存储父子结点，以及进入各个子结点的动作，自然就可以唯一确定画面了
    """
    def __init__(self, parent, player, p=0.0, action_value=0.0):
        self.parent = parent  # 父节点
        self.counter = 0
        self.child = {}  # 子节点
        self.player = player
        self.p = p  # 该节点被访问的先验概率，由神经网络给出
        self.action_value = action_value
        # self.is_fully_expanded = False  # 新创建的结点没有完全展开

    def no_child(self):
        return len(self.child) == 0

    def add_child(self, action: tuple, prior_p):
        self.child[action] = NODE(parent=self, player=-self.player, p=prior_p)

    def is_expanded(self):
        return len(self.child) > 0

    def UCB_selection(self):
        self.counter += 1  # 从这个结点出发去寻找子节点，该节点被访问了一次
        keys = list(self.child.keys())
        assert len(keys) > 0
        key = keys[0]
        max_ucb = self.child[key].UCB_value()
        max_key = key
        for key in keys[1:]:
            ucb = self.child[key].UCB_value()
            if ucb > max_ucb:
                max_ucb = ucb
                max_key = key
        node_ = self.child[max_key]
        return node_, max_key

    def backup(self, value):
        self.action_value += value
        if self.parent:
            self.parent.backup(-value)

    def UCB_value(self):
        q = self.action_value / max(self.counter, 1)
        e = self.p * math.sqrt(math.log(self.parent.counter) / max(self.counter, 1))
        return q + config.Cucb * e

    def get_distribution(self, train=True):  ## used to get the step distribution of current
        for key in self.child.keys():
            distrib_calculator.push(key, self.child[key].counter)
        return distrib_calculator.get(train=train)


class MCTS(object):
    def __init__(self, board_size, network, simulation_per_step=400):
        self.board_size = board_size
        self.network = network
        self.simulation_per_step = simulation_per_step  # 没走一步需要先模拟多少步
        self.current_node = NODE(None, 1)
        self.game_process = Game(board_size=board_size)  # 主游戏进程
        self.simulate_game = Game(board_size=board_size)  # 模拟游戏
        self.distribution_calculater = utils.DistributionCalculator(self.board_size)  # 暂时不知干嘛的

    def run(self, train=True):
        terminal = CONTINUE
        game_record = []
        begin_time = int(time.time())
        step = 1
        total_eval = 0
        total_step = 0
        while terminal == CONTINUE:
            begin_time = int(time.time())
            # 在当前主游戏的基础上往前探索
            avg_expand, avg_s_per_step = self.simulation()
            action, distribution = self.current_node.get_distribution(train=train)


    def simulation(self):
        """
        由于node不存储游戏画面信息，故state和node同步变换。
        该函数执行完毕以后，就形成了一棵以self.current_node为根节点的树，
        :return:
        """
        expand_counter, step_per_simulate = 0, 0  # 一次模拟展开的次数， 一次模拟探索的步数
        for _ in range(self.simulation_per_step):  # 一次深入探索
            is_fully_expanded, terminal = True, CONTINUE
            self.simulate_game.simulate_reset(self.game_process.current_board())  # 用主游戏当前局面初始化模拟局面
            current_node = self.current_node  # 指示当前节点
            state = self.simulate_game.current_board()  # 模拟游戏的当前局面
            while terminal == CONTINUE:
                if not current_node.is_expanded():
                    valid_move = utils.valid_move(state)
                    assert len(valid_move) > 0  # 因为state不是终止状态，所以有效移动的数目肯定有大于0
                    state_prob, _ = self.network.eval(
                        utils.trans_to_input(state * self.simulate_game.current_player))
                    expand_counter += 1
                    for move in valid_move:
                        current_node.add_child(move, state_prob[0, move[0] * self.board_size + move[1]])
                    current_node, action = current_node.UCB_selection()  # 选出最佳子节点，和跑到该子节点所执行的动作
                    terminal, state = self.simulate_game.step(action)  # 执行该动作，得到子节点的信息
                    break  # 当前节点没有完全展开，就只展开，然后跳出循环
                else:  # 当前结点已经展开过，就直接选择一个动作，然后进入下一步
                    current_node, action = current_node.UCB_selection()  # 选出最佳子节点，和跑到该子节点所执行的动作
                    terminal, state = self.simulate_game.step(action)  # 执行该动作，得到子节点的信息

            if terminal == WON_LOST:
                current_node.backup(1)
            elif terminal == DRAW:
                current_node.backup(0)
            else:
                assert not is_fully_expanded
                _, value = self.network.eval(utils.trans_to_input(state * self.simulate_game.current_player))
                current_node.backup(value[0])
        # 返回平均展开次数和平均行进步数
        return expand_counter / self.simulation_per_step, step_per_simulate / self.simulation_per_step
