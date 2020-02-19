# -*- coding: utf-8 -*-
from gobang import GoBang as Game
from gobang import CONTINUE, WON_LOST, DRAW
import utils
import time
import config
import math
import numpy as np


class NODE(object):
    """
    结点不存储游戏画面的信息，只存储父子结点，以及进入各个子结点的动作，自然就可以唯一确定画面了
    """

    def __init__(self, parent=None, p: float = 0.1, state_value: float = 0.0):
        self.parent = parent  # 父节点
        self.counter = 0
        self.child = {}  # 子节点
        self.p = p  # 该节点被访问的先验概率，由神经网络给出
        self.state_value = state_value
        # self.is_fully_expanded = False  # 新创建的结点没有完全展开

    def no_child(self):
        return len(self.child) == 0

    def add_child(self, action: tuple, prior_p):
        self.child[action] = NODE(parent=self, p=prior_p)

    def is_expanded(self):
        return len(self.child) > 0

    def UCB_selection(self):
        self.counter += 1  # 从这个结点出发根据ucb去寻找子节点，该节点被访问了一次
        keys = list(self.child.keys())
        assert len(keys) > 0
        cubs = [self.child[key].UCB_value() for key in keys]  # 求cub值，长度为keys长度
        max_cub = max(cubs)  # 最大cub
        key_idx = np.random.choice(np.where(np.array(cubs) == max_cub)[0])
        choice_key = keys[key_idx]
        node_ = self.child[choice_key]
        return node_, choice_key

    def backup(self, value):
        self.state_value += value
        if self.parent:
            self.parent.backup(-value)

    def UCB_value(self):
        # 注意，结点的state_value是指从这个结点出发的价值。父节点（黑手）在挑选子节点（白手）的时候，希望挑选白手state_value最小的那个
        # 故下面式子取个负值，后面则选择最大的那个了
        q = -self.state_value / max(self.counter, 1)
        # 对于访问次数，分子开了根号，分母没有开根号，让访问次数的重要性渐渐降低，从而增加q的重要性
        # 访问次数越少，config.Cucb的数值适当调小
        e = self.p * math.sqrt(self.parent.counter) / (self.counter + 1)
        return q + config.Cucb * e

    def get_action_probs(self, board_size, train=True, tem=1.0):
        act_counter = [(act, node.counter) for act, node in self.child.items()]
        acts, counters = zip(*act_counter)
        if not train:
            max_counter = np.max(counters)
            act_idx = np.random.choice([idx for idx in range(len(counters)) if counters[idx] == max_counter])
            return acts[int(act_idx)]
        # 概率正比于 N^(1/t),其中t是温度项
        act_probs = utils.softmax(1.0 / tem * np.log(np.array(counters) + 1e-9))
        p = 0.75 * act_probs + 0.25 * np.random.dirichlet(0.3 * np.ones(act_probs.shape))
        action_idx = np.random.choice(len(acts), p=p)
        action = acts[action_idx]
        action = (int(action[0]), int(action[1]))
        probs = np.zeros(board_size * board_size, dtype=np.float32)
        probs[[a[0] * a[1] for a in acts]] = act_probs
        return action, probs


class MCTS(object):
    def __init__(self, board_size, network, simulation_per_step=400, goal=5, game=None):
        self.board_size = board_size
        self.network = network
        self.simulation_per_step = simulation_per_step  # 每走一步需要先模拟多少步
        self.current_node = NODE(None)
        self.game_process = game if game is not None else Game(board_size=board_size, goal=goal)  # 主游戏进程
        self.simulate_game = Game(board_size=board_size, goal=goal)  # 模拟游戏

    def run(self, train=True, tmp=1.0):
        """
        主游戏执行一次到游戏终结
        :param train:
        :return:
        """
        self.renew()  # 清空树
        terminal = CONTINUE
        game_record = []  # 记录主游戏每一步所走的action
        step = 1
        total_expand = 0
        total_steps = 0
        while terminal == CONTINUE:  # while语句每执行一次，主游戏只走一步
            # begin_time1 = int(time.time())
            # 在当前主游戏的基础上往前探索，该函数执行完毕以后，就形成了一棵以self.current_node为根节点的树，
            expand_count, steps_count = self.simulation()  # 主游戏走一步，模拟游戏要展开和走的步数,这一行最耗时
            action, distribution = self.current_node.get_action_probs(board_size=self.board_size, train=train, tem=tmp)
            terminal, state = self.game_process.step(action)
            self.MCTS_step(action)  # 树往前走一步
            game_record.append({"distribution": distribution, "action": action})
            total_expand += expand_count
            total_steps += steps_count
            step += 1
        game_record.append(terminal == DRAW)
        return game_record, total_expand / step, total_steps / step

    def simulation(self):
        """
        由于node不存储游戏画面信息，故state和node同步变换。
        该函数执行完毕以后，就形成了一棵以self.current_node为根节点的树，
        该函数，只为能让主游戏进程能走上一步
        :return:
        """
        expand_counter, steps_simulate = 0, 0  # 一次模拟展开的次数， 一次模拟探索的步数
        for _ in range(self.simulation_per_step):  # 一次深入探索
            is_fully_expanded, terminal = True, CONTINUE
            # 用主游戏当前局面初始化模拟局面
            self.simulate_game.simulate_reset(self.game_process.current_board(), self.game_process.last_action)
            current_node = self.current_node  # 指示主游戏当前节点，就地
            state = self.simulate_game.current_board()  # 模拟游戏的当前局面
            value = None
            while terminal == CONTINUE:
                steps_simulate += 1  # 模拟走一步
                if not current_node.is_expanded():
                    valid_move = utils.valid_move(state)
                    assert len(valid_move) > 0  # 因为state不是终止状态，所以有效移动的数目肯定有大于0
                    state_prob, value = self.network.eval(  # 评估叶子节点
                        utils.trans_to_input(state * self.simulate_game.current_player,
                                             player=self.simulate_game.current_player,
                                             last_action=self.simulate_game.last_action)[np.newaxis, ...])
                    expand_counter += 1
                    for move in valid_move:  # 展开
                        current_node.add_child(move, state_prob[0, move[0] * self.board_size + move[1]])
                    # current_node, action = current_node.UCB_selection()  # 选出最佳子节点，和跑到该子节点所执行的动作
                    # terminal, state = self.simulate_game.step(action)  # 执行该动作，得到子节点的信息
                    break  # 当前节点没有完全展开，就只展开，然后跳出循环
                else:  # 当前结点已经展开过，就直接选择一个动作，然后进入下一步
                    current_node, action = current_node.UCB_selection()  # 选出最佳子节点，和跑到该子节点所执行的动作
                    terminal, state = self.simulate_game.step(action)  # 执行该动作，得到子节点的信息
            # 假设现在是初始画面，黑子是先手，轮到黑子落子，player=1，局面是s1，结点是node1，
            # 然后黑手执行a1，得到结点node2，局面变为s2，player=-1，于是有node1.child[a1] = node2，现在轮到白手了
            # 若局面没有结束，则估计s2*player的价值，若这个价值很低，表示白手接下来不管怎么落子都输定了，node2会有一个很低的state_value
            # 若局面结束了，表示黑手胜利了，白手对应的node2应该有一个很低的state_value，于是用-1表示
            if terminal == WON_LOST:
                current_node.backup(-1)
            elif terminal == DRAW:
                current_node.backup(0)
            else:
                assert value is not None
                # # 落子前判断价值。
                # value = self.network.get_value(
                #     utils.trans_to_input(state * self.simulate_game.current_player,
                #                          player=self.simulate_game.current_player,
                #                          last_action=self.simulate_game.last_action)[np.newaxis, ...])
                current_node.backup(value[0])
        # 返回平均展开次数和平均行进步数
        return expand_counter, steps_simulate

    def simulation_pure_MCST(self):
        """
        由于node不存储游戏画面信息，故state和node同步变换。
        该函数执行完毕以后，就形成了一棵以self.current_node为根节点的树，
        该函数，只为能让主游戏进程能走上一步
        :return:
        """
        for _ in range(self.simulation_per_step):  # 一次深入探索
            is_fully_expanded, terminal = True, CONTINUE
            self.simulate_game.simulate_reset(self.game_process.current_board(),
                                              self.game_process.last_action)  # 用主游戏当前局面初始化模拟局面
            current_node = self.current_node  # 指示主游戏当前节点，就地
            state = self.simulate_game.current_board()  # 模拟游戏的当前局面
            while terminal == CONTINUE:
                if not current_node.is_expanded():
                    valid_move = utils.valid_move(state)
                    assert len(valid_move) > 0  # 因为state不是终止状态，所以有效移动的数目肯定有大于0
                    for move in valid_move:
                        current_node.add_child(move, 1.0)
                    current_node, action = current_node.UCB_selection()  # 选出最佳子节点，和跑到该子节点所执行的动作
                    terminal, state = self.simulate_game.step(action)  # 执行该动作，得到子节点的信息
                else:  # 当前结点已经展开过，就直接选择一个动作，然后进入下一步
                    current_node, action = current_node.UCB_selection()  # 选出最佳子节点，和跑到该子节点所执行的动作
                    terminal, state = self.simulate_game.step(action)  # 执行该动作，得到子节点的信息
            # 假设现在是初始画面，黑子是先手，轮到黑子落子，player=1，局面是s1，结点是node1，
            # 然后黑手执行a1，得到结点node2，局面变为s2，player=-1，于是有node1.child[a1] = node2，现在轮到白手了
            # 若局面没有结束，则估计s2*player的价值，若这个价值很低，表示白手接下来不管怎么落子都输定了，node2会有一个很低的state_value
            # 若局面结束了，表示黑手胜利了，白手对应的node2应该有一个很低的state_value，于是用-1表示
            if terminal == WON_LOST:
                current_node.backup(-1)
            elif terminal == DRAW:
                current_node.backup(0)

    def MCTS_step(self, action, human=False):
        if human:  # 人机交互的时候，人可能选择了一个未展开结点
            if action not in self.current_node.child:
                next_node = NODE(None)
            else:
                next_node = self.current_node.child[action]
                next_node.parent = None  # 主游戏走一步以后，不需要再回溯更新祖先结点
        else:  # 非人机交互，必须要有一个结点
            assert action in self.current_node.child
            next_node = self.current_node.child[action]
            next_node.parent = None  # 主游戏走一步以后，不需要再回溯更新祖先结点
        self.current_node = next_node

    def renew(self):
        self.current_node = NODE(None)
        self.game_process.reset()
        self.simulate_game.reset()

    def interact(self, action: tuple = None, ai: int = 0):
        """
        外界与游戏进行交互
        :param action: 输入的动作
        :param ai: 0表示人类玩家，这时候需要输入一个action；1表示纯MCST，不包含神经网络；2表示包含神经网络
        :return:
        """
        if ai == 0:
            assert action is not None
            terminal, state = self.game_process.step(action)
            self.MCTS_step(action, human=True)
            return state, terminal
        elif ai == 1:
            self.simulation_pure_MCST()
        elif ai == 2:
            _, _ = self.simulation()
        else:
            raise ValueError("Invalid ai:{}".format(ai))
        action = self.current_node.get_action_probs(board_size=self.board_size, train=False)
        terminal, state = self.game_process.step(action)
        self.MCTS_step(action, human=False)
        return state, terminal
