# -*- coding: utf-8 -*-
from collections import defaultdict
import utils
import numpy as np
import random
import gc


class Action(object):
    def __init__(self):
        self.n = 0  # 初始化为1，使得概率分布smooth一丢丢
        self.w = 0  # W(s, a) : total action value
        self.q = 0  # Q(s, a) = N / W : action value
        self.p = 0  # P(s, a) : prior probability


class State(object):
    def __init__(self):
        self.a = defaultdict(Action)  # key: action, value: ActionState only valid action included
        self.sum_n = 0  # visit count


class Player(object):
    def __init__(self, cfg=None, training=True, pipe=None, pv_fn=None):
        assert pipe is not None or pv_fn is not None
        self.config = cfg
        self.training = training
        # 做成这个样子而不是树状，是因为不同的action序列可能最终得到同一state，做成树状就不利于搜索信息的搜集
        self.tree = defaultdict(State)  # 一个字符串表示的状态到包含信息的状态的映射
        self.root_state = None
        self.goal = self.config.goal
        self.tau = self.config.init_temp  # 初始温度
        self.pipe = pipe  # 通信管道
        self.job_done = False
        self.pv_fn = pv_fn

    def get_init_state(self):
        """
        用一个字符串表示棋盘，从上至下从左至由编码
        黑子用3白字用1表示，空格部分用小写字母表示，a表示0个连续空格，b表示一个连续空格，以此类推
        :return:
        """
        fen = ""
        for i in range(self.config.board_size):
            fen += chr(ord("a") + self.config.board_size) + '/'
        return fen

    def reset(self, search_tree=None):
        self.tree = defaultdict(State) if search_tree is None else search_tree
        self.root_state = None
        self.tau = self.config.init_temp  # 初始温度

    def run(self, e=0.25):
        """
        对弈一局，获得一条数据，即从初始到游戏结束的一条数据
        :return:
        """
        state = self.get_init_state()
        game_over = False
        data = []  # 收集(状态，动作)二元组
        value = 0
        last_action = None
        while not game_over:
            policy, action = self.get_action(state, e, last_action)
            data.append((state, policy, last_action))  # 装初始局面不装最终局面，装的是动作执行之前的局面
            board = utils.step(utils.state_to_board(state, self.config.board_size), action)
            state = utils.board_to_state(board)
            # self.pruning_tree(board, state)  # 走完一步以后，对其他分支进行剪枝，以节约内存；注释掉，以节约时间
            game_over, value = utils.is_game_over(board, self.goal)
            # assert value != 1.0
            last_action = action

        self.reset()  # 把树重启
        turns = len(data)
        if turns % 2 == 1:
            value = -value
        weights = utils.construct_weights(turns, gamma=self.config.gamma)
        final_data = []
        for i in range(turns):
            final_data.append((*data[i], value, weights[i]))  # (状态，policy，last_action, value， weight)
            value = -value
        return final_data

    def calc_policy(self, state, e, random_a):
        """
        根据state表示的状态的状态信息来计算policy
        :param state:
        :param e:
        :param random_a: 这个参数是为了`choose_best_player.py`而设定的，是的在各个ckpt之间进行对弈的时候有一定的随机性
        在人机对弈的时候，为了让机器的落子有一定的随机性，也可以把这个变量设置为True
        :return:
        """
        node = self.tree[state]
        policy = np.zeros((self.config.board_size, self.config.board_size), np.float32)
        most_visit_count = -1
        candidate_actions = list(node.a.keys())
        policy_valid = np.empty((len(candidate_actions),), dtype=np.float32)
        for i, action in enumerate(candidate_actions):
            policy_valid[i] = node.a[action].n
            most_visit_count = node.a[action].n if node.a[action].n > most_visit_count else most_visit_count
        best_actions = [action for action in candidate_actions if node.a[action].n == most_visit_count]
        best_action = random.choice(best_actions)
        # for i, action in enumerate(candidate_actions):
        #     print(action, node.a[action].n,node.a[action].p)
        # from IPython import embed; embed()
        if not self.training and not random_a:
            return None, best_action
        if random_a:
            self.tau *= self.config.tau_decay_rate_r
        else:
            self.tau *= self.config.tau_decay_rate
        if self.tau <= 0.01:
            for a in best_actions:
                policy[a[0], a[1]] = 1.0 / len(best_actions)
            return policy, best_action
        policy_valid /= np.max(policy_valid)  # 除以最大值,再取指数，以免溢出
        policy_valid = np.power(policy_valid, 1 / self.tau)
        policy_valid /= np.sum(policy_valid)
        for i, action in enumerate(candidate_actions):
            policy[action[0], action[1]] = policy_valid[i]
        p = policy_valid
        # alphaGo这里添加了一个噪声。本project因为在探索的时候加的噪声足够多了，这里就不需要了
        # p = (1 - e) * p + e * np.random.dirichlet(0.5 * np.ones(policy_valid.shape[0]))
        # p = p / p.sum()  # 有精度损失，导致其和不是1了
        random_action = candidate_actions[int(np.random.choice(len(candidate_actions), p=p))]
        return policy, random_action

    def get_action(self, state: str, e: float=0.25, last_action: tuple=None, random_a=False):
        """
        根据state表示的棋局状态进行多次蒙特卡洛搜索以获取一个动作
        :param state: 字符串表示的当前棋局状态
        :param e: 为训练而添加的噪声系数。后来还是没有使用他
        :param last_action:
        :param random_a: 这个参数是为了`choose_best_player.py`而设定的，是的在各个ckpt之间进行对弈的时候有一定的随机性
        在人机对弈的时候，为了让机器的落子有一定的随机性，也可以把这个变量设置为True
        :return:
        """
        self.root_state = state
        # # 该节点已经被访问了sum_n次，最多访问642次好了，节约点时间
        if state not in self.tree:
            num = self.config.simulation_per_step
        else:
            num = min(self.config.simulation_per_step, self.config.upper_simulation_per_step-self.tree[state].sum_n)
        for i in range(num):
            self.MCTS_search(state, [state], last_action)
        policy, action = self.calc_policy(state, e, random_a=random_a)
        return policy, action

    def pruning_tree(self, board: np.ndarray, state: str = None):
        """
        主游戏前进一步以后，可以对树进行剪枝，只保留前进的那一步所对应的子树
        :param board:
        :param state:
        :return:
        """
        if state is None:
            state = utils.board_to_state(board)
        keys = list(self.tree.keys())
        for key in keys:
            b = utils.state_to_board(key, self.config.board_size)
            if key != state \
                    and np.all(np.where(board == 1, 1, 0) >= np.where(b == 1, 1, 0)) \
                    and np.all(np.where(board == -1, 1, 0) >= np.where(b == -1, 1, 0)):
                del self.tree[key]

    def update_tree(self, v, history: list):
        """
        回溯更新
        :param p: policy 当前局面对黑方的策略
        :param v: value, 当前局面对黑方的价值
        :param history: 包含当前局面的一个棋局，(state, action) pair
        :return:
        """
        _ = history.pop()  # 最近的棋局
        #  注意，这里并没有把v赋给当前node
        while len(history) > 0:
            action = history.pop()
            state = history.pop()
            v = -v
            node = self.tree[state]  # 状态结点
            action_state = node.a[action]  # 该状态下的action边
            action_state.n += 1
            action_state.w += v
            action_state.q = action_state.w / action_state.n

    def evaluate_and_expand(self, state: str, board: np.ndarray = None, last_action: tuple=None):
        if board is None:
            board = utils.state_to_board(state, self.config.board_size)
        data_to_send = utils.board_to_inputs(board, last_action=last_action)
        if self.pv_fn is not None:
            policy, value = self.pv_fn(data_to_send[np.newaxis, ...])
            policy, value = policy[0], value[0]
        else:
            self.pipe.send([data_to_send])
            while not self.pipe.poll():  # 等待对方处理数据，这里能收到的时候，poll()就返回true
                pass
            policy, value = self.pipe.recv()[0]  # 收回来的是一个列表，batch大小就是列表长度
        legal_actions = utils.get_legal_actions(board)
        all_p = max(sum([policy[action[0] * self.config.board_size + action[1]] for action in legal_actions]), 1e-5)
        for action in legal_actions:
            self.tree[state].a[action].p = policy[action[0] * self.config.board_size + action[1]] / all_p
        return value

    def MCTS_search(self, state: str, history: list, last_action: tuple):
        """
        以state为根节点进行MCTS搜索，搜索历史保存在histoty之中
        :param state: 一个字符串代表的当前状态，根节点
        :param history: 包含当前状态的一个列表
        :param last_action: 上一次的落子位置
        :return:
        """
        while True:
            board = utils.state_to_board(state, self.config.board_size)
            game_over, v = utils.is_game_over(board, self.goal)  # 落子前检查game over
            if game_over:
                self.update_tree(v, history=history)
                break
            if state not in self.tree:
                # 未出现过的state，则评估然后展开
                v = self.evaluate_and_expand(state, board, last_action)  # 落子前进行评估
                self.update_tree(v, history=history)
                break
            sel_action = self.select_action_q_and_u(state)  # 根据state选择一个action
            history.append(sel_action)  # 放进action
            board = utils.step(board, sel_action)
            state = utils.board_to_state(board)
            history.append(state)
            last_action = sel_action

    def select_action_q_and_u(self, state: str) -> tuple:
        """
        根据结点状态信息返回一个action
        :param state:
        :return:
        """
        node = self.tree[state]
        node.sum_n += 1  # 从这结点出发选择动作，该节点访问次数加一
        action_keys = list(node.a.keys())
        act_count = len(action_keys)
        dirichlet = np.random.dirichlet(self.config.dirichlet_alpha * np.ones(act_count))
        scores = np.empty((act_count,), np.float32)
        q_value = np.empty((act_count,), np.float32)
        counts = np.empty((act_count,), np.int32)
        for i, ac in enumerate(action_keys):
            action_state = node.a[ac]
            p_ = action_state.p  # 该动作的先验概率
            if self.training:
                # 训练时候为根节点添加较大噪声，非根节点添加较小噪声
                if self.root_state == state:
                    # simulation阶段的这个噪声可以防止坍缩
                    p_ = 0.75 * p_ + 0.25 * dirichlet[i]
                else:
                    p_ = 0.9 * p_ + 0.1 * dirichlet[i]  # 非根节点添加较小的噪声
            # else:
            #     # 给测试的时候也适当添加噪声，以便于充分搜索，和增加一点随机性。当然，这个随机性也可以在policy的概率分布中产生
            #     if self.root_state == state:
            #         # simulation阶段的这个噪声可以防止坍缩
            #         p_ = 0.85 * p_ + 0.15 * dirichlet[i]
            #     else:
            #         p_ = 0.95 * p_ + 0.05 * dirichlet[i]  # 非根节点添加较小的噪声
            scores[i] = action_state.q + self.config.c_puct * p_ * np.sqrt(node.sum_n + 1) / (1 + action_state.n)
            q_value[i] = action_state.q
            counts[i] = action_state.n
        if self.root_state == state and self.training:
            # 对于根节点，保证每个结点至少被访问两次，其中一次是展开，另一次是探索。
            # 故要求simulation_per_step >> 2*board_size*board_size才有意义
            # 这么做使得概率分布更加smooth，从而探索得更好
            no_visits = np.where(counts == 0)[0]
            if no_visits.shape[0] > 0:
                act_idx = np.random.choice(no_visits)
                return action_keys[act_idx]
            else:
                one_visits = np.where(counts == 1)[0]
                if one_visits.shape[0] > 0:
                    act_idx = np.random.choice(one_visits)
                    return action_keys[act_idx]
        max_score = np.max(scores)
        act_idx = np.random.choice([idx for idx in range(act_count) if scores[idx] == max_score])
        return action_keys[act_idx]

    def close(self):
        self.job_done = True
        del self.tree
        gc.collect()
