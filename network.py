# -*- coding: utf-8 -*-
import tensorflow as tf
from functools import reduce
import numpy as np
DATA_FORMAT = "channels_first"


class ResNet(object):
    """
    输入是落子前的局面，输出policy是在每个地方落子的会获胜的概率，输出value是基于当前落子的选手的价值。
    黑子是先手，落子1，白字是后手，落子-1，局面是state。
    假设现在轮到黑子下，player=1，输入是state，网络预测的则是棋子1的价值；
    若现在轮到白字下，player=-1，输入则是-state，网络预测的依然是棋子1的价值。

    只对初始局面评估，不对最终局面评估
    """
    def __init__(self, board_size):
        self.board_size = board_size
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2, board_size, board_size])
        self.winner = tf.placeholder(dtype=tf.float32, shape=[None])
        self.distrib = tf.placeholder(dtype=tf.float32, shape=[None, board_size*board_size])
        self.value = None
        self.policy = None
        self.entropy = None
        self.network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100)

    def network(self):
        f = self.inputs
        with tf.variable_scope("bone"):
            f = tf.layers.conv2d(f, 32, 3, padding="SAME", data_format=DATA_FORMAT, name="conv1", activation=tf.nn.elu)
            f = tf.layers.conv2d(f, 64, 3, padding="SAME", data_format=DATA_FORMAT, name="conv2", activation=tf.nn.elu)
            f = tf.layers.conv2d(f, 128, 3, padding="SAME", data_format=DATA_FORMAT, name="conv3", activation=tf.nn.elu)

        with tf.variable_scope("value"):
            v = tf.layers.conv2d(f, 32, 1, padding="VALID", data_format=DATA_FORMAT, name="conv1", activation=tf.nn.elu)
            last_dim = reduce(lambda x, y: x*y, v.get_shape().as_list()[1:])
            v = tf.reshape(v, (-1, last_dim))
            v = tf.layers.dense(v, 256, activation=tf.nn.elu, name="fc1")
            v = tf.layers.dense(v, 64, activation=tf.nn.elu, name="fc2")
            self.value = tf.layers.dense(v, 1, activation=tf.nn.tanh, name="fc3")

        with tf.variable_scope("policy"):
            p = tf.layers.conv2d(f, 32, 1, padding="VALID", data_format=DATA_FORMAT, name="conv1", activation=tf.nn.elu)
            last_dim = reduce(lambda x, y: x * y, p.get_shape().as_list()[1:])
            p = tf.reshape(p, (-1, last_dim))
            self.policy = tf.layers.dense(p, self.board_size*self.board_size, activation=None, name="fc1")
            self.entropy = -tf.reduce_sum(tf.nn.softmax(self.policy) * tf.nn.log_softmax(self.policy, axis=1), axis=-1)

    def eval(self, inputs):
        """
        把一个eval函数拆分成下面两个get_prob和get_value， 要调用的时候分开分别调用，会快很多
        :param inputs:
        :return:
        """
        prob = tf.nn.softmax(self.policy, axis=1)
        prob_, value_ = self.sess.run([prob, self.value], feed_dict={self.inputs: inputs})
        return prob_, value_

    def get_prob(self, inputs):
        """
        网络搭建好了以后就不要再添加结点了，不然会慢很多。所以最好先求出policy，再用numpy进行softmax
        :param inputs:
        :return:
        """
        # prob = tf.nn.softmax(self.policy, axis=1)  # 这个写法是不好了，因为这个函数每次调用，都会往gpu增加结点
        policy = self.sess.run(self.policy, feed_dict={self.inputs: inputs})
        return softmax(policy)

    def get_value(self, inputs):
        value_ = self.sess.run(self.value, feed_dict={self.inputs: inputs})
        return value_

    def restore(self, ckpt_path):
        checkpoint = tf.train.get_checkpoint_state(ckpt_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            raise FileNotFoundError("Could not find old network weights")



def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)
