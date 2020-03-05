# -*- coding: utf-8 -*-
import tensorflow as tf
from functools import reduce
import numpy as np
from tensorflow.python import pywrap_tensorflow
from genData.networkAPI import NetworkAPI

DATA_FORMAT = "channels_first"


class ResNet(object):
    """
    针对当前局面进行评估
    """

    def __init__(self, board_size):
        self.graph = tf.get_default_graph()
        self.board_size = board_size
        self.units = 200
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, 3, board_size, board_size], name="inputs")
        self.winner = tf.placeholder(dtype=tf.float32, shape=[None], name="winner")
        self.distrib = tf.placeholder(dtype=tf.float32, shape=[None, board_size * board_size], name="distrib")
        self.weights = tf.placeholder(dtype=tf.float32, shape=[None], name="weights")
        self.training = tf.placeholder(dtype=tf.bool, shape=(), name="training")
        self.value = None
        self.policy = None
        self.entropy = None
        self.log_softmax = None
        self.prob = None
        self.network2()
        self.cross_entropy_loss, self.value_loss, self.total_loss = None, None, None
        self.construct_loss()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1000)
        self.api = None

    def construct_loss(self):
        x_entropy = tf.reduce_sum(tf.multiply(self.distrib, self.log_softmax), axis=1)
        self.cross_entropy_loss = tf.negative(tf.reduce_mean(x_entropy))  # 用于显示
        # cross_entropy = tf.negative(tf.reduce_mean(tf.multiply(x_entropy, self.weights)))  # 用于实际计算
        value_loss = tf.squared_difference(self.value, self.winner)
        self.value_loss = tf.reduce_mean(value_loss)
        value_loss = tf.reduce_mean(tf.multiply(value_loss, self.weights))
        L2_loss = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name and 'bn' not in v.name])
        # self.total_loss = cross_entropy + value_loss + L2_loss * 1e-5
        self.total_loss = self.cross_entropy_loss + value_loss + L2_loss * 1e-5

    def residual(self, f, name):
        # 参数量 8w
        r = f
        f = tf.layers.dense(f, self.units, activation=tf.nn.elu, name=name + "_fc1")
        f = tf.layers.dense(f, self.units, activation=None, name=name + "_fc2")
        f = tf.nn.elu(r + f)
        return f

    def network(self):
        # total params 51w
        f = self.inputs
        with tf.variable_scope("bone"):
            # 参数量，26w
            last_dim = reduce(lambda x, y: x * y, f.get_shape().as_list()[1:])
            f = tf.reshape(f, (-1, last_dim))
            f = tf.layers.dense(f, self.units, activation=tf.nn.elu, name="fc1")
            f = self.residual(f, "res1")
            f = self.residual(f, "res2")
            f = self.residual(f, "res3")
            # f = self.residual(f, "res4")

        with tf.variable_scope("value"):
            # 8w
            value = self.residual(f, name="res1")
            # value = self.residual(value, name="res2")
            # value = tf.layers.dense(value, 64, activation=tf.nn.elu, name="fc")
            self.value = tf.squeeze(tf.layers.dense(value, 1, activation=half_tanh, name="fc_v"), axis=1)

        with tf.variable_scope("policy"):
            # 18w
            policy = self.residual(f, "res1")
            policy = self.residual(policy, "res2")
            # policy = tf.layers.dense(policy, 128, activation=tf.nn.elu, name="fc")
            self.policy = tf.layers.dense(policy, self.board_size * self.board_size, activation=None, name="fc_p")

        self.log_softmax = tf.nn.log_softmax(self.policy, axis=1)
        self.entropy = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(self.policy) * self.log_softmax, axis=1))
        self.prob = tf.nn.softmax(self.policy, axis=1)

    def residual_conv(self, f, units, name):
        # params: in_channel*units*9 + units*units*9
        i = f
        f = tf.layers.conv2d(f, units, 3, padding="SAME", data_format=DATA_FORMAT,
                             name=name+"_conv1", activation=tf.nn.elu)
        f = tf.layers.conv2d(f, units, 3, padding="SAME", data_format=DATA_FORMAT, name=name + "_conv2")
        return tf.nn.elu(f + i)

    def network2(self):
        # total params 37w
        f = self.inputs
        with tf.variable_scope("bone"):
            # params: 13w
            f = tf.layers.conv2d(f, 32, 5, padding="SAME", data_format=DATA_FORMAT, name="conv1", activation=tf.nn.elu) # 800
            f = tf.layers.conv2d(f, 64, 3, padding="SAME", data_format=DATA_FORMAT, name="conv2", activation=tf.nn.elu) # 18432
            f = tf.layers.conv2d(f, 64, 3, padding="SAME", data_format=DATA_FORMAT, name="conv3", activation=tf.nn.elu)  # 36864
            f = tf.layers.conv2d(f, 128, 3, padding="SAME", data_format=DATA_FORMAT, name="conv4", activation=tf.nn.elu) # 73728

        with tf.variable_scope("value"):
            # 6688
            v = tf.layers.conv2d(f, 32, 1, padding="VALID", data_format=DATA_FORMAT, name="conv1", activation=tf.nn.elu)  # 4096
            last_dim = reduce(lambda x, y: x * y, v.get_shape().as_list()[1:])
            v = tf.reshape(v, (-1, last_dim))
            self.value = tf.squeeze(tf.layers.dense(v, 1, activation=tf.nn.tanh, name="fc3"), axis=1)  # 2592

        with tf.variable_scope("policy"):
            # 24w
            p = tf.layers.conv2d(f, 64, 1, padding="VALID", data_format=DATA_FORMAT, name="conv1", activation=tf.nn.elu) # 8192
            p = tf.layers.conv2d(p, 16, 3, padding="VALID", data_format=DATA_FORMAT, name="conv2", activation=tf.nn.elu) # 9216
            last_dim = reduce(lambda x, y: x * y, p.get_shape().as_list()[1:])
            p = tf.reshape(p, (-1, last_dim))
            self.policy = tf.layers.dense(p, self.board_size * self.board_size, activation=None, name="fc_p")
        self.log_softmax = tf.nn.log_softmax(self.policy, axis=1)
        self.entropy = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(self.policy) * self.log_softmax, axis=1))
        self.prob = tf.nn.softmax(self.policy, axis=1)

    def eval(self, inputs):
        """
        把一个eval函数拆分成下面两个get_prob和get_value， 要调用的时候分开分别调用，会快很多
        :param inputs:
        :return:
        """
        prob, value_ = self.sess.run([self.prob, self.value], feed_dict={self.inputs: inputs, self.training: False})
        return prob, value_

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
            try:
                self.saver.restore(self.sess, ckpt_path)
            except:
                raise FileNotFoundError("Could not find old network weights")

    def get_pipes(self, config, reload=True):
        """
        预测的时候，networkAPI只有一个线程，但有多个管道
        :param config:
        :param reload:
        :return:
        """
        if self.api is None:
            self.api = NetworkAPI(config, self)
            self.api.start(reload)  # 开启一个线程
        return self.api.get_pipe(reload)  # 开启一个管道，返回管道的另一端

    def load_pretrained(self, data_path):
        reader = pywrap_tensorflow.NewCheckpointReader(data_path)
        var = reader.get_variable_to_shape_map()
        load_sucess, load_ignore = [], []
        with tf.variable_scope("", reuse=True):
            for v in var:
                try:
                    value = reader.get_tensor(v)
                    self.sess.run(tf.assign(tf.get_variable(v), value))
                    load_sucess.append(v)
                except ValueError:
                    load_ignore.append(v)
                    continue
        print("loaded successed: ")
        for v in load_sucess:
            print(v)
        print("=" * 80)
        print("missed:")
        for v in load_ignore:
            print(v)

    def close(self):
        self.sess.close()
        self.api.close()


def half_tanh(x):
    # 让tanh函数平滑一点，有点类似学习率降低了0.5
    return tf.nn.tanh(x / 2)


def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)
